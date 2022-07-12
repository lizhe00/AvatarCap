import math
import gc
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob
import os
import trimesh
import trimesh.proximity
import cv2 as cv
from pytorch3d.ops.knn import knn_points, knn_gather
import scipy.io as sio
import yaml

import config
from dataset.smpl import SmplModel, smpl_params
from utils.nerf_util import sample_ray_h36m


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


class AvatarCapDataset(Dataset):
    def __init__(self, data_dir, training = False):
        super(AvatarCapDataset, self).__init__()
        self.data_dir = data_dir
        self.training = training

        self.data_config = yaml.load(open(self.data_dir + '/dataConfig.yaml', encoding = 'UTF-8'), Loader=yaml.FullLoader)

        self.smpl_pose_list = sorted(glob.glob(os.path.join(self.data_dir, 'smpl/pose_*.txt')))

        self.data_type = self.data_config.get('data_type', 'synthetic')
        if self.data_type == 'synthetic':
            print('# Synthetic data')
            self.color_img_list = sorted(glob.glob(os.path.join(self.data_dir, 'imgs/*/color_view_*.jpg')))
            self.depth_img_list = sorted(glob.glob(os.path.join(self.data_dir, 'imgs/*/depth_view_*.png')))
            self.mask_img_list = sorted(glob.glob(os.path.join(self.data_dir, 'imgs/*/mask_view_*.png')))
        elif self.data_type == 'real':
            print('# Real data')
            self.color_img_list = sorted(glob.glob(os.path.join(self.data_dir, 'imgs/color/color_*.jpg')))
            self.mask_img_list = sorted(glob.glob(os.path.join(self.data_dir, 'imgs/mask/mask_*.png')))
        else:
            raise ValueError('Invalid data type!')

        self.img_num_per_pose = len(self.color_img_list) // len(self.smpl_pose_list)
        if self.img_num_per_pose == 0:
            self.img_num_per_pose = 1
        print('# Each pose contains %d view images' % self.img_num_per_pose)

        self.start_data_idx = int(os.path.basename(self.smpl_pose_list[0]).replace('pose_', '').replace('.txt', ''))
        print('# Start data index: %d' % self.start_data_idx)

        # load smpl shape
        self.smpl_shape = np.loadtxt(os.path.join(self.data_dir, 'smpl/shape.txt'))

        # init canonical smpl
        self.cano_smpl_pose = np.zeros(75, dtype = np.float32)
        self.cano_smpl_pose[3+3*1+2] = math.radians(25)
        self.cano_smpl_pose[3+3*2+2] = math.radians(-25)
        self.cano_smpl = SmplModel(self.cano_smpl_pose, self.smpl_shape)
        self.cano_smpl_center = 0.5 * (self.cano_smpl.posed_vertices.min(0) + self.cano_smpl.posed_vertices.max(0))
        self.cano_smpl_center = torch.from_numpy(self.cano_smpl_center).to(torch.float32)
        self.cano_smpl_jnts = torch.from_numpy(self.cano_smpl.posed_joints).to(torch.float32)

        self.cano_smpl_v = torch.from_numpy(self.cano_smpl.posed_vertices).to(torch.float32)
        self.inv_cano_jnt_mats = torch.from_numpy(np.linalg.inv(self.cano_smpl.jnt_affine_mats)).to(torch.float32)

        # init positional map of posed SMPL
        self.pos_map_name = self.data_config.get('pos_map_name', 'cano')
        self.pos_map_res = self.data_config.get('pos_map_res', 256)
        if self.pos_map_name == 'cano':
            self.cano2posmap_jnt_mats = torch.eye(4, dtype = torch.float32)[None].repeat(smpl_params.joint_num, 1, 1)
        elif self.pos_map_name == 'A':
            pos_map_pose = np.zeros(75, np.float32)
            pos_map_pose[3 + 16 * 3 + 2] = -math.radians(60)
            pos_map_pose[3 + 17 * 3 + 2] = math.radians(60)
            pos_map_smpl = SmplModel(pos_map_pose, self.smpl_shape)
            pos_map_jnt_mats = torch.from_numpy(pos_map_smpl.jnt_affine_mats).to(torch.float32)
            self.cano2posmap_jnt_mats = torch.matmul(pos_map_jnt_mats, self.inv_cano_jnt_mats)
        elif self.pos_map_name == 'uv':
            pass  # not implemented
        else:
            raise ValueError('Invalid pos_map_name!')

        # calculate canonical bounds
        smpl_vertices = self.cano_smpl.posed_vertices
        min_xyz = np.min(smpl_vertices, axis = 0)
        max_xyz = np.max(smpl_vertices, axis = 0)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = np.stack([min_xyz, max_xyz], axis = 0).astype(np.float32)
        print('# Canonical volume len: {}'.format(self.cano_bounds[1] - self.cano_bounds[0]))

        # camera intrinsics
        self.K = np.identity(3, np.float32)
        self.K[0, 0] = self.data_config['camera']['fx']
        self.K[0, 2] = self.data_config['camera']['cx']
        self.K[1, 1] = self.data_config['camera']['fy']
        self.K[1, 2] = self.data_config['camera']['cy']
        self.img_w = self.data_config['camera']['img_width']
        self.img_h = self.data_config['camera']['img_height']

        # generate canonical volume points for testing
        if not training:
            vol_pts = self.generate_volume_points(self.cano_bounds, config.cfg['testing']['vol_res'])

            # select valid points by SMPL
            dist, _, _ = knn_points(vol_pts.unsqueeze(0), self.cano_smpl_v.to(vol_pts).unsqueeze(0), K = 1)
            dist = dist.squeeze()
            self.infer_pts_flag = dist < 0.1 ** 2

            self.infer_pts = vol_pts[self.infer_pts_flag]
            self.infer_pts_lbs = None

            invalid_pts = vol_pts[~self.infer_pts_flag].cpu().numpy()
            cano_smpl_trimesh = trimesh.Trimesh(self.cano_smpl.posed_vertices, smpl_params.faces, process = False, use_embree = True)
            invalid_pts_ov = cano_smpl_trimesh.contains(invalid_pts)  # [0, 1]
            invalid_pts_ov = 2. * invalid_pts_ov.astype(np.float32) - 1.  # [-1, 1]
            self.invalid_pts_ov = torch.from_numpy(invalid_pts_ov).to(torch.float32).to(config.device)

        # if using only partial training data, filter data lists
        if training:
            training_data_ids = config.cfg['training'].get('training_data_ids', None)
            if training_data_ids is not None:
                training_data_ids = np.loadtxt(training_data_ids).astype(np.int32)

                def is_in_training_data_ids(pose_path):
                    _, name_ext = os.path.split(pose_path)
                    name, _ = os.path.splitext(name_ext)
                    data_idx = int(name.replace('pose_', ''))
                    return data_idx in training_data_ids

                def is_in_training_data_ids2(img_path):
                    dir_, _ = os.path.split(img_path)
                    _, dir_ = os.path.split(dir_)
                    data_idx = int(dir_)
                    return data_idx in training_data_ids

                self.smpl_pose_list = list(filter(is_in_training_data_ids, self.smpl_pose_list))
                self.color_img_list = list(filter(is_in_training_data_ids2, self.color_img_list))
                self.depth_img_list = list(filter(is_in_training_data_ids2, self.depth_img_list))
                self.mask_img_list = list(filter(is_in_training_data_ids2, self.mask_img_list))
                print('# filtered data num: %d' % len(self.smpl_pose_list))

        # preloading training dataset (posed SMPL positional maps & sampled canonical points)
        if self.training:
            print('# Preloading all the training data...')
            self.pos_maps = []
            self.presampled_data = []
            self.data_indices = []
            for smpl_pose_file in self.smpl_pose_list:
                data_idx = int(os.path.basename(smpl_pose_file).replace('pose_', '').replace('.txt', ''))
                smpl_pos_map = cv.imread(self.data_dir + '/smpl/smpl_pos_map_%04d_%s.exr' % (data_idx, self.pos_map_name), cv.IMREAD_UNCHANGED)
                if self.pos_map_name == 'cano' or self.pos_map_name == 'A':
                    smpl_pos_map = cv.resize(smpl_pos_map, (2 * self.pos_map_res, self.pos_map_res), interpolation = cv.INTER_NEAREST)
                    smpl_pos_map = np.concatenate([smpl_pos_map[:, :self.pos_map_res, :], smpl_pos_map[:, self.pos_map_res:, :]], axis = -1)
                elif self.pos_map_name == 'uv':
                    smpl_pos_map = cv.resize(smpl_pos_map, (self.pos_map_res, self.pos_map_res), interpolation = cv.INTER_NEAREST)
                else:
                    raise ValueError('Invalid pos_map_pose_name!')
                self.pos_maps.append(smpl_pos_map)
                data = np.load(self.data_dir + '/cano_pts_ov/%03d.npz' % data_idx)
                data_ = {key: data[key].copy() for key in data.keys()}
                self.presampled_data.append(data_)
                del data
                gc.collect()

                self.data_indices.append(data_idx)

            print('# Preloading done.')

    def __len__(self):
        return len(self.smpl_pose_list) * self.img_num_per_pose

    def __getitem__(self, index):
        pose_idx = index // self.img_num_per_pose
        view_idx = index % self.img_num_per_pose

        smpl_pose_path = self.smpl_pose_list[pose_idx]

        # calculate data index
        _, name_ext = os.path.split(smpl_pose_path)
        name, _ = os.path.splitext(name_ext)
        data_idx = int(name.replace('pose_', ''))
        print('data idx: %d, view idx: %d' % (data_idx, view_idx))

        # init live smpl
        live_smpl_pose = np.loadtxt(smpl_pose_path).astype(np.float32)
        live_smpl_pose[3+22*3: 6+22*3] = 0.
        live_smpl_pose[3+23*3: 6+23*3] = 0.
        live_smpl = SmplModel(live_smpl_pose, self.smpl_shape)
        cano2live_jnt_mats = torch.matmul(torch.from_numpy(live_smpl.jnt_affine_mats).to(torch.float32), self.inv_cano_jnt_mats)
        min_xyz = live_smpl.posed_vertices.min(0) - 0.05
        max_xyz = live_smpl.posed_vertices.max(0) + 0.05
        live_bounds = np.stack([min_xyz, max_xyz], 0).astype(np.float32)

        # load smpl position map
        if self.training:
            smpl_pos_map = self.pos_maps[pose_idx].copy()
        else:
            smpl_pos_map_path = self.data_dir + '/smpl/smpl_pos_map_%04d_%s.exr' % (data_idx, self.pos_map_name)
            if not os.path.exists(smpl_pos_map_path):
                smpl_pos_map_path = self.data_dir + '/smpl/smpl_pos_map_%04d.exr' % data_idx
            smpl_pos_map = cv.imread(smpl_pos_map_path, cv.IMREAD_UNCHANGED)
            smpl_pos_map = cv.resize(smpl_pos_map, (2 * self.pos_map_res, self.pos_map_res), interpolation = cv.INTER_NEAREST)
            smpl_pos_map = np.concatenate([smpl_pos_map[:, :self.pos_map_res, :], smpl_pos_map[:, self.pos_map_res:, :]], axis = -1)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))

        # load image
        if self.training:
            color_img = cv.imread(self.color_img_list[index], cv.IMREAD_UNCHANGED)
            color_img = color_img.astype(np.float32) / 255.
            if len(self.mask_img_list) == 0:
                mask_img = (np.linalg.norm(color_img, axis = -1) > 0.).astype(np.uint8)
            else:
                mask_img = cv.imread(self.mask_img_list[index], cv.IMREAD_UNCHANGED)
        else:
            color_img = np.ones((self.img_h, self.img_w, 3), dtype = np.float32)
            mask_img = np.ones((self.img_h, self.img_w), dtype = np.uint8)

        # load camera extrinsics
        cam_path = os.path.join(self.data_dir + '/imgs/%03d/cams.mat' % data_idx)
        if os.path.exists(cam_path):
            cam_data = sio.loadmat(cam_path)
            cam_r = np.float32(cam_data['cam_rs'][view_idx])
            cam_t = np.float32(cam_data['cam_ts'][view_idx])
            w2c_RT = np.identity(4, dtype = np.float32)
            w2c_RT[:3, :3] = cv.Rodrigues(cam_r)[0]
            w2c_RT[:3, 3] = cam_t
        else:
            w2c_RT = np.identity(4, dtype = np.float32)

        sampled_ray_num = 1024
        rgb, body_mask, ray_o, ray_d, near, far, coord, mask_at_box \
            = sample_ray_h36m(color_img, mask_img, self.K, w2c_RT[:3, :3], w2c_RT[:3, 3:], live_bounds, sampled_ray_num, self.training)

        occupancy = mask_img[coord[:, 0], coord[:, 1]]
        if self.training and self.data_type == 'synthetic':
            depth_img = cv.imread(self.depth_img_list[index], cv.IMREAD_UNCHANGED)
            z = depth_img[coord[:, 0], coord[:, 1]] / 1000.
            x = (coord[:, 1] + 0.5 - self.K[0, 2]) * z / self.K[0, 0]
            y = (coord[:, 0] + 0.5 - self.K[1, 2]) * z / self.K[1, 1]
            depth = np.sqrt(x*x + y*y + z*z).astype(np.float32)  # for surface-guided sampling
        else:
            depth = np.zeros(occupancy.shape, dtype = np.float32)

        data_item = {
            'data_idx': data_idx,
            'view_idx': view_idx,
            'smpl_pose': torch.from_numpy(live_smpl_pose),
            'smpl_pos_map': torch.from_numpy(smpl_pos_map),
            'cano2live_jnt_mats': cano2live_jnt_mats,
            'cano2posmap_jnt_mats': self.cano2posmap_jnt_mats,
            'cano_bounds': torch.from_numpy(self.cano_bounds),
            'cano_smpl_center': self.cano_smpl_center,
            'cano_smpl_jnts': self.cano_smpl_jnts,
            'live_smpl_v': torch.from_numpy(live_smpl.posed_vertices.astype(np.float32))
        }

        nerf_data = {
            'coord': coord,
            'rgb': rgb,
            'depth': depth,
            'body_mask': body_mask,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'img_h': self.img_h,
            'img_w': self.img_w,
            'w2c_RT': w2c_RT
        }

        data_item.update(nerf_data)

        if self.training:
            sur_pnum = 5000
            vol_pnum = sur_pnum // 16

            presampled_data = self.presampled_data[pose_idx]

            sur_pts_ids = np.random.choice(a = presampled_data['sur_pts'].shape[0], size = sur_pnum, replace = False)
            vol_pts_ids = np.random.choice(a = presampled_data['vol_pts'].shape[0], size = vol_pnum, replace = False)
            sur_pts = presampled_data['sur_pts'][sur_pts_ids]
            vol_pts = presampled_data['vol_pts'][vol_pts_ids]
            sur_pts_ov = presampled_data['sur_pts_ov'][sur_pts_ids]
            vol_pts_ov = presampled_data['vol_pts_ov'][vol_pts_ids]

            cano_pts = np.concatenate([sur_pts, vol_pts], 0).astype(np.float32)
            pts_ov = np.concatenate([sur_pts_ov, vol_pts_ov], 0).astype(np.float32)

            occupancy_data = ({'cano_pts': cano_pts,
                               'cano_pts_ov': pts_ov,
                               'sur_pnum': sur_pnum,
                               'vol_pnum': vol_pnum})
        else:
            occupancy_data = {'cano_pts': self.infer_pts,
                              'valid_pts_flag': self.infer_pts_flag}

        data_item.update(occupancy_data)

        return data_item

    @staticmethod
    def generate_volume_points(bounds, testing_res = (256, 256, 256)):
        x_coords = torch.linspace(0, 1, steps = testing_res[0], dtype = torch.float32, device = config.device).detach()
        y_coords = torch.linspace(0, 1, steps = testing_res[1], dtype = torch.float32, device = config.device).detach()
        z_coords = torch.linspace(0, 1, steps = testing_res[2], dtype = torch.float32, device = config.device).detach()
        xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
        xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
        yv = torch.reshape(yv, (-1, 1))
        zv = torch.reshape(zv, (-1, 1))
        pts = torch.cat([xv, yv, zv], dim = -1)

        # transform to canonical space
        pts = pts * torch.from_numpy(bounds[1] - bounds[0]).to(pts) + torch.from_numpy(bounds[0]).to(pts)

        return pts


def to_cuda(items: dict, add_batch = False):
    """
    Update data to GPU device.
    :param items: CPU data dictionary from dataloader.
    :param add_batch: (bool) whether to add batch dimension at the first dimension for torch.Tensor object.
    :return: GPU data dictionary.
    """
    items_cuda = dict()
    for key, data in items.items():
        if isinstance(data, torch.Tensor):
            items_cuda[key] = data.to(config.device)
        elif isinstance(data, np.ndarray):
            items_cuda[key] = torch.from_numpy(data).to(config.device)
        else:
            items_cuda[key] = data
        if add_batch and isinstance(items_cuda[key], torch.Tensor):
            items_cuda[key] = items_cuda[key].unsqueeze(0)
    return items_cuda


class AvatarCapDataloader(DataLoader):
    def __init__(self, data_dir, training = False, batch_size = 4, num_workers = 0):
        self.dataset = AvatarCapDataset(data_dir, training)
        self.batch_size = batch_size

        super(AvatarCapDataloader, self).__init__(self.dataset,
                                                  batch_size = batch_size,
                                                  shuffle = training,
                                                  num_workers = num_workers,
                                                  worker_init_fn = worker_init_fn,
                                                  drop_last = True)

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import random
import numpy as np
import torch
import time
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter

from network.arch_avatar import NerfRenderer, OccupancyNet, GeoTexAvatar
from network.arch_recon import ReconNetwork
from dataset.avatarcap_dataset import AvatarCapDataloader, to_cuda

import config
import utils.lr_schedule as lr_schedule
from utils.smpl_util import smpl_util
import utils.visualize_util as visualize_util
import utils.recon_util as recon_util
from utils.renderer import Renderer
import utils.obj_io as obj_io
from normal_fusion.normal_fusion import canonicalize_normal_map, merge_normal_images, merge_normal_images_cover

import datetime

now = datetime.datetime.now()


def train_avatar():
    """
    Train the GeoTexAvatar Network for the subject with 3D scans.
    """
    cfg = config.cfg
    network = GeoTexAvatar().to(config.device)
    network.train()
    nerf_renderer = NerfRenderer(network)
    occ_net = OccupancyNet(network)

    optm = torch.optim.Adam([
        {'params': network.cano_template.parameters(), 'lr': cfg['model']['cano_template_lr']},
        {'params': network.warping_field.parameters(), 'lr': cfg['model']['warping_field_lr']}
    ])
    lr_schedule_cano_template = lr_schedule.get_learning_rate_schedules('Step', Initial = cfg['model']['cano_template_lr'], Interval = 5000, Factor = 0.5)
    lr_schedule_warping_field = lr_schedule.get_learning_rate_schedules('Step', Initial = cfg['model']['warping_field_lr'], Interval = 20000, Factor = 0.5)

    image_loss_func = torch.nn.MSELoss()
    geo_loss_func = torch.nn.L1Loss() if config.if_type == 'sdf' else torch.nn.BCELoss()
    img_loss_weight = cfg['model']['img_loss_weight']
    occ_loss_weight = cfg['model']['occ_loss_weight']
    geo_offset_reg_loss_weight = cfg['model']['geo_offset_reg_loss_weight']
    tex_offset_reg_loss_weight = cfg['model']['tex_offset_reg_loss_weight']

    print('# Network: optimizable network parameter number: %d' % sum(p.numel() for p in network.parameters() if p.requires_grad))

    os.makedirs(config.cfg['training']['net_ckpt_dir'], exist_ok = True)
    log_dir = config.cfg['training']['net_ckpt_dir'] + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_dir, exist_ok = True)
    writer = SummaryWriter(log_dir)

    if cfg['training']['net_ckpt'] is not None:
        print('# Loading pretrained network from %s' % cfg['training']['net_ckpt'])
        network.load_state_dict(torch.load(cfg['training']['net_ckpt'] + '/net.pt')['network'])
        print('# Loading optimizer from %s' % cfg['training']['net_ckpt'])
        optm.load_state_dict(torch.load(cfg['training']['net_ckpt'] + '/optm.pt')['optm'])

    # initialize dataloader
    batch_size = cfg['training'].get('batch_size', 1)
    num_workers = cfg['training'].get('num_workers', 0)
    print('# Batch size: %d' % batch_size)
    print('# Workers num: %d' % num_workers)
    loader = AvatarCapDataloader(cfg['training']['training_data_dir'], True, batch_size = batch_size, num_workers = num_workers)
    batch_num = len(loader.dataset) // batch_size

    # set auxiliary variables
    smpl_util.set_cano_smpl_vertices(loader.dataset.cano_smpl_v)

    start_epoch = cfg['training'].get('start_epoch', 0)
    end_epoch = cfg['training'].get('end_epoch', 30)

    for epoch_idx in range(start_epoch, end_epoch):
        # update lr
        lr_cano_template = lr_schedule_cano_template.get_learning_rate(epoch_idx * batch_num)
        lr_cano_template = max(5e-4, lr_cano_template)
        lr_warping_field = lr_schedule_warping_field.get_learning_rate(epoch_idx * batch_num)
        if epoch_idx < 1:
            lr_warping_field = 0.  # firstly learn a plausible template with warping field fixed to zero
        else:
            lr_warping_field = max(5e-5, lr_warping_field)
        optm.param_groups[0]['lr'] = lr_cano_template
        optm.param_groups[1]['lr'] = lr_warping_field

        epoch_losses = dict()
        time_epoch_start = time.time()
        for batch_idx, items in enumerate(loader):
            items = to_cuda(items)

            # forward
            network.warping_field.precompute_conv(items)
            occ_output = occ_net.query(items)
            nerf_output = nerf_renderer.render(items)

            # calculate losses
            img_loss = image_loss_func(nerf_output['rgb_map'], items['rgb'])
            if config.if_type == 'sdf':
                items['cano_pts_ov'] = torch.clip(items['cano_pts_ov'], -config.sdf_thres, config.sdf_thres) / config.sdf_thres
            else:
                inside_flag = items['cano_pts_ov'] > 0.
                items['cano_pts_ov'][inside_flag] = 1.
                items['cano_pts_ov'][~inside_flag] = 0.
            geo_loss = geo_loss_func(occ_output['cano_pts_ov'].squeeze(-1), items['cano_pts_ov'])
            geo_offset_reg_loss = torch.linalg.norm(occ_output['nonrigid_offset'], dim = -1).mean()
            tex_offset_reg_loss = torch.linalg.norm(nerf_output['nonrigid_offset'], dim = -1).mean()
            total_loss = img_loss_weight * img_loss + occ_loss_weight * geo_loss + geo_offset_reg_loss_weight * geo_offset_reg_loss + tex_offset_reg_loss_weight * tex_offset_reg_loss

            optm.zero_grad()
            total_loss.backward()
            optm.step()

            batch_losses = dict()
            batch_losses['tex_loss'] = img_loss.item()
            batch_losses['geo_loss'] = geo_loss.item()
            batch_losses['geo_offset_reg_loss'] = geo_offset_reg_loss.item()
            batch_losses['tex_offset_reg_loss'] = tex_offset_reg_loss.item()

            # record batch loss
            log_info = 'epoch %d, batch %d, lr: %e, %e, ' % (epoch_idx, batch_idx, lr_cano_template, lr_warping_field)
            for key in batch_losses.keys():
                log_info = log_info + ('%s: %f, ' % (key, batch_losses[key]))
                writer.add_scalar('%s/Batch' % key, batch_losses[key], epoch_idx * batch_num + batch_idx)
                if key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                else:
                    epoch_losses[key] = batch_losses[key]
            print(log_info)
            with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                fp.write(log_info + '\n')
        print('\033[1;31m This epoch costs %f secs\033[0m' % (time.time() - time_epoch_start))

        # record epoch loss
        for key in epoch_losses.keys():
            epoch_losses[key] /= batch_num
            writer.add_scalar('%s/Epoch' % key, epoch_losses[key], epoch_idx)

        # save the network every 'ckpt_interval' epochs
        if epoch_idx % cfg['training']['ckpt_interval'] == 0:
            model_folder = cfg['training']['net_ckpt_dir'] + '/epoch_%d' % epoch_idx
            os.makedirs(model_folder, exist_ok = True)
            torch.save({'network': network.state_dict()}, model_folder + '/net.pt')
            torch.save({'optm': optm.state_dict()}, model_folder + '/optm.pt')

        # save the latest network
        latest_folder = cfg['training']['net_ckpt_dir'] + '/epoch_latest'
        os.makedirs(latest_folder, exist_ok = True)
        torch.save({'network': network.state_dict()}, latest_folder + '/net.pt')
        torch.save({'optm': optm.state_dict()}, latest_folder + '/optm.pt')

    writer.close()

    if cfg['training'].get('finetune_tex', True):  # finetune texture template using a single scan
        finetune_texture_template(nerf_renderer, occ_net, loader)


def finetune_texture_template(nerf_renderer: NerfRenderer,
                              occ_net: OccupancyNet,
                              loader: AvatarCapDataloader):
    """
    Finetune the texture template for more high-quality texture.
    """
    print('# Starting finetuing the texture template...')
    cfg = config.cfg
    network = nerf_renderer.net
    nerf_renderer.net.train()

    # initial network
    network_init = GeoTexAvatar().to(config.device)
    network_init.load_state_dict(network.state_dict())
    occ_net_init = OccupancyNet(network_init)

    # freeze other nets
    for name, params in network.named_parameters():
        if not name.startswith('cano_template'):
            params.requires_grad = False

    optm = torch.optim.Adam([
        {'params': network.cano_template.parameters(), 'lr': 5e-4}
    ])
    lr_schedule_cano_template = lr_schedule.get_learning_rate_schedules('Constant', Value = 5e-4)

    image_loss_func = torch.nn.MSELoss()
    geo_loss_func = torch.nn.L1Loss()

    print('# Network: optimizable network parameter number: %d' % sum(p.numel() for p in network.parameters() if p.requires_grad))

    os.makedirs(config.cfg['training']['net_ckpt_dir'] + '/finetune_tex', exist_ok = True)
    log_dir = config.cfg['training']['net_ckpt_dir'] + '/finetune_tex/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_dir, exist_ok = True)
    writer = SummaryWriter(log_dir)

    # set auxiliary variables
    smpl_util.set_cano_smpl_vertices(loader.dataset.cano_smpl_v)

    finetune_data_idx = cfg['training'].get('finetune_tex_data_idx', 0)
    print('# Scan %d is used for finetuning the texture template' % finetune_data_idx)
    rel_idx = loader.dataset.data_indices.index(finetune_data_idx)
    assert rel_idx >= 0
    data_indices = list(range(loader.dataset.img_num_per_pose * rel_idx, loader.dataset.img_num_per_pose * (rel_idx+1)))
    batch_num = len(data_indices)

    start_epoch = 0
    end_epoch = 1000
    for epoch_idx in range(start_epoch, end_epoch):
        # update lr
        lr_cano_template = lr_schedule_cano_template.get_learning_rate(epoch_idx * batch_num)
        optm.param_groups[0]['lr'] = lr_cano_template

        epoch_losses = dict()
        time_epoch_start = time.time()
        random.shuffle(data_indices)
        for batch_idx, index in enumerate(data_indices):
            item = loader.dataset.__getitem__(index)
            items = to_cuda(item, add_batch = True)

            # forward
            if epoch_idx == 0 and batch_idx == 0:
                network.warping_field.precompute_conv(items)
            nerf_output = nerf_renderer.render(items)
            occ_output = occ_net.query(items)

            with torch.no_grad():
                if network_init.warping_field.pose_feat_map is None:
                    network_init.warping_field.precompute_conv(items)
                occ_output_init = occ_net_init.query(items)

            # calculate losses
            img_loss = image_loss_func(nerf_output['rgb_map'], items['rgb'])
            geo_loss = geo_loss_func(occ_output['cano_pts_ov'], occ_output_init['cano_pts_ov'])
            total_loss = img_loss + 0.5 * geo_loss

            optm.zero_grad()
            total_loss.backward()
            optm.step()

            batch_losses = {
                'tex_loss': img_loss.item()
                , 'geo_loss': geo_loss.item()
            }

            # record batch loss
            log_info = 'epoch %d, batch %d, lr: %e ' % (epoch_idx, batch_idx, lr_cano_template)
            for key in batch_losses.keys():
                log_info = log_info + ('%s: %f, ' % (key, batch_losses[key]))
                writer.add_scalar('%s/Batch' % key, batch_losses[key], epoch_idx * batch_num + batch_idx)
                if key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                else:
                    epoch_losses[key] = batch_losses[key]
            print(log_info)
            with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                fp.write(log_info + '\n')
        print('\033[1;31m This epoch costs %f secs\033[0m' % (time.time() - time_epoch_start))

        # record epoch loss
        for key in epoch_losses.keys():
            epoch_losses[key] /= batch_num
            writer.add_scalar('%s/Epoch' % key, epoch_losses[key], epoch_idx)

        if epoch_idx % 20 == 0 and epoch_idx > 0:
            model_folder = cfg['training']['net_ckpt_dir'] + '/finetune_tex/epoch_%d' % epoch_idx
            os.makedirs(model_folder, exist_ok = True)
            torch.save({'network': network.state_dict()}, model_folder + '/net.pt')
            torch.save({'optm': optm.state_dict()}, model_folder + '/optm.pt')

    writer.close()


def run_avatarcap(w_recon = False, save_avatar_mesh = False, save_final_mesh = False, w_nerf = False, frame_idx = None, view_idx = 0, interval = 1, integrate_manner = 'merge'):
    """
    Test AvatarCap (the whole capture pipeline) or only the GeoTexAvatar module.
    Reconstructed and animated results will be saved in cfg['testing]['output_dir'],
    and "cano_avatar", "live_avatar" and "live_recon" folders contain rendered images of
    animated canonical avatar, live avatar and final reconstructed mesh, respectively.
    :param w_recon: True: test AvatarCap, False: test only GeoTexAvatar.
    :param save_avatar_mesh: True: save animated results of GeoTexAvatar as ply file.
    :param save_final_mesh: True: save reconstructed results of AvatarCap as ply file.
    :param w_nerf: True: get textured results, False: only geometry.
    :param frame_idx: the frame indices that you want to reconstruct or animate.
    :param view_idx: only useful for synthetic multi-view data, default: 0.
    :param interval: frame interval for reconstruction or animation.
    :param integrate_manner: the method of fusing image and avatar normals.
    """
    cfg = config.cfg
    os.makedirs(cfg['testing']['output_dir'], exist_ok = True)
    os.makedirs(cfg['testing']['output_dir'] + '/cano_avatar', exist_ok = True)
    os.makedirs(cfg['testing']['output_dir'] + '/live_avatar', exist_ok = True)
    os.makedirs(cfg['testing']['output_dir'] + '/live_recon', exist_ok = True)

    network = GeoTexAvatar().to(config.device)
    network.eval()
    occ_net = OccupancyNet(network)
    nerf_renderer = NerfRenderer(network)
    recon_net = ReconNetwork().to(config.device)

    if cfg['testing']['net_ckpt'] is not None:
        print('# Loading GeoTexAvatar network from %s' % cfg['testing']['net_ckpt'])
        net_data = torch.load(cfg['testing']['net_ckpt'] + '/net.pt')
        network.load_state_dict(net_data['network'])

    net_ckpt_finetuned = cfg['testing'].get('net_ckpt_finetuned', None)
    if net_ckpt_finetuned is not None:
        print('# Loading finetuned GeoTexAvatar network from %s' % net_ckpt_finetuned)
        net_data = torch.load(net_ckpt_finetuned + '/net.pt')
        network_finetuned = GeoTexAvatar().to(config.device)
        network_finetuned.eval()
        network_finetuned.load_state_dict(net_data['network'])
        nerf_renderer.net = network_finetuned

    recon_net_ckpt = cfg['testing'].get('recon_net_ckpt', None)
    if recon_net_ckpt is not None:
        print('# Loading reconstruction network from %s' % recon_net_ckpt)
        net_data = torch.load(recon_net_ckpt + '/recon_net.pt')
        recon_net.load_state_dict(net_data['network'])

    # init data loader
    loader = AvatarCapDataloader(cfg['testing']['testing_data_dir'], False, batch_size = 1, num_workers = 0)
    data_num = len(loader.dataset) // loader.dataset.img_num_per_pose
    print('# Data num: %d' % data_num)

    # init renderers
    cam = loader.dataset.data_config['camera']
    phong_renderer = Renderer(512, 512, shader_name = 'phong_geometry', bg_color = (1, 1, 1), window_name = 'Phong')
    normal_renderer = Renderer(512, 512, shader_name = 'vertex_attribute', window_name = 'Normal')
    position_renderer = Renderer(cam['img_width'], cam['img_height'], shader_name = 'position', window_name = 'Position')
    front_mv, back_mv = None, None

    # set auxiliary variables
    smpl_util.set_cano_smpl_vertices(loader.dataset.cano_smpl_v)
    cano_smpl_center = 0.5 * (smpl_util.cano_smpl_vertices.max(0)[0] + smpl_util.cano_smpl_vertices.min(0)[0]).cpu().numpy()
    cano_smpl_center_for_render = cano_smpl_center.copy()

    if frame_idx is None:
        inferred_list = list(range(0, data_num, interval))
    elif isinstance(frame_idx, int):
        inferred_list = [frame_idx - loader.dataset.start_data_idx]
    elif isinstance(frame_idx, list):
        inferred_list = (np.array(frame_idx, np.int32) - loader.dataset.start_data_idx).tolist()
    else:
        raise TypeError('Invalid frame_idx!')

    for i in inferred_list:
        item_idx = i * loader.dataset.img_num_per_pose + view_idx
        item = loader.dataset.__getitem__(item_idx)
        items = to_cuda(item, add_batch = True)
        data_idx = items['data_idx']

        """
        1. Generate geometric avatar in canonical space
        """
        occ_volume = torch.zeros(cfg['testing']['vol_res'], dtype = torch.float32, device = config.device).reshape(-1)
        with torch.no_grad():
            network.warping_field.precompute_conv(items)
            output = occ_net.query(items)

        occ_volume[items['valid_pts_flag'][0]] = output['cano_pts_ov'][0, :, 0]
        occ_volume[~items['valid_pts_flag'][0]] = loader.dataset.invalid_pts_ov
        occ_volume = occ_volume.reshape(cfg['testing']['vol_res'])

        bounds = items['cano_bounds'][0].cpu().numpy()
        vertices, faces, normals = recon_util.recon_mesh(occ_volume, cfg['testing']['vol_res'], bounds, iso_value = config.iso_value)

        front_avatar_normal, back_avatar_normal = visualize_util.render_cano_mesh(normal_renderer, vertices, normals, faces, cano_smpl_center_for_render)

        # save results: canonical avatar
        cano_img_front, cano_img_back = visualize_util.render_cano_mesh(phong_renderer, vertices, normals, faces, cano_smpl_center_for_render)
        cano_img = np.concatenate([cano_img_front, cano_img_back], 1)
        cano_img = (255 * cano_img).astype(np.uint8)
        cv.imwrite('%s/%04d.jpg' % (cfg['testing']['output_dir'] + '/cano_avatar', data_idx), cano_img)

        # store cano avatar mesh
        cano_avatar_mesh = {'v': vertices.copy(),
                            'vn': normals.copy(),
                            'f': faces.copy()}

        # skinning to live space
        vertices_gpu = torch.from_numpy(vertices).to(torch.float32).to(config.device)
        normals_gpu = torch.from_numpy(normals).to(torch.float32).to(config.device)
        lbs = smpl_util.calculate_lbs(vertices_gpu[None, ...])
        live_vertices, vert_mats = smpl_util.skinning(vertices_gpu[None, ...], lbs, items['cano2live_jnt_mats'], True)
        live_vertices = live_vertices[0].cpu().numpy()
        vert_mats = vert_mats[0]
        live_normals = torch.einsum('vij,vj->vi', vert_mats[:, :3, :3], normals_gpu).cpu().numpy()

        # store live avatar mesh
        live_avatar_mesh = {'v': live_vertices.copy(),
                            'vn': live_normals.copy(),
                            'f': faces.copy()}

        # save results: live avatar
        if front_mv is None or back_mv is None:
            front_mv = visualize_util.calc_front_mv(live_vertices, rot_x_angle = -0.15)
            back_mv = visualize_util.calc_back_mv(live_vertices, rot_x_angle = -0.15)
        live_img_front, live_img_back = visualize_util.render_live_mesh(phong_renderer, live_vertices, live_normals, faces, front_mv = front_mv, back_mv = back_mv)
        live_img = np.concatenate([live_img_front, live_img_back], 1)
        live_img = (255 * live_img).astype(np.uint8)
        cv.imwrite('%s/%04d.jpg' % (cfg['testing']['output_dir'] + '/live_avatar', data_idx), live_img)

        """
        2. Canonical normal fusion (if w_recon == True)
        """
        if w_recon:
            if loader.dataset.data_config['data_type'] == 'synthetic':
                inferred_normal = cv.imread(loader.dataset.data_dir + '/imgs/%03d/normal_view_%03d.exr' % (data_idx, view_idx), cv.IMREAD_UNCHANGED)  # synthetic multi-view data
            elif loader.dataset.data_config['data_type'] == 'real':
                inferred_normal = cv.imread(loader.dataset.data_dir + '/imgs/normal/normal_%04d.exr' % data_idx, cv.IMREAD_UNCHANGED)  # monocular data
            else:
                raise ValueError('Invalid data type!')
            front_image_normal, back_image_normal = canonicalize_normal_map(position_renderer, normal_renderer, vertices, live_vertices, faces, inferred_normal, vert_mats,
                                                                            mv = items['w2c_RT'][0].cpu().numpy(), fx = cam['fx'], fy = cam['fy'], cx = cam['cx'], cy = cam['cy'],
                                                                            cano_smpl_center = cano_smpl_center_for_render)

            if integrate_manner == 'merge':
                neck_vert = smpl_util.cano_smpl_vertices[3068].cpu().numpy() - cano_smpl_center_for_render
                neck_y = int((1. - neck_vert[1]) / 2. * 512)
                neck_x = int((neck_vert[0] - 1) / 2. * 512)
                front_merged_normal = merge_normal_images(front_avatar_normal, front_image_normal, iter_num = 100, neck_xy = (neck_x, neck_y))
            elif integrate_manner == 'cover':
                front_merged_normal = merge_normal_images_cover(front_avatar_normal, front_image_normal)
            else:
                raise ValueError('Invalid integration manner!')

            # suppose that the performer is facing the camera, i.e., the visible regions are almost in the front of the body
            back_merged_normal = back_avatar_normal

            items['front_normal'] = torch.from_numpy(front_merged_normal.transpose(2, 0, 1)).to(torch.float32).to(config.device)[None]
            items['back_normal'] = torch.from_numpy(back_merged_normal.transpose(2, 0, 1)).to(torch.float32).to(config.device)[None]

            """
            3. Reconstruction network (if w_recon == True)
            """
            occ_volume = occ_volume.reshape(-1).fill_(0.)
            with torch.no_grad():
                output = recon_net.infer(items)

            occ_volume[items['valid_pts_flag'][0]] = output[0]
            occ_volume[~items['valid_pts_flag'][0]] = loader.dataset.invalid_pts_ov
            vertices, faces, normals = recon_util.recon_mesh(occ_volume, cfg['testing']['vol_res'],
                                                             items['cano_bounds'][0].cpu().numpy())

            vertices = torch.from_numpy(vertices).to(torch.float32).to(config.device)
            normals = torch.from_numpy(normals).to(vertices)

            # skinning to live space
            lbs = smpl_util.calculate_lbs(vertices[None, ...])
            live_vertices = smpl_util.skinning(vertices[None, ...], lbs, items['cano2live_jnt_mats'])[0].cpu().numpy()
            live_normals = smpl_util.skinning_normal(normals[None, ...], lbs, items['cano2live_jnt_mats'])[0].cpu().numpy()

            # store live recon mesh
            live_recon_mesh = {'v': live_vertices.copy(),
                               'vn': live_normals.copy(),
                               'f': faces.copy()}

        """
        4. Evaluate vertex colors in canonical space (if w_nerf == True)
        """

        if w_nerf:  # generate vertex color from the texture template
            with torch.no_grad():
                # integrate on the avatar surface
                vertices_avatar = torch.from_numpy(cano_avatar_mesh['v']).to(torch.float32).to(config.device)
                normals_avatar = torch.from_numpy(cano_avatar_mesh['vn']).to(torch.float32).to(config.device)
                items['ray_o'] = (vertices_avatar + normals_avatar)[None]
                items['ray_d'] = -normals_avatar[None]
                items['depth'] = torch.ones((1, vertices_avatar.shape[0])).to(vertices_avatar)
                items['near'] = items['depth'] - 0.05
                items['far'] = items['depth'] + 0.05
                items['occupancy'] = items['depth'].clone()
                nerf_renderer.net.warping_field.precompute_conv(items)
                nerf_output = nerf_renderer.render(items, pts_space = 'cano', near_dist = 0.02, far_dist = 0.05)
                color_avatar = nerf_output['rgb_map'][0][:, [2, 1, 0]]
                live_avatar_mesh['vc'] = color_avatar.cpu().numpy()

            if w_recon:
                # nn for recon vertices
                from pytorch3d.ops import knn_points, knn_gather
                _, indices, _ = knn_points(vertices[None], vertices_avatar[None])
                color_recon = knn_gather(color_avatar[None], indices)[0, :, 0]
                live_recon_mesh['vc'] = color_recon.cpu().numpy()
        else:
            live_avatar_mesh['vc'] = None
            if w_recon:
                live_recon_mesh['vc'] = None

        if save_avatar_mesh:
            obj_io.save_mesh_as_ply('%s/%04d_avatar.ply' % (cfg['testing']['output_dir'], data_idx),
                                    live_avatar_mesh['v'], live_avatar_mesh['f'], live_avatar_mesh['vn'], live_avatar_mesh['vc'])

        if w_recon:
            if save_final_mesh:
                obj_io.save_mesh_as_ply('%s/%04d_recon.ply' % (cfg['testing']['output_dir'], data_idx),
                                        live_vertices, faces, live_normals, live_recon_mesh['vc'])

            # save results: reconstructed mesh
            live_img_front, live_img_back = visualize_util.render_live_mesh(phong_renderer, live_vertices, live_normals, faces, front_mv = front_mv, back_mv = back_mv)
            live_img = np.concatenate([live_img_front, live_img_back], 1)
            live_img = (255*live_img).astype(np.uint8)
            cv.imwrite('%s/%04d.jpg' % (cfg['testing']['output_dir'] + '/live_recon', data_idx), live_img)


if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)

    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-m', '--mode', type = str, default = 'test', choices = ['train', 'test'], help = 'Train or test.')
    args = arg_parser.parse_args()

    config.cfg = config.load_config(args.config_path)

    if args.mode == 'train':
        train_avatar()
    else:
        run_avatarcap(w_recon = True,
                      save_avatar_mesh = False,
                      save_final_mesh = False,
                      w_nerf = False,
                      frame_idx = None,
                      view_idx = 0,
                      interval = 1)

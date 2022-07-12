import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.ops import knn_points, knn_gather

from network.unets import UnetNoCond7DS, UnetNoCond5DS, UnetNoCond6DS
from network.mlp import MLP, OffsetDecoder

import config
from utils.net_util import get_embedder
from utils.smpl_util import smpl_util
from utils.nerf_util import raw2outputs


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class DoubleTNet(nn.Module):
    """
    Template Geo-Tex Network (Occupancy + NeRF)
    """
    def __init__(self):
        super(DoubleTNet, self).__init__()

        pos_encoding_freq = config.cfg['model']['cano_template'].get('pos_encoding', 10)
        print('# Canonical Template: Positional encoding %d' % pos_encoding_freq)
        self.position_embedder, position_dim = get_embedder(pos_encoding_freq, input_dims = 3)
        in_channels = position_dim

        self.shared_mlp = MLP(in_channels = in_channels,
                              out_channels = 256,
                              inter_channels = [256, 256, 256, 256, 256, 256],
                              res_layers = [4],
                              nlactv = 'relu',
                              last_op = None,
                              norm = None)

        self.geo_mlp = MLP(in_channels = 256,
                           out_channels = 2,
                           inter_channels = [128],
                           nlactv = 'leaky_relu',
                           last_op = None,
                           norm = None)

        self.clr_mlp = MLP(in_channels = 256,
                           out_channels = 3,
                           inter_channels = [256, 128],
                           nlactv = 'relu',
                           last_op = None,
                           norm = None)

        self.geo_mlp.fc_list[-1].apply(init_out_weights)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = False)

    def forward(self, pts):
        """
        :param pts: (B, N, 3)
        :return: rgb (B, N, 3), alpha (B, N, 1)
        """
        pts = self.position_embedder(pts)
        shared_feat = self.shared_mlp(pts.permute((0, 2, 1)))
        geo_output = self.geo_mlp(shared_feat)
        clr_output = self.clr_mlp(shared_feat)

        rgb = self.sigmoid(clr_output).permute((0, 2, 1))
        alpha = self.relu(geo_output[:, 1:, :].permute((0, 2, 1)))
        if config.if_type == 'occupancy':
            occ = self.sigmoid(geo_output[:, :1, :]).permute((0, 2, 1))
        elif config.if_type == 'sdf':
            occ = geo_output[:, :1, :].permute((0, 2, 1))
        else:
            raise ValueError('Invalid config.if_type!')
        return rgb, alpha, occ


class WarpingField(nn.Module):
    """
    Pose-dependent warping field network
    """
    def __init__(self):
        super(WarpingField, self).__init__()

        self.pose_feat_dim = 64
        input_nc = 6
        self.unet = UnetNoCond7DS(input_nc=input_nc, output_nc=self.pose_feat_dim, nf=32, up_mode='upconv', use_dropout=False)

        pos_encoding_freq = config.cfg['model']['warping_field'].get('pos_encoding', 0)
        print('# Warping Field: positional encoding %d' % pos_encoding_freq)
        self.position_embedder, position_dim = get_embedder(pos_encoding_freq, input_dims = 3)
        in_channels = position_dim + self.pose_feat_dim

        self.mlp = OffsetDecoder(in_channels)

        self.out_layer_coord_affine = nn.Conv1d(256, 3, 1)
        self.out_layer_coord_affine.apply(init_out_weights)

        self.pose_feat_map = None

    def precompute_conv(self, batch):
        smpl_pos_map = batch['smpl_pos_map']
        self.pose_feat_map = self.unet(smpl_pos_map)

    def query(self, pts, batch):
        """
        :param pts: (B, N, 3)
        :param batch: dict
        :return: (B, N, 3)
        """
        B, N, _ = pts.shape

        # positional encoding
        pts_en = self.position_embedder(pts).permute((0, 2, 1))

        with torch.no_grad():
            pts_ = pts - batch['cano_smpl_center'][:, None, :]
            x_grid = pts_[..., 0].view(B, N, 1, 1)
            y_grid = -pts_[..., 1].view(B, N, 1, 1)
            grid = torch.cat([x_grid, y_grid], dim = -1)

        pose_feat_map = self.pose_feat_map

        # sample
        pose_feat = F.grid_sample(pose_feat_map, grid, 'bilinear', 'border', True)
        pose_feat = pose_feat.squeeze(-1)

        in_feat = torch.cat([pts_en, pose_feat], dim = 1)
        in_feat = self.mlp(in_feat)
        offset = self.out_layer_coord_affine.forward(in_feat).permute((0, 2, 1))

        return offset


class CanoBlendWeightVolume:
    def __init__(self, base_weight_volume_path):
        base_weight_volume = np.load(base_weight_volume_path)
        base_weight_volume = base_weight_volume.transpose((3, 0, 1, 2))[None]
        self.base_weight_volume = torch.from_numpy(base_weight_volume).to(torch.float32).to(config.device)

    def forward(self, pts):
        """
        :param pts: (B, N, 3), scaled to [0, 1]
        :return: (B, N, 24)
        """
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        base_w = F.grid_sample(self.base_weight_volume,
                               grid,
                               padding_mode = 'border',
                               align_corners = True)
        base_w = base_w[0, :, :, 0, 0].reshape(-1, B, N).permute((1, 2, 0))

        return base_w


class GeoTexAvatar(nn.Module):
    def __init__(self):
        super(GeoTexAvatar, self).__init__()

        self.cano_template = DoubleTNet()

        self.cano_weight_volume = CanoBlendWeightVolume(config.cfg['training']['training_data_dir'] + '/cano_base_blend_weight_volume.npy')

        self.warping_field = WarpingField()

    def forward(self, wpts, viewdirs, dists, batch, pts_space = 'posed'):
        """
        :param wpts: (B, N, 3), in live space
        :param viewdirs: (B, N, 3), not used
        :param dists: (B, N)
        :param batch: dict()
        :param pts_space: 'posed': posed space, 'cano': canonical space, 'temp': template space
        :return:
        """
        assert (pts_space == 'posed' or pts_space == 'cano' or pts_space == 'temp')
        B, N = wpts.shape[:2]
        if pts_space == 'posed':  # require inverse skinning
            dists_to_smpl, indices, _ = knn_points(wpts, batch['live_smpl_v'])
            near_flag = dists_to_smpl[:, :, 0] < 0.08 * 0.08

            """
            NN, but using the fetched canonical blending weights
            """
            with torch.no_grad():
                cano_pts_w = knn_gather(smpl_util.smpl_skinning_weights[None].expand(B, -1, -1), indices)
                cano_pts_w = cano_pts_w[:, :, 0]
                live2cano_jnt_mats = torch.linalg.inv(batch['cano2live_jnt_mats'])  # (B, J, 4, 4)
                cano_pts_ = smpl_util.skinning(wpts, cano_pts_w, live2cano_jnt_mats)
                cano_min_xyz = batch['cano_bounds'][:, 0]
                cano_max_xyz = batch['cano_bounds'][:, 1]
                cano_pts_ = (cano_pts_ - cano_min_xyz[:, None]) / (cano_max_xyz - cano_min_xyz)[:, None]
            cano_pts_w = self.cano_weight_volume.forward(cano_pts_)
            cano_pts = smpl_util.skinning(wpts, cano_pts_w, live2cano_jnt_mats)
        else:
            cano_pts = wpts
            dists_to_smpl, indices, _ = knn_points(wpts, smpl_util.cano_smpl_vertices[None])
            near_flag = dists_to_smpl[:, :, 0] < 0.08 * 0.08

        if pts_space == 'posed' or pts_space == 'cano':
            offsets = self.warping_field.query(cano_pts, batch)  # (B, N, 3)
            cano_pts += offsets
        else:
            offsets = torch.zeros_like(cano_pts)

        # query rgb, alpha, occupancy
        rgb, alpha, occ = self.cano_template.forward(cano_pts)

        # post-processing (according to animatable NeRF (ICCV 21))
        inside = cano_pts > batch['cano_bounds'][:, :1]
        inside = inside * (cano_pts < batch['cano_bounds'][:, 1:])
        outside = torch.sum(inside, dim = 2) != 3
        alpha[outside] = 0
        alpha[~near_flag] = 0

        raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists)

        alpha = raw2alpha(alpha, dists)

        raw = torch.cat([rgb, alpha], dim = -1)

        output = {'raw': raw,
                  'occ': occ,
                  'nonrigid_offset': offsets}

        return output


class NerfRenderer:
    def __init__(self, net: GeoTexAvatar):
        self.net = net

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        """
        sample pts on rays
        """
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=config.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if config.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, batch, pts_space):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch, n_pixel * n_sample, -1)

        ret = self.net(wpts, viewdir, dists, batch, pts_space)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, depth, batch, pts_space = 'posed', near_dist = 0.05, far_dist = 0.05):
        # update near far
        valid_depth_flag = depth > 1e-6
        near[valid_depth_flag] = depth[valid_depth_flag] - near_dist
        far[valid_depth_flag] = depth[valid_depth_flag] + far_dist

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, batch, pts_space)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, white_bkgd = False)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret.update({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4)
        })

        return ret

    def render(self, batch, pts_space = 'posed', near_dist = 0.05, far_dist = 0.05):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        occ = batch['occupancy']  # image mask info
        depth = batch['depth']  # only valid in training

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            depth_chunk = depth[:, i:i+chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, depth_chunk,
                                               batch, pts_space,
                                               near_dist, far_dist)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim = 1) for k in keys}

        return ret


class OccupancyNet:
    def __init__(self, net: GeoTexAvatar):
        self.net = net

    def query(self, batch):
        """
        :param cano_bounds: (B, 2)
        :param cano_pts: (B, N, 3)
        :param pose: (B, 23*3)
        :return: (B, N, 1), (B, N, 3)
        """
        cano_pts = batch['cano_pts']
        B, N = cano_pts.shape[:2]

        chunk = 256 * 256 * 4
        offset_list = []
        occ_list = []
        for i in range(0, N, chunk):
            cano_pts_chunk = cano_pts[:, i: i + chunk]
            offset_chunk = self.net.warping_field.query(cano_pts_chunk, batch)
            _, _, occ_chunk = self.net.cano_template.forward(cano_pts_chunk + offset_chunk)
            offset_list.append(offset_chunk)
            occ_list.append(occ_chunk)

        offset_list = torch.cat(offset_list, dim = 1)
        occ_list = torch.cat(occ_list, dim = 1)

        output = {'cano_pts_ov': occ_list,
                  'nonrigid_offset': offset_list}
        return output

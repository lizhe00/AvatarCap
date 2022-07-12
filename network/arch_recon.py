import torch
import torch.nn as nn
import torch.nn.functional as F

from network.mlp import MLP
import network.HGFilters as hg


class ReconNetwork(nn.Module):
    def __init__(self):
        super(ReconNetwork, self).__init__()

        # model configuration
        img_encoder_config = {
            'in_channels': 6,
            'out_channels': 32
        }
        img_decoder_config = {
            'in_channels': None,
            'out_channels': 1,
            'inter_channels': [512, 256, 128],
            'res_layers': [1, 2],
            'nlactv': 'leaky_relu',
            'norm': 'weight',
            'last_op': 'sigmoid'
        }

        self.image_encoder = hg.HGFilter(1, 4, img_encoder_config['in_channels'], img_encoder_config['out_channels'], 'group', 'no_down', False)

        if img_decoder_config['in_channels'] is None:
            img_decoder_config['in_channels'] = img_encoder_config['out_channels'] + 1  # [img_feat, z]

        self.image_decoder = MLP(in_channels = img_decoder_config['in_channels'],
                                 out_channels = img_decoder_config['out_channels'],
                                 inter_channels = img_decoder_config['inter_channels'],
                                 res_layers = img_decoder_config['res_layers'],
                                 nlactv = img_decoder_config['nlactv'],
                                 last_op = img_decoder_config['last_op'],
                                 norm = img_decoder_config['norm'])

    def get_feat_maps(self, image):
        feat_maps, _ = self.image_encoder(image)
        return feat_maps

    def infer(self, items):
        with torch.no_grad():
            group_size = 256 * 256 * 4
            B, N, _ = items['cano_pts'].shape
            group_num = (N + group_size - 1) // group_size

            imgs = torch.cat([items['front_normal'], items['back_normal']], dim = 1)
            img_feat_map = self.get_feat_maps(imgs)[-1]

            pts_ov_list = []
            for group_idx in range(group_num):
                start_idx = group_idx * group_size
                end_idx = min(start_idx + group_size, N)
                cano_pts = items['cano_pts'][:, start_idx: end_idx, :]
                n = end_idx - start_idx

                # fetch image feature
                cano_pts_ = cano_pts - items['cano_smpl_center'][:, None, :]
                x_grid = cano_pts_[..., 0].view(B, n, 1, 1)
                y_grid = -cano_pts_[..., 1].view(B, n, 1, 1)
                grid = torch.cat([x_grid, y_grid], dim = -1)

                # decode
                pts_img_feat = F.grid_sample(img_feat_map, grid, 'bilinear', 'border', True).squeeze(-1)
                z = cano_pts_[..., 2].view(B, 1, n)
                pts_total_feat = torch.cat([pts_img_feat, z], dim = 1)
                pts_ov = self.image_decoder(pts_total_feat)

                pts_ov_list.append(pts_ov)
            pts_ov_list = torch.cat(pts_ov_list, dim = -1).squeeze(0)

            return pts_ov_list

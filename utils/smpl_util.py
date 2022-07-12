import os.path

import numpy as np
import torch
import math

import config
from pytorch3d.ops import knn_points, knn_gather


class SmplUtil:
    def __init__(self, smpl_dir):
        self.smpl_skinning_weights = torch.from_numpy(np.loadtxt(smpl_dir + '/weightsFile.txt').astype(np.float32)).to(config.device)

        self.cano_smpl_pose = np.zeros(75, dtype = np.float32)
        self.cano_smpl_pose[3 + 3 * 1 + 2] = math.radians(25)
        self.cano_smpl_pose[3 + 3 * 2 + 2] = math.radians(-25)
        self.cano_smpl_vertices = None

    def set_cano_smpl_vertices(self, cano_smpl_vertices: torch.Tensor):
        self.cano_smpl_vertices = cano_smpl_vertices.to(config.device)

    def calculate_lbs(self, points):
        '''
        Approximate the blending weights of points around smpl
        :param points: (B, N, 3), on device
        :return: (B, N, 24)
        '''
        if self.cano_smpl_vertices is None:
            raise ValueError('Canonical smpl vertices are invalid!')
        B, N, _ = points.shape
        dists, indices, _ = knn_points(points, self.cano_smpl_vertices[None, ...].expand(B, -1, -1), K = 4)
        r = 0.05
        weights = torch.exp(-dists / (2 * r * r))
        weights /= torch.sum(weights, dim = -1, keepdim = True) + 1e-16  # (B, N, 1)
        lbs = knn_gather(self.smpl_skinning_weights[None, ...].expand(B, -1, -1), indices)  # (B, N, K, 24)
        lbs = torch.sum(lbs * weights[..., None], dim = -2)  # (B, N, 24)
        return lbs

    def calculate_lbs2(self, points, smpl_v):
        '''
        Approximate the blending weights of points around smpl
        :param points: (B, N, 3), on device
        :param smpl_v: (B, N', 3), on device
        :return: (B, N, 24)
        '''
        B, N, _ = points.shape
        dists, indices, _ = knn_points(points, smpl_v, K = 4)
        r = 0.05
        weights = torch.exp(-dists / (2 * r * r))
        weights /= torch.sum(weights, dim = -1, keepdim = True) + 1e-16  # (B, N, 1)
        lbs = knn_gather(self.smpl_skinning_weights[None, ...].expand(B, -1, -1), indices)  # (B, N, K, 24)
        lbs = torch.sum(lbs * weights[..., None], dim = -2)  # (B, N, 24)
        # lbs[dists[:, :, 0] > 0.05 * 0.05, :] = 0.
        return lbs

    def skinning(self, points, lbs, jnt_mats, return_pt_mats = False):
        '''
        forward skinning
        :param points: (B, N, 3)
        :param lbs: (B, N, 24)
        :param jnt_mats: (B, 24, 4, 4)
        :return:
        '''
        # lbs
        cano2live_pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, jnt_mats)

        live_pts = torch.einsum('bnxy,bny->bnx', cano2live_pt_mats[..., :3, :3], points) + cano2live_pt_mats[..., :3, 3]

        if return_pt_mats:
            return live_pts, cano2live_pt_mats
        else:
            return live_pts

    def skinning_normal(self, normals, lbs, cano2live_jnt_mats):
        # lbs
        cano2live_pt_mats = torch.einsum('bnj,bjxy->bnxy', lbs, cano2live_jnt_mats)

        live_normals = torch.einsum('bnxy,bny->bnx', cano2live_pt_mats[..., :3, :3], normals)
        return live_normals


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
smpl_util = SmplUtil(PROJ_DIR + '/smpl_files/ModelTxt_%s' % config.smpl_gender)

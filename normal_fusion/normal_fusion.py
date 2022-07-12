import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from pytorch3d.transforms import axis_angle_to_matrix

import config
from utils.renderer import Renderer, gl_perspective_projection_matrix
from utils.visualize_util import render_cano_mesh


def canonicalize_normal_map(pos_renderer: Renderer, attri_renderer: Renderer, cano_vertices, live_vertices, faces, normal_map, vert_mats, mv, fx, fy, cx, cy, cano_smpl_center):
    img_h, img_w = normal_map.shape[:2]
    vertices_ = live_vertices[faces].reshape(-1, 3).astype(np.float32)
    pos_renderer.set_model(vertices_)
    proj_mat = gl_perspective_projection_matrix(fx, fy, cx, cy, img_w, img_h, gl_space = False)
    mvp = np.dot(proj_mat, mv)
    pos_renderer.set_mvp_mat(mvp)
    position_map = pos_renderer.render()

    # cv.imshow('position_map', position_map)
    # cv.imshow('normal_map', normal_map)
    # cv.waitKey(0)

    # check visibility
    vertices_gpu = torch.from_numpy(live_vertices).to(torch.float32).to(config.device)
    mv_gpu = torch.from_numpy(mv).to(torch.float32).to(config.device)
    vertices_gpu_cam = torch.einsum('ij,vj->vi', mv_gpu[:3, :3], vertices_gpu) + mv_gpu[:3, 3][None]
    coord_x = vertices_gpu_cam[:, 0] / vertices_gpu_cam[:, 2] * fx + cx
    coord_y = vertices_gpu_cam[:, 1] / vertices_gpu_cam[:, 2] * fy + cy
    coord_x = 2. * (coord_x / img_w) - 1.
    coord_y = 2. * (coord_y / img_h) - 1.
    coord_2d = torch.stack([coord_x, coord_y], dim = -1)[None, :, None]
    position_map = torch.from_numpy(position_map.copy()).to(torch.float32).to(config.device).permute((2, 0, 1))[None]
    proj_v = F.grid_sample(position_map, coord_2d, 'nearest', 'border', True)[0, :, :, 0].permute((1, 0))[:, :3]
    vis_flag = torch.linalg.norm(vertices_gpu - proj_v, dim = -1) < 0.05

    # # debug: visibility flag
    # import trimesh
    # vis_vertices = vertices_gpu_cam#[vis_flag]
    # vis_vertices_trimesh = trimesh.PointCloud(vis_vertices.cpu().numpy())
    # vis_vertices_trimesh.export('./debug/vis_vertices_trimesh.obj')
    # print(proj_v.shape)

    normal_map = torch.from_numpy(normal_map).to(torch.float32).to(config.device).permute((2, 0, 1))[None]
    proj_n = F.grid_sample(normal_map, coord_2d, 'nearest', 'border', True)[0, :, :, 0].permute((1, 0))[:, :3]
    valid_flag = torch.logical_and(vis_flag, torch.linalg.norm(proj_n, dim = -1) > 1e-6)

    # # debug: visibility flag
    # import trimesh
    # valid_vertices = vertices_gpu[valid_flag]
    # valid_normals = proj_n[valid_flag]
    # valid_vertices_trimesh = trimesh.PointCloud(valid_vertices.cpu().numpy(), colors = 0.5*valid_normals.cpu().numpy()+0.5)
    # valid_vertices_trimesh.export('./debug/valid_vertices_w_normal_trimesh.obj')

    # canonicalize normal
    proj_n[:, 1:] *= -1  # y, z
    proj_n = torch.einsum('ij,vj->vi', torch.linalg.inv(mv_gpu)[:3, :3], proj_n)
    proj_n = torch.einsum('vij,vj->vi', torch.linalg.inv(vert_mats)[:, :3, :3], proj_n)
    proj_n[~valid_flag] = 0.
    proj_n = proj_n.cpu().numpy()
    front_image_normal, back_image_normal = render_cano_mesh(attri_renderer, cano_vertices, proj_n, faces, cano_smpl_center)
    return front_image_normal, back_image_normal


def get_neighbor_images(img, win_size = 3):
    H, W, _ = img.shape
    half_win_size = win_size // 2
    neighbor_images = []
    for i in range(-half_win_size, half_win_size + 1):  # row
        for j in range(-half_win_size, half_win_size + 1):  # col
            if i == 0 and j == 0:
                continue
            theta = torch.tensor([[1, 0, j/(H/2)], [0, 1, i/(W/2)]], dtype = torch.float32, device = config.device)
            grid = F.affine_grid(theta.unsqueeze(0), torch.Size((1, 1, H, W)), align_corners = True)
            affine_img = F.grid_sample(input = img.permute((2, 0, 1)).unsqueeze(0),
                                       grid = grid, mode = 'nearest', align_corners = True)
            affine_img = affine_img.squeeze(0).permute((1, 2, 0))
            neighbor_images.append(affine_img)
    return neighbor_images


def resize_img(src_img, tar_shape):
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]],
                         dtype = torch.float32, device = config.device)
    grid = F.affine_grid(theta.unsqueeze(0), torch.Size((1, 1, tar_shape[0], tar_shape[1])), align_corners = True)
    sampled_img = F.grid_sample(src_img.permute((2, 0, 1)).unsqueeze(0), grid, 'bilinear', 'border', True)
    return sampled_img.squeeze(0).permute((1, 2, 0))


def merge_normal_images(src_img, tar_img, iter_num, neck_xy):
    """
    Canonical normal fusion using 2D rotation grids.
    :param src_img: avatar normal, (512, 512, 3)
    :param tar_img: image-observed normal, (512, 512, 3)
    :param iter_num: iteration number in normal fusion
    :param neck_xy: tuple or list, the 2D neck position on the canonical image plane
    :return: merged_img: merged normal, (512, 512, 3)
    """
    src_img = torch.from_numpy(src_img).to(torch.float32).to(config.device)
    tar_img = torch.from_numpy(tar_img).to(torch.float32).to(config.device)
    src_mask = torch.linalg.norm(src_img, dim = -1) > 0.
    tar_mask = torch.linalg.norm(tar_img, dim = -1) > 0.

    # erode tar mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    tar_mask = cv.erode(tar_mask.cpu().numpy().astype(np.uint8), kernel, iterations = 3)
    dt_tar_mask = cv.distanceTransform(tar_mask, cv.DIST_L1, 3)
    dt_tar_mask = torch.from_numpy(dt_tar_mask).to(config.device)
    tar_mask = torch.from_numpy(tar_mask > 0).to(config.device)

    valid_mask = torch.logical_and(src_mask, tar_mask)
    src_img.requires_grad_()
    init_src_img = src_img.detach().clone()

    rot_aa_img = torch.zeros((64, 64, 3), dtype = torch.float32, device = config.device)
    rot_aa_img.requires_grad_()

    optm_rot = torch.optim.Adam([rot_aa_img], lr = 1e-2)
    optm_normal = torch.optim.Adam([src_img], lr = 1e-1)
    smooth_lambda = 1.
    for iter_idx in range(iter_num):
        sampled_rot_aa_img = resize_img(rot_aa_img, (512, 512))
        rot_mat_img = axis_angle_to_matrix(sampled_rot_aa_img)

        # data term
        data_loss = torch.square(torch.einsum('ijab,ijb->ija', rot_mat_img, src_img) - tar_img)[valid_mask].mean()

        # smooth term
        neighbor_rot_aa_imgs = get_neighbor_images(rot_aa_img)
        smooth_loss = 0.
        for neighbor_rot_aa_img in neighbor_rot_aa_imgs:
            smooth_loss += torch.square(neighbor_rot_aa_img - rot_aa_img).mean()

        total_loss = data_loss + smooth_lambda * smooth_loss
        if iter_idx < iter_num / 2:
            optm_rot.zero_grad()
            total_loss.backward()
            optm_rot.step()
        else:
            optm_normal.zero_grad()
            total_loss.backward()
            optm_normal.step()

    # dt mask blend
    dt_tar_mask = dt_tar_mask[..., None]
    dt_tar_mask /= 5.
    init_src_img_weight = torch.ones_like(dt_tar_mask)
    init_src_img_weight[dt_tar_mask > 1.] = 0.
    src_img = (src_img * dt_tar_mask + init_src_img * init_src_img_weight) / (dt_tar_mask + init_src_img_weight)

    # face regions should follow the avatar normal
    face_rect = [neck_xy[1] - 90, neck_xy[0] - 35, neck_xy[1], neck_xy[0] + 35]
    src_img[face_rect[0]: face_rect[2], face_rect[1]: face_rect[3]] = init_src_img[face_rect[0]: face_rect[2], face_rect[1]: face_rect[3]]
    return src_img.detach().cpu().numpy()


def merge_normal_images_cover(src_img, tar_img):
    """
    Directly cover the avatar normal with image-observed one.
    :param src_img: avatar normal, (512, 512, 3)
    :param tar_img: image-observed normal, (512, 512, 3)
    :return: merged normal, (512, 512, 3)
    """
    valid_mask = np.linalg.norm(tar_img, axis = -1) > 1e-6
    src_img[valid_mask] = tar_img[valid_mask]
    return src_img

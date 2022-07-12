import os
import torch
import torch.nn.functional as F
import config
from utils.renderer import Renderer, gl_orthographic_projection_matrix, gl_perspective_projection_matrix
import cv2 as cv
import numpy as np
import math


def render_cano_mesh(renderer: Renderer, vertices, normals, faces, mesh_center = np.zeros(3), colors = None):
    vertices_ = vertices[faces.reshape(-1)].astype(np.float32)
    normals_ = normals[faces.reshape(-1)].astype(np.float32)

    # front & back mvp
    model_RT = np.identity(4, dtype = np.float32)
    model_RT[:3, 3] = -mesh_center
    model_RT[2, 3] -= 10
    proj_mat = gl_orthographic_projection_matrix()
    front_mvp = np.dot(proj_mat, model_RT)
    front_mv = model_RT

    # model_RT = np.identity(4, dtype = np.float32)
    # model_RT[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
    # model_RT[:3, 3] = -mesh_center
    # model_RT[2, 3] -= 10
    trans_cen = np.identity(4, np.float32)
    trans_cen[:3, 3] = -mesh_center
    rot_y = np.identity(4, np.float32)
    rot_y[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
    trans_z = np.identity(4, np.float32)
    trans_z[2, 3] = -10
    model_RT = np.dot(trans_z, np.dot(rot_y, trans_cen))
    proj_mat = gl_orthographic_projection_matrix()
    back_mvp = np.dot(proj_mat, model_RT)
    back_mv = model_RT

    if colors is None:
        renderer.set_model(vertices_, normals_)
    else:
        colors_ = colors[faces.reshape(-1)].astype(np.float32)
        renderer.set_model(vertices_, normals_, colors_)
    renderer.set_mvp_mat(front_mvp)
    renderer.set_mv_mat(front_mv)
    front_normal_img = renderer.render()
    front_normal_img = cv.cvtColor(front_normal_img, cv.COLOR_RGBA2RGB)
    renderer.set_mvp_mat(back_mvp)
    renderer.set_mv_mat(back_mv)
    back_normal_img = renderer.render()
    back_normal_img = cv.cvtColor(back_normal_img, cv.COLOR_RGBA2RGB)
    back_normal_img = cv.flip(back_normal_img, 1)
    return front_normal_img, back_normal_img


def calc_front_mv(mesh_vertices, rot_x_angle = 0., rot_y_angle = 0.):
    center = 0.5 * (mesh_vertices.max(0) + mesh_vertices.min(0))
    T_0 = np.identity(4, np.float32)
    T_0[:3, 3] = -center
    rot_x = np.identity(4, np.float32)
    rot_x[:3, :3] = cv.Rodrigues(np.array([rot_x_angle, 0, 0]))[0]
    rot_y = np.identity(4, np.float32)
    rot_y[:3, :3] = cv.Rodrigues(np.array([0, rot_y_angle, 0]))[0]
    T_0 = np.dot(rot_x, T_0)
    T_0 = np.dot(rot_y, T_0)
    T_1 = np.identity(4, np.float32)
    T_1[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
    T_2 = np.identity(4, np.float32)
    T_2[2, 3] = 20

    front_mv = np.dot(T_2, T_0)
    return front_mv


def calc_back_mv(mesh_vertices, rot_x_angle = 0.):
    center = 0.5 * (mesh_vertices.max(0) + mesh_vertices.min(0))
    T_0 = np.identity(4, np.float32)
    T_0[:3, 3] = -center
    rot_x = np.identity(4, np.float32)
    rot_x[:3, :3] = cv.Rodrigues(np.array([rot_x_angle, 0, 0]))[0]
    T_0 = np.dot(rot_x, T_0)
    T_1 = np.identity(4, np.float32)
    T_1[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
    T_2 = np.identity(4, np.float32)
    T_2[2, 3] = 20

    back_mv = np.dot(T_2, np.dot(T_1, T_0))
    return back_mv


def render_live_mesh(renderer: Renderer, vertices, normals, faces, colors = None, front_mv = None, back_mv = None,
                     fx = 5000, fy = 5000, cx = 256, cy = 256, img_w = 512, img_h = 512):
    real2gl = np.identity(4, np.float32)
    real2gl[:3, :3] = cv.Rodrigues(np.array([math.pi, 0, 0]))[0]

    if front_mv is None:
        front_mv = calc_front_mv(vertices)

    if back_mv is None:
        back_mv = calc_back_mv(vertices)

    front_mv = np.dot(real2gl, front_mv)
    back_mv = np.dot(real2gl, back_mv)

    vertices_ = vertices[faces.reshape(-1)].astype(np.float32)
    normals_ = normals[faces.reshape(-1)].astype(np.float32)
    if colors is None:
        renderer.set_model(vertices_, normals_)
    else:
        colors_ = colors[faces.reshape(-1)].astype(np.float32)
        renderer.set_model(vertices_, normals_, colors_)

    proj_mat = gl_perspective_projection_matrix(fx, fy, cx, cy, img_w, img_h, gl_space = True)

    # front view
    renderer.set_mv_mat(front_mv)
    renderer.set_mvp_mat(np.dot(proj_mat, front_mv))
    front_geo_img = renderer.render()
    front_geo_img = cv.cvtColor(front_geo_img, cv.COLOR_RGBA2RGB)

    # back view
    renderer.set_mv_mat(back_mv)
    renderer.set_mvp_mat(np.dot(proj_mat, back_mv))
    back_geo_img = renderer.render()
    back_geo_img = cv.cvtColor(back_geo_img, cv.COLOR_RGBA2RGB)

    return front_geo_img, back_geo_img


def normal2color(normal_img):
    mask = np.linalg.norm(normal_img, axis = -1) > 1e-6
    valid_normal = normal_img[mask]
    valid_normal /= np.linalg.norm(valid_normal, axis = -1, keepdims = True)
    normal_img[mask] = 0.5 * valid_normal + 0.5
    normal_img = cv.cvtColor(normal_img, cv.COLOR_RGB2BGR)
    return normal_img

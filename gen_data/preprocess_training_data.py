"""
Process training data.
Suppose scans and corresponding SMPL poses are ready.

Layout of the given data before processing:
data_dir:
├── scan
│   └── 000.ply
│   └── 001.ply
├── smpl
│   └── pose_000.txt
│   └── pose_001.txt
│   └── shape.txt

"""
import glob
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import trimesh
import numpy as np
import math
import torch
from pytorch3d.ops import knn_points, knn_gather
import igl
import cv2 as cv
import scipy.io as sio
import yaml


import config
from dataset.smpl import SmplModel, smpl_params
from gen_data.nonrigid_fitting import nonrigid_fitting
from utils.obj_io import save_mesh_as_ply
from utils.renderer import Renderer, gl_perspective_projection_matrix, gl_orthographic_projection_matrix


# define canonical smpl pose
cano_smpl_pose = np.zeros(75, dtype = np.float32)
cano_smpl_pose[3 + 3 * 1 + 2] = math.radians(25)
cano_smpl_pose[3 + 3 * 2 + 2] = math.radians(-25)


# gpu device
device = config.device


def canonicalize(scan: trimesh.Trimesh,
                 smpl_pose: np.ndarray,
                 smpl_shape: np.ndarray,
                 idx,
                 tmp_dir = './debug'):
    # cano smpl pose
    cano_smpl = SmplModel(cano_smpl_pose, smpl_shape)
    cano_smpl_trimesh = trimesh.Trimesh(vertices = cano_smpl.posed_vertices, faces = smpl_params.faces)

    live_smpl_pose = smpl_pose.copy()
    live_smpl_pose[-12:] = 0.
    live_smpl = SmplModel(live_smpl_pose, smpl_shape)

    live_scan = scan
    live_scan_v = torch.from_numpy(live_scan.vertices.copy()).to(torch.float32).to(device)
    live_scan_n = torch.from_numpy(live_scan.vertex_normals.copy()).to(torch.float32).to(device)
    live_scan_n /= torch.norm(live_scan_n, dim = -1, keepdim = True) + 1e-16
    live_smpl_v = torch.from_numpy(live_smpl.posed_vertices.copy()).to(torch.float32).to(device)
    live_smpl_trimesh = trimesh.Trimesh(vertices = live_smpl.posed_vertices, faces = smpl_params.faces)

    """
    1. Inverse skinning
    """
    dists, face_indices, _ = igl.signed_distance(live_scan.vertices, live_smpl.posed_vertices, smpl_params.faces)
    triangles = live_smpl.posed_vertices[smpl_params.faces[face_indices]]
    triangle_lbs = smpl_params.weights[smpl_params.faces[face_indices]]
    weights = trimesh.triangles.points_to_barycentric(triangles, live_scan.vertices)
    lbs = (weights[..., None] * triangle_lbs).sum(1)
    lbs = torch.from_numpy(lbs).to(torch.float32).to(device)

    # inverse skinning
    live_jnt_mats = torch.from_numpy(live_smpl.jnt_affine_mats).to(torch.float32).to(device)
    cano_jnt_mats = torch.from_numpy(cano_smpl.jnt_affine_mats).to(torch.float32).to(device)
    live2cano_jnt_mats = torch.matmul(cano_jnt_mats, torch.linalg.inv(live_jnt_mats))
    vertex_mats = torch.einsum('vj,jab->vab', lbs, live2cano_jnt_mats)
    cano_scan_v = torch.einsum('vij,vj->vi', vertex_mats[:, :3, :3], live_scan_v) + vertex_mats[:, :3, 3]
    cano_scan_n = torch.einsum('vij,vj->vi', vertex_mats[:, :3, :3], live_scan_n)

    # live normal flag
    knn_smpl_n = live_smpl_trimesh.face_normals[face_indices]
    knn_smpl_n = torch.from_numpy(knn_smpl_n).to(torch.float32).to(device)
    normal_cos = torch.einsum('vi,vi->v', knn_smpl_n, live_scan_n)
    live_normal_flag = normal_cos > 0.
    live_normal_flag = live_normal_flag.cpu().numpy()

    # canonical normal flag
    knn_smpl_n = cano_smpl_trimesh.face_normals[face_indices]
    knn_smpl_n = torch.from_numpy(knn_smpl_n).to(torch.float32).to(device)
    normal_cos = torch.einsum('vi,vi->v', knn_smpl_n, cano_scan_n)
    cano_normal_flag = normal_cos > 0.
    cano_normal_flag = cano_normal_flag.cpu().numpy()
    normal_flag = live_normal_flag & cano_normal_flag

    valid_flag = torch.from_numpy(normal_flag)
    cano_scan_v = cano_scan_v[valid_flag]
    cano_scan_n = cano_scan_n[valid_flag]
    cano_scan_pc = trimesh.Trimesh(vertices = cano_scan_v.cpu().numpy(), process = False, vertex_normals = cano_scan_n.cpu().numpy())


    """
    1.5 Non-rigid fitting
    """
    template = trimesh.Trimesh(cano_smpl.posed_vertices, smpl_params.faces, process = False)
    left_wrist_pos = cano_smpl.posed_vertices[1931]
    right_wrist_pos = cano_smpl.posed_vertices[5392]
    v0 = template.vertices[template.faces[:, 0]]
    v1 = template.vertices[template.faces[:, 1]]
    v2 = template.vertices[template.faces[:, 2]]
    template_f_flag = (v0[..., 0] < left_wrist_pos[0]) & (v0[..., 0] > right_wrist_pos[0])
    template_f_flag = template_f_flag & (v1[..., 0] < left_wrist_pos[0]) & (v1[..., 0] > right_wrist_pos[0])
    template_f_flag = template_f_flag & (v2[..., 0] < left_wrist_pos[0]) & (v2[..., 0] > right_wrist_pos[0])
    template.update_faces(template_f_flag)
    template = template.subdivide()

    fitted_template = nonrigid_fitting(template, cano_scan_pc, None, fix_vertex_indices = None, iteration_num = 200)
    # template.export('../debug/ori_template.obj')
    # fitted_template.export('../debug/fitted_template.obj')
    # exit(1)

    """
    2. Inpainting
    """
    template_v = torch.from_numpy(fitted_template.vertices).to(torch.float32).to(device)
    template_n = torch.from_numpy(fitted_template.vertex_normals).to(torch.float32).to(device)
    template_f = torch.from_numpy(fitted_template.faces.copy()).to(torch.long).to(device)
    template_v = template_v[template_f].reshape(-1, 3)  # remove hand vertices
    template_n = template_n[template_f].reshape(-1, 3)

    # knn for template
    dists, _, _ = knn_points(template_v[None, ...], cano_scan_v[None, ...], K = 1)
    dists = dists.squeeze()
    inpainting_flag = dists > 0.01 * 0.01
    inpainting_v = template_v[inpainting_flag]
    inpainting_n = template_n[inpainting_flag]

    all_v = torch.cat([cano_scan_v, inpainting_v], dim = 0)
    all_n = torch.cat([cano_scan_n, inpainting_n], dim = 0)

    # # inverse filter fragment of cano_scan_v
    # dists, _ = sdf_cuda_tensor(template_v, template_f, all_v, config.device)
    # valid_flag = dists < 0.02
    # all_v = all_v[valid_flag]
    # all_n = all_n[valid_flag]

    save_mesh_as_ply(tmp_dir + '/cano_inpainted_pc.ply', vertices = all_v.cpu().numpy(), normals = all_n.cpu().numpy())

    """
    3. Poisson reconstruction
    """
    output_path = tmp_dir + '/wt_cano_scan_%03d' % idx
    command = '.\\gen_data\\bin\\PoissonRecon.exe --in ' + tmp_dir + '/cano_inpainted_pc.ply' + ' --out ' + output_path + ' --depth 9'
    os.system(command)
    os.remove(tmp_dir + '/cano_inpainted_pc.ply')

    """
    4. Calculate original surface flag
    """
    wt_cano_scan = trimesh.load(output_path + '.ply', process = False)
    wt_cano_scan_v = torch.from_numpy(wt_cano_scan.vertices.copy()).to(torch.float32).to(config.device)
    wt_cano_scan_f = torch.from_numpy(wt_cano_scan.faces.copy()).to(torch.long).to(config.device)
    pc_v = torch.from_numpy(cano_scan_pc.vertices.copy()).to(torch.float32).to(config.device)

    dists, _, _ = knn_points(wt_cano_scan_v[None, ...], pc_v[None, ...], K = 1)
    dists = dists.squeeze()
    ori_v_flag = dists < 0.01 * 0.01

    ori_f_flag = ori_v_flag[wt_cano_scan_f[:, 0]] & ori_v_flag[wt_cano_scan_f[:, 1]] & ori_v_flag[wt_cano_scan_f[:, 2]]

    ori_v_flag = ori_v_flag.cpu().numpy()
    ori_f_flag = ori_f_flag.cpu().numpy()

    # # debug
    # wt_cano_scan.visual.vertex_colors = 0.8 * np.ones((wt_cano_scan.vertices.shape[0], 4), dtype = np.float32)
    # wt_cano_scan.visual.vertex_colors[ori_v_flag] = np.array([255, 0, 0, 0])
    # wt_cano_scan.export('./debug/wt_cano_scan_w_ori_sur_flag.ply')

    wt_cano_scan = trimesh.load(output_path + '.ply', process = False)
    return wt_cano_scan, ori_f_flag


def sample_surface_pts(mesh, count, mask = None, w_color = False):
    """
    Modified from Scanimate code
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """
    valid_faces = mesh.faces[mask]
    face_index = np.random.choice(a = valid_faces.shape[0], size = count, replace = True)
    selected_faces = valid_faces[face_index]

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[selected_faces[:, 0]]
    tri_vectors = mesh.vertices[selected_faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis = 1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis = 1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    colors = None
    normals = None
    if w_color:
        colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
        colors = colors / 255.0
        colors = colors.view(np.ndarray)[selected_faces]
        clr_origins = colors[:, 0]
        clr_vectors = colors[:, 1:]
        clr_vectors -= np.tile(clr_origins, (1, 2)).reshape((-1, 2, 3))

        sample_color = (clr_vectors * random_lengths).sum(axis=1)
        colors = sample_color + clr_origins

        normals = mesh.face_normals[face_index]

    return samples, colors, normals


def presample_pts(wt_cano_scan, ori_sur_flag):
    sur_pts_count = 2200000
    vol_pts_count = 10000
    sigma = 0.02

    min_xyz = wt_cano_scan.vertices.min(0) - 0.2
    max_xyz = wt_cano_scan.vertices.max(0) + 0.2
    bounds = np.stack([min_xyz, max_xyz], 0)

    sur_pts, _, _ = sample_surface_pts(wt_cano_scan, sur_pts_count, ori_sur_flag, False)

    # adaptive random shifts
    curv_radius = 0.002
    curv_thresh = 0.004
    curvs = trimesh.curvature.discrete_gaussian_curvature_measure(wt_cano_scan, sur_pts, curv_radius)
    curvs = abs(curvs)
    curvs = curvs / max(curvs)  # normalize curvature
    sigmas = np.zeros(curvs.shape)
    sigmas[curvs <= curv_thresh] = sigma
    sigmas[curvs > curv_thresh] = sigma / 5

    random_shifts = np.random.randn(sur_pts.shape[0], sur_pts.shape[1]) * sigmas[:, None]
    # random_shifts = np.random.randn(sur_pts.shape[0], sur_pts.shape[1]) * sigma
    sur_pts = sur_pts + random_shifts
    sur_pts = sur_pts.astype(np.float32)

    vol_pts = np.random.random((vol_pts_count, 3))  # [0, 1]
    vol_pts = (bounds[1] - bounds[0]) * vol_pts + bounds[0]
    vol_pts = vol_pts.astype(np.float32)

    # filter volume points by dist to scan
    vol_pts_gpu = torch.from_numpy(vol_pts).to(config.device)
    invalid_faces = wt_cano_scan.faces[~ori_sur_flag]
    invalid_vertices = wt_cano_scan.vertices[invalid_faces].reshape(-1, 3)
    invalid_vertices = torch.from_numpy(invalid_vertices).to(torch.float32).to(config.device)
    dists, _, _ = knn_points(vol_pts_gpu[None], invalid_vertices[None])
    dists = torch.sqrt(dists)[0, :, 0]
    vol_pts_gpu = vol_pts_gpu[dists > 0.05]
    vol_pts = vol_pts_gpu.cpu().numpy()
    if len(vol_pts) < 1000:
        raise RuntimeError('vol pts are not enough')

    sur_pts_gpu = torch.from_numpy(sur_pts).to(config.device)
    dists, _, _ = knn_points(sur_pts_gpu[None], invalid_vertices[None])
    dists = torch.sqrt(dists)[0, :, 0]
    sur_pts_gpu = sur_pts_gpu[dists > 0.02]
    sur_pts = sur_pts_gpu.cpu().numpy()
    sur_pts_num = sur_pts.shape[0]
    print('sur_pts_num: %d' % sur_pts_num)

    # calculate pts ov/sdf
    all_pts = np.concatenate([sur_pts, vol_pts], 0)
    all_pts_ov, _, _ = igl.signed_distance(all_pts, wt_cano_scan.vertices, wt_cano_scan.faces)

    all_pts_ov *= -1  # assume the inside is positive

    sur_pts_ov = all_pts_ov[: sur_pts_num]
    vol_pts_ov = all_pts_ov[sur_pts_num:]

    return sur_pts, sur_pts_ov, vol_pts, vol_pts_ov


def render_images(color_renderer: Renderer,
                  pos_renderer: Renderer,
                  scan: trimesh.Trimesh,
                  output_dir: str,
                  cam: dict,
                  view_num = 60):
    dist = 2.3
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    img_w, img_h = cam['img_width'], cam['img_height']
    proj_mat = gl_perspective_projection_matrix(fx, fy, cx, cy, img_w, img_h)

    mesh_center = 0.5 * (scan.vertices.max(0) + scan.vertices.min(0))
    trans_center = np.identity(4, np.float32)
    trans_center[:3, 3] = -mesh_center
    rot_x = np.identity(4, np.float32)
    rot_x[:3, :3] = cv.Rodrigues(np.array([math.pi, 0., 0.]))[0]
    trans_z = np.identity(4, np.float32)
    trans_z[2, 3] = dist

    # set model
    vertices = scan.vertices[scan.faces.reshape(-1)].astype(np.float32)
    colors = scan.visual.vertex_colors[scan.faces.reshape(-1)][:, :3].astype(np.float32)
    if colors.max() > 1.1:
        colors /= 255.
    color_renderer.set_model(vertices, colors)
    pos_renderer.set_model(vertices)

    cam_rs = []
    cam_ts = []
    for view_idx in range(view_num):
        rot_y_angle = (2 * math.pi) * view_idx / view_num
        rot_y = np.identity(4, np.float32)
        rot_y[:3, :3] = cv.Rodrigues(np.array([0., rot_y_angle, 0.]))[0]

        # extr: trans_center -> rot_y -> rot-x -> trans_z
        extr = np.dot(rot_y, trans_center)
        extr = np.dot(rot_x, extr)
        extr = np.dot(trans_z, extr)
        mvp = np.dot(proj_mat, extr)

        # render color
        color_renderer.set_mvp_mat(mvp)
        color_img = color_renderer.render()[:, :, :3]
        color_img = cv.cvtColor(color_img, cv.COLOR_RGB2BGR)
        color_img = (255 * color_img).astype(np.uint8)
        cv.imwrite(output_dir + '/color_view_%03d.jpg' % view_idx, color_img)

        # render depth & mask
        pos_renderer.set_mvp_mat(mvp)
        pos_map = pos_renderer.render()[:, :, :3]
        mask_img = (np.linalg.norm(pos_map, axis = -1) > 0.).astype(np.uint8)
        mask_img = 255 * mask_img
        cv.imwrite(output_dir + '/mask_view_%03d.png' % view_idx, mask_img)

        pos_map[mask_img > 0] = np.einsum('ij,vj->vi', extr[:3, :3], pos_map[mask_img > 0]) + extr[:3, 3]
        depth_img = (1000 * pos_map[:, :, 2]).astype(np.uint16)
        cv.imwrite(output_dir + '/depth_view_%03d.png' % view_idx, depth_img)

        cam_r = cv.Rodrigues(extr[:3, :3])[0]
        cam_t = extr[:3, 3]
        cam_rs.append(cam_r[:, 0])
        cam_ts.append(cam_t)

    cam_rs = np.stack(cam_rs, axis = 0)
    cam_ts = np.stack(cam_ts, axis = 0)
    sio.savemat(output_dir + '/cams.mat', {'cam_rs': cam_rs, 'cam_ts': cam_ts})


def render_smpl_position_map(attri_renderer: Renderer,
                             pose: np.ndarray,
                             shape: np.ndarray):
    cano_smpl = SmplModel(cano_smpl_pose, shape)
    cano_smpl_center = 0.5 * (cano_smpl.posed_vertices.max(0) + cano_smpl.posed_vertices.min(0))

    # calculate front & back mvp
    model_RT = np.identity(4, dtype = np.float32)
    model_RT[:3, 3] = -cano_smpl_center
    model_RT[2, 3] -= 10
    proj_mat = gl_orthographic_projection_matrix()
    front_mvp = np.dot(proj_mat, model_RT)
    front_mv = model_RT.copy()

    model_RT = np.identity(4, dtype = np.float32)
    model_RT[:3, :3] = cv.Rodrigues(np.array([0, math.pi, 0]))[0]
    model_RT[:3, 3] = -cano_smpl_center
    model_RT[2, 3] -= 10
    proj_mat = gl_orthographic_projection_matrix()
    back_mvp = np.dot(proj_mat, model_RT)
    back_mv = model_RT.copy()

    # render
    pose_ = pose.copy()
    pose_[:6] = 0.  # global transformation is set to zero
    pose_[3 + 22 * 3: 6 + 22 * 3] = 0.
    pose_[3 + 23 * 3: 6 + 23 * 3] = 0.
    posed_smpl = SmplModel(pose_, shape)
    posed_smpl.posed_vertices -= posed_smpl.posed_joints[0]
    posed_vertices = posed_smpl.posed_vertices[smpl_params.faces].astype(np.float32).reshape(-1, 3)
    cano_vertices = cano_smpl.posed_vertices[smpl_params.faces].astype(np.float32).reshape(-1, 3)
    attri_renderer.set_model(cano_vertices, posed_vertices)

    attri_renderer.set_mvp_mat(front_mvp)
    front_smpl_pos_map = attri_renderer.render()[:, :, :3]

    attri_renderer.set_mvp_mat(back_mvp)
    back_smpl_pos_map = attri_renderer.render()[:, :, :3]
    back_smpl_pos_map = cv.flip(back_smpl_pos_map, 1)

    smpl_pos_map = np.concatenate([front_smpl_pos_map, back_smpl_pos_map], 1)
    return smpl_pos_map


def calc_cano_weight_volume(shape):
    def get_grid_points(xyz):
        min_xyz = np.min(xyz, axis = 0)
        max_xyz = np.max(xyz, axis = 0)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        bounds = np.stack([min_xyz, max_xyz], axis = 0)
        vsize = 0.025
        voxel_size = [vsize, vsize, vsize]
        x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
        y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
        z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing = 'ij'), axis = -1)
        return pts

    from dataset.smpl import SmplModel, smpl_params

    cano_smpl = SmplModel(cano_smpl_pose, shape)
    cano_smpl_trimesh = trimesh.Trimesh(vertices = cano_smpl.posed_vertices, faces = smpl_params.faces, process = False, use_embree = True)

    # generate volume pts
    pts = get_grid_points(cano_smpl_trimesh.vertices)
    X, Y, Z, _ = pts.shape
    pts = pts.reshape(-1, 3)

    # barycentric
    dists, face_id, closest_pts = igl.signed_distance(pts, cano_smpl.posed_vertices, smpl_params.faces)
    dists = np.abs(dists)
    triangles = cano_smpl_trimesh.vertices[cano_smpl_trimesh.faces[face_id]]
    weights = smpl_params.weights[cano_smpl_trimesh.faces[face_id]]
    barycentric_weight = trimesh.triangles.points_to_barycentric(triangles, closest_pts)
    weights = (barycentric_weight[:, :, None] * weights).sum(1)
    weights[dists > 0.08] = 0.

    weights = weights.reshape(X, Y, Z, -1).astype(np.float32)
    return weights


def main(data_dir):
    scan_paths = sorted(glob.glob(data_dir + '/scan/*.ply'))
    pose_paths = sorted(glob.glob(data_dir + '/smpl/pose_*.txt'))
    shape = np.loadtxt(data_dir + '/smpl/shape.txt')

    wt_cano_scan_dir = data_dir + '/wt_cano_scan'
    os.makedirs(wt_cano_scan_dir, exist_ok = True)
    cano_pts_ov_dir = data_dir + '/cano_pts_ov'
    os.makedirs(cano_pts_ov_dir, exist_ok = True)

    # init renderers
    color_renderer = Renderer(512, 512, shader_name = 'vertex_attribute', window_name = 'Color Renderer')
    pos_renderer = Renderer(512, 512, shader_name = 'position', window_name = 'Position Renderer')
    attri_renderer = Renderer(256, 256, shader_name = 'vertex_attribute', window_name = 'Attribute Renderer')

    """
    0. compute canonical blend weight volume
    """
    weight_volume = calc_cano_weight_volume(shape)
    np.save('%s/cano_base_blend_weight_volume.npy' % data_dir, weight_volume)

    # rendering configuration
    camera = {
        'img_width': 512,
        'img_height': 512,
        'fx': 550.0,
        'fy': 550.0,
        'cx': 256.0,
        'cy': 256.0
    }
    view_num = 60
    for i, scan_path in enumerate(scan_paths):
        print('Processing data %d...' % i)
        scan = trimesh.load(scan_path, process = False)
        pose = np.loadtxt(pose_paths[i])

        idx = int(os.path.basename(scan_path).replace('.ply', ''))

        """
        1. canonicalization
        """
        wt_cano_scan, ori_sur_flag = canonicalize(scan, pose, shape, idx, tmp_dir = wt_cano_scan_dir)

        """
        2. sample canonical points
        """
        sur_pts, sur_pts_ov, vol_pts, vol_pts_ov = presample_pts(wt_cano_scan, ori_sur_flag)
        np.savez(cano_pts_ov_dir + '/%03d.npz' % idx,
                 sur_pts = sur_pts, sur_pts_ov = sur_pts_ov,
                 vol_pts = vol_pts, vol_pts_ov = vol_pts_ov)

        """
        3. render images
        """
        img_dir = data_dir + '/imgs/%03d' % idx
        os.makedirs(img_dir, exist_ok = True)
        render_images(color_renderer,
                      pos_renderer,
                      scan,
                      img_dir,
                      camera,
                      view_num)

        """
        4. render SMPL positional maps
        """
        smpl_pos_map = render_smpl_position_map(attri_renderer, pose, shape)
        cv.imwrite(data_dir + '/smpl/smpl_pos_map_%04d_cano.exr' % idx, smpl_pos_map)

    # export data configuration
    data_config = {
        'data_type': 'synthetic',
        'view_num': view_num,
        'camera': camera,
        'pos_map_name': 'cano',
        'pos_map_res': 256
    }
    yaml.dump(data_config, open(data_dir + '/dataConfig.yaml', 'w'), sort_keys = False)


if __name__ == '__main__':
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type = str, help = 'Data directory.')
    args = arg_parser.parse_args()

    main(args.data_dir)

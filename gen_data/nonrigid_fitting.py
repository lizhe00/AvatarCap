import torch
import pytorch3d
import trimesh
import math

device = 'cuda:0'


class NodeGraph():
    def __init__(self, mesh, cano_mesh = None, node_radius = 0.04):
        self.mesh = mesh
        self.cano_mesh = cano_mesh
        self.node_radius = node_radius
        self.cano_vertices = torch.from_numpy(self.mesh.vertices.copy()).to(torch.float32).to(device)
        self.live_vertices = self.cano_vertices.clone()
        self.cano_normals = torch.from_numpy(self.mesh.vertex_normals.copy()).to(torch.float32).to(device)
        self.live_normals = self.cano_normals.clone()

        if cano_mesh is None:
            self.sample_nodes(mesh)
        else:
            self.sample_nodes(cano_mesh)

    def sample_nodes(self, mesh):
        nodes, _ = trimesh.sample.sample_surface_even(self.mesh, self.mesh.vertices.shape[0], radius=self.node_radius)
        # nodes = mesh.vertices.copy()

        self.nodes = torch.from_numpy(nodes).to(torch.float32).to(device)
        node_num = self.nodes.shape[0]
        self.node_axisangle = torch.zeros((node_num, 3), dtype = torch.float32, device = device)  # rotation
        self.node_trans = torch.zeros((node_num, 3), dtype = torch.float32, device = device)  # translation
        self.node_axisangle.requires_grad_()
        self.node_trans.requires_grad_()
        print('sampled node number: %d' % node_num)

        # construct node graph
        _, indices, _ = pytorch3d.ops.knn_points(self.nodes.unsqueeze(0), self.nodes.unsqueeze(0), K = 9)
        indices = indices[:, :, 1:]
        self.node_neighbor_indices = indices.squeeze(0)

        # calculate knn indices for mesh vertices
        knn = 4
        sq_dists, indices, _ = pytorch3d.ops.knn_points(torch.from_numpy(mesh.vertices.copy()).to(torch.float32).to(device).unsqueeze(0),
                                                        self.nodes.unsqueeze(0), K = knn + 1)
        sq_dists = sq_dists[:, :, 1:]
        indices = indices[:, :, 1:]
        self.vertex_knn_indices = indices.squeeze(0)
        self.vertex_knn_weights = 1. / torch.sqrt(sq_dists.squeeze(0))
        self.vertex_knn_weights /= torch.sum(self.vertex_knn_weights, dim = -1, keepdim = True) + 1e-16

    def deform(self):
        vertex_knn_axisangle = pytorch3d.ops.knn_gather(self.node_axisangle.unsqueeze(0), self.vertex_knn_indices.unsqueeze(0)).squeeze(0)
        vertex_knn_translation = pytorch3d.ops.knn_gather(self.node_trans.unsqueeze(0), self.vertex_knn_indices.unsqueeze(0)).squeeze(0)
        vertex_knn_rotmat = pytorch3d.transforms.axis_angle_to_matrix(vertex_knn_axisangle)
        vertex_rotation = torch.sum(self.vertex_knn_weights[..., None, None] * vertex_knn_rotmat, dim = 1).squeeze(1)
        vertex_translation = torch.sum(self.vertex_knn_weights[..., None] * vertex_knn_translation, dim = 1).squeeze(1)

        self.live_vertices = torch.einsum('vij,vj->vi', vertex_rotation, self.cano_vertices) + vertex_translation
        self.live_normals = torch.einsum('vij,vj->vi', vertex_rotation, self.cano_normals)

    def save_canonical_model(self, path):
        self.mesh.export(path)

    def save_live_model(self, path):
        live_mesh = trimesh.Trimesh(vertices = self.live_vertices.detach().cpu().numpy(),
                                         faces = self.mesh.faces, process = False)
        live_mesh.export(path)

    def get_live_mesh(self):
        return trimesh.Trimesh(vertices = self.live_vertices.detach().cpu().numpy(), faces = self.mesh.faces, process = False)

    def save_node_graph(self, path):
        vertices = self.nodes.cpu().numpy()
        lines = self.node_neighbor_indices.cpu().numpy()
        with open(path, 'w') as fp:
            for i in range(vertices.shape[0]):
                v = vertices[i]
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for i in range(lines.shape[0]):
                for j in range(lines.shape[1]):
                    fp.write('l %d %d\n' % (i + 1, lines[i, j] + 1))
        fp.close()

    def construct_smooth_loss(self):
        node_neighbor_axisangle = pytorch3d.ops.knn_gather(self.node_axisangle.unsqueeze(0), self.node_neighbor_indices.unsqueeze(0)).squeeze(0)
        node_neighbor_translation = pytorch3d.ops.knn_gather(self.node_trans.unsqueeze(0), self.node_neighbor_indices.unsqueeze(0)).squeeze(0)
        node_rotation = pytorch3d.transforms.axis_angle_to_matrix(self.node_axisangle)
        live_nodes = torch.einsum('vij,vj->vi', node_rotation, self.nodes) + self.node_trans
        node_neighbor_rotation = pytorch3d.transforms.axis_angle_to_matrix(node_neighbor_axisangle)
        live_nodes_driven_by_neighbor = torch.einsum('vnij,vj->vni', node_neighbor_rotation, self.nodes) + node_neighbor_translation
        smooth_loss = live_nodes[:, None, :] - live_nodes_driven_by_neighbor
        smooth_loss = torch.square(smooth_loss).sum()

        return smooth_loss


def construct_icp_loss(src_vertices, src_normals, tar_vertices, tar_normals, dist_thres = 0.05, normal_thres = 0.5):
    K = 4
    _, knn_indices, knn_vertices = pytorch3d.ops.knn_points(src_vertices.unsqueeze(0), tar_vertices.unsqueeze(0), K = K, return_nn = True)
    knn_normals = pytorch3d.ops.knn_gather(tar_normals.unsqueeze(0), knn_indices)

    knn_vertices = knn_vertices.squeeze(0)
    knn_normals = knn_normals.squeeze(0)

    col_id = torch.zeros(knn_vertices.shape[0], dtype = torch.long).cuda()  # [0, 1, 2, ..., K - 1]
    valid_flag = torch.zeros(knn_vertices.shape[0], dtype = torch.bool).cuda()
    for i in range(K):
        dist_flag = torch.norm(src_vertices - knn_vertices[:, i], dim = -1) < dist_thres
        normal_flag = torch.einsum('vi,vi->v', src_normals, knn_normals[:, i]) > normal_thres
        flag = torch.logical_and(dist_flag, normal_flag)
        # flag = dist_flag
        col_id[torch.logical_and(~valid_flag, flag)] = i
        valid_flag = torch.logical_or(valid_flag, flag)

    col_id = col_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)
    tar_v = torch.gather(knn_vertices, 1, col_id).squeeze(1)[valid_flag]
    tar_n = torch.gather(knn_normals, 1, col_id).squeeze(1)[valid_flag]
    src_v = src_vertices[valid_flag]

    delta_v = src_v - tar_v
    v2n_dist = torch.einsum('vi,vi->v', delta_v, tar_n)
    v2n_dist = torch.square(v2n_dist).sum()
    return v2n_dist


def nonrigid_fitting(src_model, tar_model, cano_model, fix_vertex_indices = None, iteration_num = 200):
    src_node_graph = NodeGraph(src_model, cano_model, node_radius = 0.008)
    tar_v = torch.from_numpy(tar_model.vertices.copy()).to(torch.float32).to(device)
    tar_n = torch.from_numpy(tar_model.vertex_normals.copy()).to(torch.float32).to(device)

    optm = torch.optim.LBFGS([src_node_graph.node_axisangle, src_node_graph.node_trans], max_iter = 1)

    lambda_icp = 1.
    lambda_smooth = 0.5

    src_node_graph.deform()

    def closure():
        src_node_graph.deform()

        if fix_vertex_indices is not None:
            valid_vertices = src_node_graph.live_vertices[fix_vertex_indices]
            valid_normals = src_node_graph.live_normals[fix_vertex_indices]
        else:
            valid_vertices = src_node_graph.live_vertices
            valid_normals = src_node_graph.live_normals

        icp_loss = construct_icp_loss(valid_vertices,
                                      valid_normals,
                                      tar_v,
                                      tar_n,
                                      dist_thres = dist_thres,
                                      normal_thres = normal_thres)

        smooth_loss = src_node_graph.construct_smooth_loss()

        total_loss = lambda_icp * icp_loss + lambda_smooth * smooth_loss
        # print('ICP loss: %f, SMOOTH loss: %f' % (icp_loss.item(), smooth_loss.item()))

        optm.zero_grad()
        total_loss.backward(retain_graph = True)
        # optm.step()
        return total_loss

    for iter_idx in range(iteration_num):
        if iter_idx < 100:
            dist_thres = 0.1
            normal_thres = math.cos(math.pi / 4.)
        elif 100 <= iter_idx <= 250:
            dist_thres = 0.05
            normal_thres = math.cos(math.pi / 4.)
        else:
            dist_thres = 0.02
            normal_thres = math.cos(math.pi / 4.)

        optm.step(closure)

    return src_node_graph.get_live_mesh()


def subdivide(vertices, faces, iter_num = 1):
    for i in range(iter_num):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return vertices, faces

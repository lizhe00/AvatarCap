import numpy as np
from scipy import sparse
import os
import cv2 as cv
import pickle

import config


class SmplParams:
    """
    Smpl parameters.
    """
    def __init__(self, _model_path='./smpl_files/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'):
        self.model_path = _model_path

        data = pickle.load(open(self.model_path, 'rb'), encoding='latin1')

        # load mean vertices
        self.mean_vertices = data['v_template'].astype(np.float32)
        self.vnum = self.mean_vertices.shape[0]

        # load faces
        self.faces = data['f']
        self.faces = self.faces.astype(np.int32)
        self.fnum = self.faces.shape[0]

        # load joints
        self.joints = data['J'].astype(np.float32)

        # load kintree
        self.kintree = data['kintree_table'].astype(np.int32).transpose()
        self.joint_num = self.kintree.shape[0]

        # load blend weights
        self.weights = data['weights'].astype(np.float32)

        # load joint regressor
        self.sparse_regressor = data['J_regressor']

        # shape blending matrix
        self.shape_blend_shape = np.array(data['shapedirs'], dtype = np.float32).reshape(self.vnum * 3, -1)


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
smpl_params = SmplParams(PROJ_DIR + '/smpl_files/basicmodel_%s_lbs_10_207_0_v1.0.0.pkl' % config.smpl_gender)


class SmplModel:
    """
    SMPL model implemented by Numpy.
    """
    def __init__(self, _pose_coeff, _shape_coeff):
        self.pose_coeff = _pose_coeff.reshape(75, 1)
        self.shape_coeff = _shape_coeff.reshape(10, 1)

        self.shaped_vertices = None
        self.joints = None
        self.jnt_affine_mats = None
        self.posed_joints = None
        self.vertex_affine_mats = None
        self.posed_vertices = None

        self.change_shape()
        self.change_pose()

    def change_shape(self):
        # shape blend shape
        mean_vertices_vec = smpl_params.mean_vertices.reshape(smpl_params.vnum*3, 1)
        shaped_vertices_vec = mean_vertices_vec + np.dot(smpl_params.shape_blend_shape, self.shape_coeff)
        self.shaped_vertices = shaped_vertices_vec.reshape(-1, 3)

        # calculate joint
        self.joints = smpl_params.sparse_regressor * self.shaped_vertices

    def change_pose(self):
        # calculate local matrix
        local_mats = []
        for jidx in range(smpl_params.joint_num):
            theta = self.pose_coeff[3*jidx + 3: 3*jidx + 6]
            r = cv.Rodrigues(theta)[0]
            t = np.dot(np.identity(3)-r, self.joints[jidx].transpose())
            local_mat = np.identity(4)
            local_mat[0:3, 0:3] = r
            if jidx == 0:
                # local_mat[0:3,3] = local_mat[0:3,3] + self.pose_coeff[0:3,0]
                local_mat[0:3, 3] = self.pose_coeff[0:3, 0]
            else:
                local_mat[0:3, 3] = t
            local_mats.append(local_mat)

        # calculate each joint transformation matrix
        jnt_affine_mats = []
        jnt_affine_mats.append(local_mats[0])
        for jidx in range(1, smpl_params.joint_num):
            parent_idx = smpl_params.kintree[jidx, 0]
            jnt_affine_mat = np.dot(jnt_affine_mats[parent_idx], local_mats[jidx])
            jnt_affine_mats.append(jnt_affine_mat)

        # calculate joint positions
        self.jnt_affine_mats = np.array(jnt_affine_mats)
        self.posed_joints = np.zeros_like(smpl_params.joints)
        for jidx in range(smpl_params.joint_num):
            jnt_affine_mat = jnt_affine_mats[jidx]
            self.posed_joints[jidx] = np.dot(jnt_affine_mat[:3, :3], self.joints[jidx]) + jnt_affine_mat[:3, 3]

        # skinning vertex
        jnt_affine_mats = np.array(jnt_affine_mats)
        self.vertex_affine_mats = np.einsum('vj,jab->vab', smpl_params.weights, jnt_affine_mats)
        self.posed_vertices = np.einsum('vab,vb->va', self.vertex_affine_mats[:, :3, :3], self.shaped_vertices) + self.vertex_affine_mats[:, :3, 3]

    def save_mesh_as_obj(self, output_path):
        with open(output_path, 'w') as fp:
            for i in range(self.posed_vertices.shape[0]):
                fp.write('v %f %f %f\n' % (self.posed_vertices[i,0], self.posed_vertices[i,1], self.posed_vertices[i,2]))

            for i in range(smpl_params.fnum):
                fp.write('f %d %d %d\n' % (smpl_params.faces[i,0]+1, smpl_params.faces[i,1]+1, smpl_params.faces[i,2]+1))

    def save_skeleton_as_obj(self, output_path):
        with open(output_path, 'w') as fp:
            for jidx in range(smpl_params.joint_num):
                j = self.posed_joints[jidx]
                fp.write('v %f %f %f\n' % (j[0], j[1], j[2]))
            for jidx in range(1, smpl_params.joint_num):
                parent_idx = smpl_params.kintree[jidx, 0]
                fp.write('l %d %d\n' % (parent_idx + 1, jidx + 1))

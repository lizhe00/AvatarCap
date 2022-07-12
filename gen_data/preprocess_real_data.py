"""
Process real data.
Suppose the monocular image sequence (with mask) and SMPL are ready.

Layout of the given data before processing:
data_dir:
├── imgs
│   └── color
│      └── color_0000.jpg
│      └── color_0001.jpg
│   └── mask
│      └── mask_0000.png
│      └── mask_0001.png
│   └── camera.yaml
├── smpl
│   └── pose_0000.txt
│   └── pose_0001.txt
│   └── shape.txt

"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import torch
import cv2 as cv
import torchvision.transforms as transforms
import yaml
import glob

from network.networks import define_G
from dataset.smpl import SmplModel
from utils.renderer import Renderer
from gen_data.preprocess_training_data import render_smpl_position_map
import config


def load_pretrained_model(path, network):
    if os.path.exists(path):
        print('Loading network from %s' % path)
        data = torch.load(path)
        network.load_state_dict(data['network'])
    else:
        raise FileNotFoundError("%s is not available!" % path)


def main(data_dir,
         pretrained_model_path):
    cam = yaml.load(open(data_dir + '/imgs/camera.yaml', encoding = 'UTF-8'), Loader = yaml.FullLoader)
    os.makedirs(data_dir + '/imgs/normal', exist_ok = True)

    img_paths = sorted(glob.glob(data_dir + '/imgs/color/*.jpg'))
    mask_paths = sorted(glob.glob(data_dir + '/imgs/mask/*.png'))
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance").to(config.device)  # normal network
    load_pretrained_model(pretrained_model_path, netF)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    attri_renderer = Renderer(256, 256, shader_name = 'vertex_attribute', window_name = 'Attribute Renderer')

    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        frame_id = img_name.replace('.jpg', '').replace('color_', '')
        print('Processing frame %s' % frame_id)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = cv.imread(mask_paths[i], cv.IMREAD_UNCHANGED)
        img[mask == 0] = 0

        img_h, img_w, img_c = img.shape
        if img_c == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # crop
        pose = np.loadtxt(data_dir + '/smpl/pose_%s.txt' % frame_id)
        shape = np.loadtxt(data_dir + '/smpl/shape.txt')
        smpl = SmplModel(pose, shape)
        u = cam['fx'] * smpl.posed_vertices[:, 0] / smpl.posed_vertices[:, 2] + cam['cx']
        v = cam['fy'] * smpl.posed_vertices[:, 1] / smpl.posed_vertices[:, 2] + cam['cy']
        min_u, max_u = u.min(), u.max()
        min_v, max_v = v.min(), v.max()
        min_u = max(0, min_u - 10)
        max_u = min(img_w - 1, max_u + 10)
        min_v = max(0, min_v - 10)
        max_v = min(img_h - 1, max_v + 10)
        center = np.array([0.5 * (min_u + max_u), 0.5 * (min_v + max_v)])
        win_len = int(max(max_u - min_u, max_v - min_v))
        left_corner = (center - 0.5 * win_len).astype(np.int32)
        left_corner = np.clip(left_corner, 0, max(img_w, img_h))
        cropped_img = img[left_corner[1]: left_corner[1] + win_len, left_corner[0]: left_corner[0] + win_len]
        ori_shape = (cropped_img.shape[1], cropped_img.shape[0])
        cropped_img_512 = cv.resize(cropped_img, (512, 512), interpolation = cv.INTER_LINEAR)

        with torch.no_grad():
            cropped_img_512 = to_tensor(cropped_img_512).to(config.device)
            normal_img = netF(cropped_img_512.unsqueeze(0))

        normal_img = normal_img.squeeze(0).permute((1, 2, 0))
        normal_img_512 = normal_img.cpu().numpy()

        normal_img = np.zeros((img_h, img_w, 3), np.float32)
        cropped_normal_img = cv.resize(normal_img_512, ori_shape, interpolation = cv.INTER_LINEAR)
        normal_img[left_corner[1]: left_corner[1] + win_len, left_corner[0]: left_corner[0] + win_len] = cropped_normal_img

        normal_img[mask < 1e-6] = 0
        normal_img_path = data_dir + '/imgs/normal/normal_%s.exr' % frame_id
        cv.imwrite(normal_img_path, normal_img)

        smpl_pos_map = render_smpl_position_map(attri_renderer, pose, shape)
        cv.imwrite(data_dir + '/smpl/smpl_pos_map_%s_cano.exr' % frame_id, smpl_pos_map)

    # export data configuration
    data_config = {
        'data_type': 'real',
        'view_num': 1,
        'camera': cam,
        'pos_map_name': 'cano',
        'pos_map_res': 256
    }
    yaml.dump(data_config, open(data_dir + '/dataConfig.yaml', 'w'), sort_keys = False)


if __name__ == '__main__':
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type = str, help = 'Data directory.')
    arg_parser.add_argument('--normal_net', type = str, help = 'Normal network path.')
    args = arg_parser.parse_args()

    main(args.data_dir,
         args.normal_net)

import torch

device = torch.device('cuda')

# SMPL gender, required to be specified
smpl_gender = 'M'  # 'F' (female), 'M' (male) or 'N' (neutral)

# nerf-related
N_samples = 64
perturb = 1

# implicit field type
if_type = 'sdf'
# if_type = 'occupancy'

if if_type == 'sdf':
    iso_value = 0.
    sdf_thres = 0.1
elif if_type == 'occupancy':
    iso_value = 0.5
else:
    raise ValueError('Invalid if_type!')


cfg = dict()  # configurations from yaml file


def load_config(path):
    import yaml
    data = yaml.load(open(path, encoding = 'UTF-8'), Loader=yaml.FullLoader)
    return data

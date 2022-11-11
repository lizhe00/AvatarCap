# Data Preprocessing

Example dataset can be found in [README.md](../README.md). Please check it for data organization and format.

## Additional Requirement
- [libigl](https://libigl.github.io/)

## Training Data
1. Input: 3D scans & fitted SMPL poses.
2. The input data is organized as the following structure:
```
training_data_dir
├── scan
│   └── 000.ply
│   └── 001.ply
├── smpl
│   └── pose_000.txt
│   └── pose_001.txt
│   └── shape.txt
```
3. 3D scans (```*.ply```) contain vertex colors as texture, the SMPL pose in ```pose_*.txt``` is defined as ```[global_trans, global_rot, joint_1_rot, ...]```,
where ```global_trans, global_rot, joint_*_rot``` are all 3-dimensional vectors.
4. Download [PoissonRecon.exe](http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version13.8/AdaptiveSolvers.x64.zip), and place it in ```./gen_data/bin```.
5. Run the following script to process the training data.
```
python -m gen_data.preprocess_training_data --data_dir=training_data_dir
```
6. Note that this code uses the executable of [Poisson Reconstruction](https://github.com/mkazhdan/PoissonRecon), so it can only run on Windows.


## Testing Data
1. Input: image sequence & fitted SMPL poses.
2. The input data is organized as the following structure.
```
testing_data_dir
├── imgs
│   └── color
│      └── color_0000.jpg
│   └── mask
│      └── mask_0000.png
│   └── camera.yaml
├── smpl
│   └── pose_0000.txt
│   └── shape.txt
```
3. The ```smpl``` folder is the same as that in the training data, and the ```imgs``` folder contains color & mask sequences and camera information (```camera.yaml```).
4. Make sure that the normal network checkpoint is in ```./pretrained_models/normal_net```. This checkpoint is extracted from the trained model of [PIFuHD](https://github.com/facebookresearch/pifuhd). Many thanks to the authors!
5. Run the following script to process the testing data.
```
python -m gen_data.preprocess_real_data --data_dir=testing_data_dir --normal_net=./pretrained_models/normal_net/netF.pth
```
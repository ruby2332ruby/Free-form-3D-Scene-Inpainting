# Free-form-3D-Scene-Inpainting
## Code
### Installation: 
Training is implemented with [PyTorch](https://pytorch.org/)
* Python 2.7
* CUDA 10.0
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.2.0

For conda virtual env:
```
conda create --name dualGAN python=2.7
conda activate dualGAN
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd dual_stream_GAN/torch
sh install_utils.sh
conda install -c conda-forge imageio
conda install -c conda-forge plyfile
conda install scikit-image
conda install -c menpo pathlib
pip install wandb
```

For other users, please install all above packages following [SPSG](https://github.com/angeladai/spsg) and compile the extension modules by running the `install_utils.sh` script.

### Training:
* See `python train.py --help` for all train options. 
* Example command: 
```
train.py --gpu 0 --data_path path_to_ff_dataset(data-geo-color-ffchole) --save path_to_save_checkpoint_folder --frame_info_path path_to_frame_info_file(data-frames) --frame_path path_to_frame_file(images) --max_epoch 6 --weight_missing_geo 12 --weight_missing_color 12 --weight_color_loss 0.6 --weight_sdf_loss 0.3 --weight_content_loss 0.01 --color_space lab
```
* Trained model: [dual_stream_GAN.pth](https://github.com/ruby2332ruby/Free-form-3D-Scene-Inpainting/tree/main/torch/pretrained_model) (7.5M)

### Testing:
* See `python test_scene_as_chunks.py --help` for all train options. 
* Example command: 
```
test_scene_as_chunks.py --gpu 0 --input_data_path path_to_input_ff_dataset(mp_sdf_2cm_input_ffchole) --target_data_path path_to_target_ff_dataset(mp_sdf_2cm_target) --test_file_list ../filelists/mp-rooms_test_ff-scenes.txt --model_path path_to_model_to_test --output path_to_output_folder --color_space lab
```

### Data:

### Data Generation:
* Dowload [SPSG](https://github.com/angeladai/spsg) Matterport3D TSDF scene dataset.
* For training dataset generation, see `datagen/train_freeform_mask_gen.py`:
```
python train_freeform_mask_gen.py --input path_to_spsg_dataset(data-geo-color) --output path_to_output_dataset
```
* Testing dataset generation, see `datagen/test_freeform_mask_gen.py`:
```
python test_freeform_mask_gen.py --input path_to_spsg_dataset(mp_sdf_2cm_target) --output path_to_output_dataset

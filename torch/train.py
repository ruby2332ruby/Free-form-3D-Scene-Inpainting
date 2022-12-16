from __future__ import division #python2
from __future__ import print_function #python2

import argparse
import os, sys, time
import shutil
import random
import torch
import numpy as np
import gc
import torchvision.models
import wandb
import sys #wandb
reload(sys) #wandb
sys.setdefaultencoding('utf8') #wandb

import data_util
import scene_dataloader
import model_gated_dual_stream_GAN as model_util
import model_disc_edge as model_discedge #cannydisc #for 2D Edge Discriminator (include 2d edge detection)
import loss as loss_util
import style

from utils.raycast_rgbd.raycast_rgbd import RaycastRGBD
from utils.raycast_rgbd.raycast_rgbd import RaycastOcc
from utils.depth_utils.depth_utils import Depth2Normals
import utils.color_utils_cpu.color_utils as color_utils

COLOR_MEAN = [0.485, 0.456, 0.406] # for vgg19
COLOR_STD = [0.229, 0.224, 0.225]

GAN_TYPES = ['hinge', 'vanilla', 'wgan', 'wgan_gp']
COLOR_SPACES = ['rgb', 'lab']

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data_path', required=True, help='path to data')
parser.add_argument('--train_file_list', default='../filelists/mp_train_ff.txt', help='path to file list of train data')
parser.add_argument('--val_file_list', default='../filelists/mp_val_ff.txt', help='path to file list of val data')
parser.add_argument('--save', default='./logs', help='folder to output model checkpoints')
parser.add_argument('--max_num_trainval_chunks', type=int, default=0, help='limit #train or #val chunks (0 to disable)')
parser.add_argument('--num_workers', type=int, default=2, help='')
# model params
parser.add_argument('--retrain', type=str, default='', help='model to load from')
parser.add_argument('--retrain_disc', type=str, default='', help='model to load from (disc)')
parser.add_argument('--input_dim', type=int, default=0, help='#points / voxel dim.')
# train params
parser.add_argument('--voxelsize', type=float, default=0.02, help='voxel size in meters.')
parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--max_epoch', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--d_lr_factor', type=float, default=4, help='lr disc = d_lr_factor*lr')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument('--num_iters_geo_only', type=int, default=1000, help='#iters to train geo before introducing color')
parser.add_argument('--num_iters_before_content', type=int, default=60000, help='#iters to train geo before introducing color')
parser.add_argument('--weight_occ_loss', type=float, default=1.0, help='weight geo loss vs rest (0 to disable).')
parser.add_argument('--weight_depth_loss', type=float, default=1.0, help='weight geo loss vs rest (0 to disable).')
parser.add_argument('--weight_sdf_loss', type=float, default=0.3, help='weight geo loss vs rest (0 to disable).')
parser.add_argument('--weight_color_loss', type=float, default=0.6, help='weight color loss vs rest (0 to disable).')
parser.add_argument('--color_thresh', type=int, default=15, help='mask colors with all values < color_thresh')
parser.add_argument('--start_iter', type=int, default=0, help='start iteration')
parser.add_argument('--color_truncation', type=float, default=0, help='truncation in voxels for color')
parser.add_argument('--no_color_augmentation', dest='augment_rgb_scaling', action='store_false')
parser.add_argument('--augment_scale_min', type=float, default=0.5, help='for color augmentation')
parser.add_argument('--augment_scale_max', type=float, default=1.5, help='for color augmentation')
parser.add_argument('--color_space', type=str, default='lab', help='[rgb, lab]')
parser.add_argument('--input_mask', type=int, default=1, help='input mask')
# geo pred
parser.add_argument('--no_logweight_sdf', dest='logweight_sdf', action='store_false')
parser.add_argument('--weight_missing_geo', type=float, default=12.0, help='per-voxel weight for missing geo')
parser.add_argument('--weight_missing_color', type=float, default=12.0, help='per-voxel weight for missing color')
parser.add_argument('--weight_surf_geo', type=float, default=1.0, help='per-voxel weight for surf geo')
parser.add_argument('--no_pass_geo_feats', dest='pass_geo_feats', action='store_false')
# 2d proj part
parser.add_argument('--weight_style_loss', type=float, default=0.0, help='weight style loss vs rest (0 to disable).')
parser.add_argument('--weight_content_loss', type=float, default=0.01, help='weight content loss vs rest (0 to disable).')
parser.add_argument('--frame_info_path', type=str, required=True, help='path to frame info files')
parser.add_argument('--frame_path', type=str, required=True, help='path to frame files')
parser.add_argument('--style_width', type=int, default=320, help='width of input for 2d style')
parser.add_argument('--style_height', type=int, default=256, help='height of input for 2d style')
parser.add_argument('--randomize_frames', type=int, default=1, help='randomize frames for 2d associations')
parser.add_argument('--subsample_tgt2d_factor', type=int, default=1, help='height of input for 2d style')
parser.add_argument('--max_depth_fill_iters', type=int, default=40, help='max #fill iters for depth filling (0 for no fill)')
# adversarial part
parser.add_argument('--weight_disc_loss', type=float, default=1.0, help='weight disc loss.')
parser.add_argument('--weight_discgen_loss', type=float, default=0.005, help='weight disc loss.')
parser.add_argument('--nf_disc', type=int, default=8, help='controls #channels of discriminator')
parser.add_argument('--nf_gen', type=int, default=20, help='controls #channels of generator')
parser.add_argument('--no_project_targets', dest='project_targets', action='store_false')
parser.add_argument('--no_patch_disc', dest='patch_disc', action='store_false')
parser.add_argument('--patch_size', type=int, default=96, help='patch size for patch disc')
parser.add_argument('--disc_loss_type', type=str, default='vanilla', help='loss type for gan')
parser.add_argument('--valid_thresh', type=float, default=0.1, help='valid thresh 2d')
parser.add_argument('--no_filter_proj_tgt', dest='filter_proj_tgt', action='store_false')
parser.add_argument('--weight_by_percent_pixels', dest='weight_by_percent_pixels', action='store_true')
# random seed
parser.add_argument('--random_seed', type=int, default=2022, help='random seed for random, np, torch')
#cannydisc
parser.add_argument('--weight_discedge_loss', type=float, default=1.0, help='weight disc loss.')
parser.add_argument('--weight_discgenedge_loss', type=float, default=0.005, help='weight disc loss.')
parser.add_argument('--retrain_discedge', type=str, default='', help='model to load from (discedge)')
#dpl
parser.add_argument('--parallel', action='store_true', help="Multi-GPU Training")
#known
parser.add_argument('--known', type=float, default=3, help='known truncation for sdf')

parser.set_defaults(logweight_sdf=True, use_loss_masking=True, augment_rgb_scaling=True, pass_geo_feats=True, project_targets=True, patch_disc=True, filter_proj_tgt=True, weight_by_percent_pixels=False)
args = parser.parse_args()
assert(args.disc_loss_type in GAN_TYPES)
assert(args.color_space in COLOR_SPACES)
if args.input_dim == 0: # set default values
    args.input_dim = (128, 64, 64)
args.input_nf = 4
UP_AXIS = 0
_SPLITTER = ','

#wandb
wandb.init(
    project='your project name', #please filled in yourself
    name='your run name') #please filled in yourself
config = wandb.config
config.update(args)
#wandb

print(args)
# specify gpu
if args.parallel:
    os.environ['CUDA_VISIBLE_DEVICES']="0, 1" #dpl
    print("[cml18] CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"]) #dpl
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    print("[cml18] CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

random_seed = args.random_seed #default 2022
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
print("fixed random seed:", random_seed)
torch.backends.cudnn.benchmark = True


# style part
model_style = None
raycaster_rgbd = None
gan_loss = None
#lab2rgb_converter = None
if args.weight_disc_loss > 0 or args.weight_style_loss > 0 or args.weight_content_loss > 0 or args.weight_depth_loss > 0:
    if args.weight_style_loss > 0 or args.weight_content_loss > 0:
        model_style = torchvision.models.vgg19(pretrained=True).cuda().eval()
        model_style = style.Model(model_style, COLOR_MEAN, COLOR_STD)
    ray_increment = 0.3 * args.truncation
    thresh_sample_dist = 50.5 * ray_increment
    max_num_locs_per_sample = 640000
    raycast_depth_max = 6.0
    raycaster_rgbd = RaycastRGBD(args.batch_size, args.input_dim, args.style_width, args.style_height, depth_min=0.1/args.voxelsize, depth_max=raycast_depth_max/args.voxelsize, thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment, max_num_locs_per_sample=max_num_locs_per_sample)
    depth2normals = Depth2Normals(args.batch_size, args.style_width, args.style_height, depth_min=0.1/args.voxelsize, depth_max=raycast_depth_max/args.voxelsize, max_num_fill_iters=args.max_depth_fill_iters)
    
    raycaster_occ = RaycastOcc(args.batch_size, args.input_dim, args.style_width, args.style_height, depth_min=0.1/args.voxelsize, depth_max=4.0/args.voxelsize, ray_increment=ray_increment)
    
    gan_loss = loss_util.GANLoss(args.disc_loss_type)

# create model
nf_in_color = 3+1 if args.input_mask == 0 else 4+1 #gated
print("nf_in_color", nf_in_color)
#model = model_util.Generator(nf_in_geo=1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats, truncation=args.truncation, max_data_size=args.input_dim).cuda() #no_ff_mask
model = model_util.Generator(nf_in_geo=1+1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats, truncation=args.truncation, max_data_size=args.input_dim).cuda() #gated
wandb.watch(model) #wandb
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
disc = None
if args.weight_disc_loss > 0:
    nf_in = 3*(args.weight_depth_loss>0) + 3*(args.weight_color_loss>0) + (3*(args.weight_depth_loss>0) + 3*(args.weight_color_loss>0))
    disc = model_util.Discriminator2D(nf_in=nf_in, nf=args.nf_disc, patch_size=args.patch_size, image_dims=(args.style_height, args.style_width), patch=args.patch_disc, use_bias=True, disc_loss_type=args.disc_loss_type).cuda()
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=args.d_lr_factor*args.lr, weight_decay=args.weight_decay)
    wandb.watch(disc) #wandb
#cannydisc
if args.weight_discedge_loss > 0:
    nf_in_edge = 3 + 1 + 3 + 1 #grayscale_image, 2D edge #labdisc
    discedge = model_discedge.Discriminator2DEdge(nf_in=nf_in_edge, nf=args.nf_disc, patch_size=args.patch_size, image_dims=(args.style_height, args.style_width), patch=args.patch_disc, use_bias=True, disc_loss_type=args.disc_loss_type).cuda()
    optimizer_discedge = torch.optim.Adam(discedge.parameters(), lr=args.d_lr_factor*args.lr, weight_decay=args.weight_decay)
    wandb.watch(discedge) #wandb
#cannydisc
if args.retrain:
    print('loading model:', args.retrain)
    checkpoint = torch.load(args.retrain)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
if args.weight_disc_loss > 0 and args.retrain_disc:
    disc_checkpoint = torch.load(args.retrain_disc)
    disc.load_state_dict(disc_checkpoint['state_dict']) #, strict=False)
    optimizer_disc.load_state_dict(disc_checkpoint['optimizer'])
#cannydisc
if args.weight_discedge_loss > 0 and args.retrain_discedge:
    discedge_checkpoint = torch.load(args.retrain_discedge)
    discedge.load_state_dict(discedge_checkpoint['state_dict']) #, strict=False)
    optimizer_discedge.load_state_dict(discedge_checkpoint['optimizer'])
#cannydisc
if args.parallel:
    assert torch.cuda.device_count() > 1 #dpl
    model = torch.nn.DataParallel(model) #dpl
    if args.weight_disc_loss > 0:
        disc = torch.nn.DataParallel(disc) #dpl
    if args.weight_discedge_loss > 0:
        discedge = torch.nn.DataParallel(discedge) #dpl

# data files
train_files, val_files, _OVERFIT = data_util.get_train_files(args.data_path, args.train_file_list, args.val_file_list, args.max_num_trainval_chunks)
num_overfit_train = 0 if not _OVERFIT else 640
num_overfit_val = 0 if not _OVERFIT else 160
if _OVERFIT:
    args.use_loss_masking = False
    print('overfit -> no loss masking')
print('#train files = ', len(train_files))
print('#val files = ', len(val_files))
train_dataset = scene_dataloader.SceneDataset(train_files, args.input_dim, args.truncation, args.use_loss_masking, args.augment_rgb_scaling, (args.augment_scale_min, args.augment_scale_max), args.color_truncation, args.color_space, args.frame_info_path, args.frame_path, image_dims=(args.style_width, args.style_height), load_depth=args.weight_depth_loss > 0, subsamp2d_factor=args.subsample_tgt2d_factor, randomize_frames=args.randomize_frames, num_overfit=num_overfit_train)
print('train_dataset', len(train_dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=scene_dataloader.collate_voxels)
if len(val_files) > 0:
    val_dataset = scene_dataloader.SceneDataset(val_files, args.input_dim, args.truncation, args.use_loss_masking, args.augment_rgb_scaling, (args.augment_scale_min, args.augment_scale_max), args.color_truncation, args.color_space, args.frame_info_path, args.frame_path, image_dims=(args.style_width, args.style_height), load_depth=args.weight_depth_loss > 0, subsamp2d_factor=args.subsample_tgt2d_factor, randomize_frames=args.randomize_frames, num_overfit=num_overfit_val)
    print('val_dataset', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=scene_dataloader.collate_voxels)


def make_log_header_list(id):
    headers = ['%s_loss(total)' % id, '%s_loss(occ)' % id, '%s_iou(occ)' % id, '%s_loss(sdf)' % id, '%s_loss(depth)' % id, '%s_loss(color)' % id, '%s_loss(disc)' % id, '%s_loss(disc-real)' % id, '%s_loss(disc-fake)' % id, '%s_loss(gen)' % id, '%s_loss(discedge)' % id, '%s_loss(discedge-real)' % id, '%s_loss(discedge-fake)' % id, '%s_loss(genedge)' % id, '%s_loss(style)' % id, '%s_loss(content)' % id, 'time'] #cannydisc
    return headers
def write_header(log_file, log_file_val):
    headers = ['epoch','iter']
    headers.extend(make_log_header_list('train'))
    log_file.write(_SPLITTER.join(headers) + '\n')
    log_file.flush()
    
    headers = headers[:-1]
    headers.extend(make_log_header_list('val'))
    log_file_val.write(_SPLITTER.join(headers) + '\n')
    log_file_val.flush()
    return
  
def print_log_info(epoch, iter, mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss, mean_val_loss, mean_val_occloss, mean_val_occiou, mean_val_sdfloss, mean_val_depthloss, mean_val_colorloss, mean_val_discloss, mean_val_discloss_real, mean_val_discloss_fake, mean_val_genloss, mean_val_discedgeloss, mean_val_discedgeloss_real, mean_val_discedgeloss_fake, mean_val_genedgeloss, mean_val_styleloss, mean_val_contentloss, time, log): #cannydisc
    splitters = ['Epoch: ', ' iter: '] if log is None else ['', ',']
    values = [epoch, iter]
    values.extend([mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss]) #cannydisc
    if log is None:
        splitters.extend([' loss_train: ', ' loss_train(occ): ', ' iou_train(occ): ', ' loss_train(sdf): ', ' loss_train(depth): ', ' loss_train(color): ', ' loss_train(disc): ', ' loss_train(disc-real): ', ' loss_train(disc-fake): ', ' loss_train(gen): ', ' loss_train(discedge): ', ' loss_train(discedge-real): ', ' loss_train(discedge-fake): ', ' loss_train(genedge): ', ' loss_train(style): ', ' loss_train(content): ']) #cannydisc
    else:
        splitters.extend([','] * (12+4)) #cannydisc
    if mean_val_loss is not None:
        values.extend([mean_val_loss, mean_val_occloss, mean_val_occiou, mean_val_sdfloss, mean_val_depthloss, mean_val_colorloss, mean_val_discloss, mean_val_discloss_real, mean_val_discloss_fake, mean_val_genloss, mean_val_discedgeloss, mean_val_discedgeloss_real, mean_val_discedgeloss_fake, mean_val_genedgeloss, mean_val_styleloss, mean_val_contentloss]) #cannydisc
        if log is None:
            splitters.extend([' loss_val: ', ' loss_val(occ): ', ' iou_val(occ): ', ' loss_val(sdf): ', ' loss_val(depth): ', ' loss_val(color): ', ' loss_val(disc): ', ' loss_val(disc-real): ', ' loss_val(disc-fake): ', ' loss_val(gen): ', ' loss_val(discedge): ', ' loss_val(discedge-real): ', ' loss_val(discedge-fake): ', ' loss_val(genedge): ', ' loss_val(style): ', ' loss_val(content): ']) #cannydisc
        else:
            splitters.extend([','] * (12+4)) #cannydisc
    else:
        splitters.extend([''] * 4)
        values.extend([''] * 4)
    values.append(time)
    if log is None:
        splitters.append(' time: ')
    else:
        splitters.append(',')
    info = ''
    for k in range(len(splitters)):
        if log is None and isinstance(values[k], float):
            info += splitters[k] + '{:.6f}'.format(values[k])
        else:
            info += splitters[k] + str(values[k])
    if log is None:
        print(info, file=sys.stdout)
        if mean_val_loss is not None: #wandb
            #wandb
            wandb.log({
                "val_losses": mean_val_loss,
                "val_lossocc": mean_val_occloss, "val_iouocc": mean_val_occiou, "val_losssdf": mean_val_sdfloss,
                "val_lossdepth": mean_val_depthloss, "val_losscolor": mean_val_colorloss,
                "val_lossdisc": mean_val_discloss, "val_lossdisc_real": mean_val_discloss_real, "val_lossdisc_fake": mean_val_discloss_fake, "val_lossgen": mean_val_genloss,
                "val_lossdiscedge": mean_val_discedgeloss, "val_lossdiscedge_real": mean_val_discedgeloss_real, "val_lossdiscedge_fake": mean_val_discedgeloss_fake, "val_lossgenedge": mean_val_genedgeloss,
                "val_lossstyle": mean_val_styleloss,"val_losscontent": mean_val_contentloss,
            })
            #wandb
        #train
        #wandb
        wandb.log({
            "train_losses": mean_train_loss,
            "train_lossocc": mean_train_occloss, "train_iouocc": mean_train_occiou, "train_losssdf": mean_train_sdfloss,
            "train_lossdepth": mean_train_depthloss, "train_losscolor": mean_train_colorloss,
            "train_lossdisc": mean_train_discloss, "train_lossdisc_real": mean_train_discloss_real, "train_lossdisc_fake": mean_train_discloss_fake, "train_lossgen": mean_train_genloss,
            "train_lossdiscedge": mean_train_discedgeloss, "train_lossdiscedge_real": mean_train_discedgeloss_real, "train_lossdiscedge_fake": mean_train_discedgeloss_fake, "train_lossgenedge": mean_train_genedgeloss,
            "train_lossstyle": mean_train_styleloss,"train_losscontent": mean_train_contentloss,
        })
        #wandb
    else:
        print(info, file=log)

def print_log(log, epoch, iter, train_losses, train_lossoccs, train_iouoccs, train_losssdfs, train_lossdepths, train_losscolors, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, train_lossdiscedge_fake, train_lossgenedge, train_lossstyles, train_losscontents, val_losses, val_lossoccs, val_iouoccs, val_losssdfs, val_lossdepths, val_losscolors, val_lossdisc, val_lossdisc_real, val_lossdisc_fake, val_lossgen, val_lossdiscedge, val_lossdiscedge_real, val_lossdiscedge_fake, val_lossgenedge, val_lossstyles, val_losscontents, time): #cannydisc
    train_losses = np.array(train_losses)
    train_lossoccs = np.array(train_lossoccs)
    train_iouoccs = np.array(train_iouoccs)
    train_losssdfs = np.array(train_losssdfs)
    train_lossdepths = np.array(train_lossdepths)
    train_losscolors = np.array(train_losscolors)
    train_lossstyles = np.array(train_lossstyles)
    train_losscontents = np.array(train_losscontents)
    train_lossdisc = np.array(train_lossdisc)
    train_lossdisc_real = np.array(train_lossdisc_real)
    train_lossdisc_fake = np.array(train_lossdisc_fake)
    train_lossdiscedge = np.array(train_lossdiscedge) #cannydisc
    train_lossdiscedge_real = np.array(train_lossdiscedge_real) #cannydisc
    train_lossdiscedge_fake = np.array(train_lossdiscedge_fake) #cannydisc
    mean_train_loss = -1 if (len(train_losses) == 0 or np.all(train_losses < 0)) else np.mean(train_losses[train_losses >= 0])
    mean_train_occloss = -1 if (len(train_lossoccs) == 0 or np.all(train_lossoccs < 0)) else np.mean(train_lossoccs[train_lossoccs >= 0])
    mean_train_occiou = -1 if (len(train_iouoccs) == 0 or np.all(train_iouoccs < 0)) else np.mean(train_iouoccs[train_iouoccs >= 0])
    mean_train_sdfloss = -1 if (len(train_losssdfs) == 0 or np.all(train_losssdfs < 0)) else np.mean(train_losssdfs[train_losssdfs >= 0])
    mean_train_depthloss = -1 if (len(train_lossdepths) == 0 or np.all(train_lossdepths < 0)) else np.mean(train_lossdepths[train_lossdepths >= 0])
    mean_train_colorloss = -1 if (len(train_losscolors) == 0 or np.all(train_losscolors < 0)) else np.mean(train_losscolors[train_losscolors >= 0])
    mean_train_discloss = -1 if len(train_lossdisc) == 0 else np.mean(train_lossdisc)
    mean_train_discloss_real = -1 if len(train_lossdisc_real) == 0 else np.mean(train_lossdisc_real)
    mean_train_discloss_fake = -1 if len(train_lossdisc_fake) == 0 else np.mean(train_lossdisc_fake)
    mean_train_genloss = -1 if len(train_lossgen) == 0 else np.mean(train_lossgen)
    mean_train_discedgeloss = -1 if len(train_lossdiscedge) == 0 else np.mean(train_lossdiscedge) #cannydisc
    mean_train_discedgeloss_real = -1 if len(train_lossdiscedge_real) == 0 else np.mean(train_lossdiscedge_real) #cannydisc
    mean_train_discedgeloss_fake = -1 if len(train_lossdiscedge_fake) == 0 else np.mean(train_lossdiscedge_fake) #cannydisc
    mean_train_genedgeloss = -1 if len(train_lossgenedge) == 0 else np.mean(train_lossgenedge) #cannydisc
    mean_train_styleloss = -1 if (len(train_lossstyles) == 0 or np.all(train_lossstyles < 0)) else np.mean(train_lossstyles[train_lossstyles >= 0])
    mean_train_contentloss = -1 if (len(train_losscontents) == 0 or np.all(train_losscontents < 0)) else np.mean(train_losscontents[train_losscontents >= 0])
    mean_val_loss = None
    mean_val_occloss = None
    mean_val_occiou = None
    mean_val_sdfloss = None
    mean_val_depthloss = None
    mean_val_colorloss = None
    mean_val_discloss = None
    mean_val_discloss_real = None
    mean_val_discloss_fake = None
    mean_val_genloss = None
    mean_val_discedgeloss = None #cannydisc
    mean_val_discedgeloss_real = None #cannydisc
    mean_val_discedgeloss_fake = None #cannydisc
    mean_val_genedgeloss = None #cannydisc
    mean_val_styleloss = None
    mean_val_contentloss = None
    if val_losses:
        val_losses = np.array(val_losses)
        val_lossoccs = np.array(val_lossoccs)
        val_iouoccs = np.array(val_iouoccs)
        val_losssdfs = np.array(val_losssdfs)
        val_lossdepths = np.array(val_lossdepths)
        val_losscolors = np.array(val_losscolors)
        val_lossstyles = np.array(val_lossstyles)
        val_losscontents = np.array(val_losscontents)
        val_lossdisc = np.array(val_lossdisc)
        val_lossdisc_real = np.array(val_lossdisc_real)
        val_lossdisc_fake = np.array(val_lossdisc_fake)
        val_lossdiscedge = np.array(val_lossdiscedge) #cannydisc
        val_lossdiscedge_real = np.array(val_lossdiscedge_real) #cannydisc
        val_lossdiscedge_fake = np.array(val_lossdiscedge_fake) #cannydisc
        mean_val_loss = -1 if (len(val_losses) == 0 or np.all(val_losses < 0))  else np.mean(val_losses[val_losses >= 0])
        mean_val_occloss = -1 if (len(val_lossoccs) == 0 or np.all(val_lossoccs < 0))  else np.mean(val_lossoccs[val_lossoccs >= 0])
        mean_val_occiou = -1 if (len(val_iouoccs) == 0 or np.all(val_iouoccs < 0))  else np.mean(val_iouoccs[val_iouoccs >= 0])
        mean_val_sdfloss = -1 if (len(val_losssdfs) == 0 or np.all(val_losssdfs < 0))  else np.mean(val_losssdfs[val_losssdfs >= 0])
        mean_val_depthloss = -1 if (len(val_lossdepths) == 0 or np.all(val_lossdepths < 0))  else np.mean(val_lossdepths[val_lossdepths >= 0])
        mean_val_colorloss = -1 if (len(val_losscolors) == 0 or np.all(val_losscolors < 0))  else np.mean(val_losscolors[val_losscolors >= 0])
        mean_val_discloss = -1 if len(val_lossdisc) == 0 else np.mean(val_lossdisc)
        mean_val_discloss_real = -1 if len(val_lossdisc_real) == 0 else np.mean(val_lossdisc_real)
        mean_val_discloss_fake = -1 if len(val_lossdisc_fake) == 0 else np.mean(val_lossdisc_fake)
        mean_val_genloss = -1 if len(val_lossgen) == 0 else np.mean(val_lossgen)
        mean_val_discedgeloss = -1 if len(val_lossdiscedge) == 0 else np.mean(val_lossdiscedge) #cannydisc
        mean_val_discedgeloss_real = -1 if len(val_lossdiscedge_real) == 0 else np.mean(val_lossdiscedge_real) #cannydisc
        mean_val_discedgeloss_fake = -1 if len(val_lossdiscedge_fake) == 0 else np.mean(val_lossdiscedge_fake) #cannydisc
        mean_val_genedgeloss = -1 if len(val_lossgenedge) == 0 else np.mean(val_lossgenedge) #cannydisc
        mean_val_styleloss = -1 if (len(val_lossstyles) == 0 or np.all(val_lossstyles < 0)) else np.mean(val_lossstyles[val_lossstyles >= 0])
        mean_val_contentloss = -1 if (len(val_losscontents) == 0 or np.all(val_losscontents < 0)) else np.mean(val_losscontents[val_losscontents >= 0])
        print_log_info(epoch, iter, mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss, mean_val_loss, mean_val_occloss, mean_val_occiou, mean_val_sdfloss, mean_val_depthloss, mean_val_colorloss, mean_val_discloss, mean_val_discloss_real, mean_val_discloss_fake, mean_val_genloss, mean_val_discedgeloss, mean_val_discedgeloss_real, mean_val_discedgeloss_fake, mean_val_genedgeloss, mean_val_styleloss, mean_val_contentloss, time, None) #cannydisc
        print_log_info(epoch, iter, mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss, mean_val_loss, mean_val_occloss, mean_val_occiou, mean_val_sdfloss, mean_val_depthloss, mean_val_colorloss, mean_val_discloss, mean_val_discloss_real, mean_val_discloss_fake, mean_val_genloss, mean_val_discedgeloss, mean_val_discedgeloss_real, mean_val_discedgeloss_fake, mean_val_genedgeloss, mean_val_styleloss, mean_val_contentloss, time, log) #cannydisc
    else:
        print_log_info(epoch, iter, mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, time, None) #cannydisc
        print_log_info(epoch, iter, mean_train_loss, mean_train_occloss, mean_train_occiou, mean_train_sdfloss, mean_train_depthloss, mean_train_colorloss, mean_train_discloss, mean_train_discloss_real, mean_train_discloss_fake, mean_train_genloss, mean_train_discedgeloss, mean_train_discedgeloss_real, mean_train_discedgeloss_fake, mean_train_genedgeloss, mean_train_styleloss, mean_train_contentloss, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, time, log) #cannydisc
    log.flush()

def train(epoch, iter, dataloader, log_file, output_save):
    train_losses = []
    train_lossocc = []
    train_iouocc = []
    train_losssdf = []
    train_lossdepth = []
    train_lossdisc = []
    train_lossdisc_real = []
    train_lossdisc_fake = []
    train_lossgen = []
    train_lossdiscedge = [] #cannydisc
    train_lossdiscedge_real = [] #cannydisc
    train_lossdiscedge_fake = [] #cannydisc
    train_lossgenedge = [] #cannydisc
    train_losscolor = []
    train_lossstyle = []
    train_losscontent = []
    model.train()
    disc.train()
    discedge.train() #cannydisc
    start = time.time()
    
    use_disc = args.weight_disc_loss > 0
    num_batches = len(dataloader)
    for t, sample in enumerate(dataloader):    
        sdfs = sample['sdf']
        if sdfs is None:
            #torch.cuda.empty_cache()
            print("continue 1")
            continue
        if sdfs.shape[0] < args.batch_size:
            print("sdfs.shape[0]", sdfs.shape[0], "args.batch_size", args.batch_size)
            #torch.cuda.empty_cache()
            print("continue 2")
            continue  # maintain same batch size for training
        if iter > args.num_iters_geo_only and (use_disc or args.weight_depth_loss > 0) and sample['images_color'] is None:
            print('skipping no frames')
            #torch.cuda.empty_cache()
            print("continue 3")
            continue
        inputs = sample['input']
        
        known = sample['known']
        colors = sample['colors'].cuda()
        if args.use_loss_masking:
            #known = (known <= 1).cuda()
            known = (known <= args.known).cuda() #(4, 1, 128, 64, 64)
        inputs = inputs.cuda()
        target_for_sdf, target_for_colors = loss_util.compute_targets(sdfs.cuda(), args.truncation, args.use_loss_masking, known, colors)
        pred_sdf = [iter > args.num_iters_geo_only//3 and args.weight_sdf_loss > 0, iter > args.num_iters_geo_only*2//3 and args.weight_sdf_loss > 0]
        pred_color = iter > args.num_iters_geo_only and (args.weight_color_loss > 0 or args.weight_style_loss > 0 or args.weight_content_loss > 0)
        compute_2dstyle = iter > args.num_iters_before_content and args.weight_style_loss > 0
        compute_2dcontent = iter > args.num_iters_before_content and args.weight_content_loss > 0
        
        if use_disc:
            optimizer_disc.zero_grad()
        if args.weight_discedge_loss > 0: #cannydisc
            optimizer_discedge.zero_grad() #cannydisc
        optimizer.zero_grad()
        mask = sample['mask'].cuda()
        mask_hole = sample['mask_hole'].cuda()
        output_occ = None
        output_occ, output_sdf, output_color = model(inputs, mask, mask_hole, pred_sdf=pred_sdf, pred_color=pred_color)
        output_coarse_sdf = None
        if args.weight_depth_loss == 0:
            output_sdf = [[],[]]

        loss = 0.0
        input_occ = torch.abs(inputs[:,:1]) < (args.truncation-0.01)
        #input_occ = torch.abs(inputs[:,:1]) < args.truncation #empty
        weight_occ_loss = 1 if iter <= args.num_iters_geo_only else args.weight_occ_loss
        weight = loss_util.compute_dense_geo_weights(target_for_sdf, input_occ, args.truncation, args.weight_surf_geo, args.weight_missing_geo)
        if output_occ is not None:
            empty = torch.nn.Sigmoid()(output_occ.detach()) < 0.5
            weight[empty] = 0
            loss_occ = loss_util.compute_geo_occ_loss(target_for_sdf, output_occ, known, weight, args.truncation)
            loss += weight_occ_loss * loss_occ
            train_lossocc.append(loss_occ.item())
            iou_occ = loss_util.compute_iou_occ(target_for_sdf, output_occ, known, args.truncation)
            train_iouocc.append(iou_occ)
        if pred_sdf[1] and (args.weight_sdf_loss > 0 or args.weight_depth_loss > 0):
            if args.weight_sdf_loss > 0:
                loss_sdf = loss_util.compute_geo_loss(target_for_sdf, output_coarse_sdf, output_sdf, known, weight, args.logweight_sdf)
                loss += args.weight_sdf_loss * loss_sdf
                train_losssdf.append(loss_sdf.item())
            if output_occ is not None:
                locs = torch.nonzero((torch.abs(output_sdf.detach()[:,0]) < args.truncation) & ~empty[:,0])
            else:
                locs = torch.nonzero(torch.abs(output_sdf.detach()[:,0]) < args.truncation)
            locs = torch.cat([locs[:,1:], locs[:,:1]],1)
            output_sdf = [locs, output_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
        else:
            output_sdf = None
        if pred_color:
            if output_coarse_sdf is not None:
                locs_coarse = torch.nonzero(torch.abs(output_coarse_sdf[:,0]) < args.truncation)
                locs_coarse = torch.cat([locs_coarse[:,1:], locs_coarse[:,:1]],1)
            output_color = [locs, output_color[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
        elif not pred_color:
            output_color = None
                
        missing_mask = None
        synth = None
        target = None
        input2d = None
        target2d = None
        target_depth = None
        pred_depth = None
        
        if iter > args.num_iters_geo_only and len(output_sdf[0]) > 0 and (args.weight_depth_loss > 0 or compute_2dstyle or compute_2dcontent or use_disc):
            if len(output_sdf[0]) > raycaster_rgbd.get_max_num_locs_per_sample()*args.batch_size:
                print('(iter %d) too many voxels for raycast (%d)' % (iter, len(output_sdf[0])))
                #torch.cuda.empty_cache()
                print("continue 4")
                continue
            images_color = sample['images_color'].cuda()
            #images_graycolor = sample['images_graycolor'].cuda() #cannydisc
            images_edge = sample['images_edge'].cuda() #cannydisc
            images_depth = sample['images_depth'].cuda()
            poses = sample['images_pose'].cuda()
            intrinsics = sample['images_intrinsic'].cuda()
            view_matrix = style.compute_view_matrix(sample['world2grid'].cuda(), poses)
            
            images_depth = images_depth.unsqueeze(1)
            images_normals = depth2normals(images_depth.clone(), intrinsics.clone()) #dpl
            
            if use_disc and images_normals is None:
                #torch.cuda.empty_cache()
                print("continue 5")
                continue
            output_normals = loss_util.compute_normals_sparse(output_sdf[0].cuda(), output_sdf[1], target_for_sdf.shape[2:], transform=torch.inverse(view_matrix))
            
            weight_color = None
            if args.weight_missing_color > 1:
                missing_mask_3d = loss_util.compute_missing_geo_mask(input_occ, target_for_sdf, args.truncation)
                missing_mask = raycaster_occ(missing_mask_3d.byte(), view_matrix, intrinsics).bool().clone()
                target_mask_3d = torch.abs(target_for_sdf) < 1
                target_mask_2d = raycaster_occ(target_mask_3d.byte(), view_matrix, intrinsics).bool()
                weight_color = (target_mask_2d & missing_mask).float()
                weight_color[weight_color > 0] = args.weight_missing_color
                weight_color[weight_color == 0] = 1
            
            if True:
                input_locs = torch.nonzero(torch.abs(inputs[:,0]) < args.truncation)
                input_locs = torch.cat([input_locs[:,1:], input_locs[:,:1]], 1)
                input_vals = inputs[input_locs[:,-1], :, input_locs[:,0], input_locs[:,1], input_locs[:,2]]
                input_normals = loss_util.compute_normals(inputs[:,:1], input_locs, transform=torch.inverse(view_matrix))
                raycast_color, _, raycast_normal = raycaster_rgbd(input_locs.cuda(), input_vals[:,:1].contiguous(), input_vals[:,1:].contiguous(), input_normals, view_matrix, intrinsics)
                if pred_color:
                    invalid = raycast_color == -float('inf')
                    input2d = raycast_color.clone()*2-1
                    input2d[invalid] = 0
                normals = raycast_normal.clone()
                invalid = raycast_normal == -float('inf')
                normals[invalid] = 0
                if pred_color:
                    #assert args.color_space == 'rgb' #cannydisc
                    input2d_g = input2d.clone() #labdisc
                    input2d_g = torch.cat([input2d_g, images_edge.float().unsqueeze(3)], 3) #cannydisc
                    input2d = torch.cat([input2d, normals], 3)
                else:
                    input2d = normals
                input2d = input2d.permute(0, 3, 1, 2).contiguous()
                input2d_g = input2d_g.permute(0, 3, 1, 2).contiguous() #cannydisc
            weight_sample_tgt2d = None
            weight_sample_pred2d = None
            if args.project_targets:
                locs = torch.nonzero(torch.abs(target_for_sdf[:,0]) < args.truncation)
                locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()
                vals = target_for_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
                colors = target_for_colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float()/255.0
                target_normals = loss_util.compute_normals_sparse(locs, vals, target_for_sdf.shape[2:], transform=torch.inverse(view_matrix))
                
                raycast_color, _, raycast_normal = raycaster_rgbd(locs, vals, colors.contiguous(), target_normals, view_matrix, intrinsics)
                if args.filter_proj_tgt:
                    invalid = loss_util.filter_proj_target(raycast_color, args.color_thresh, args.color_space)
                    invalid = invalid.unsqueeze(3).repeat(1,1,1,3) | (raycast_color == -float('inf'))
                else:
                    invalid = raycast_color == -float('inf')
                if args.weight_by_percent_pixels:
                    weight_sample_tgt2d = 1.0 - torch.sum(invalid[:,:,:,0].view(invalid.shape[0],-1),1).float()/float(args.style_width*args.style_height)
                    weight_sample_tgt2d = torch.clamp(weight_sample_tgt2d, 0, 0.3)/0.3
                if args.weight_color_loss > 0:
                    target2d = raycast_color.clone()
                    target2d[invalid] = images_color.permute(0,2,3,1)[invalid] # copy to image
                    target2d = target2d*2-1
                    target2d_invalid = invalid[:,:,:,0].clone() #cannydisc
                invalid = raycast_normal == -float('inf')
                normals = raycast_normal.clone()
                normals[invalid] = images_normals.permute(0,2,3,1)[invalid] # copy to image
                if pred_color:
                    target2d_g = target2d.clone() #labdisc
                    target2d_g = torch.cat([target2d_g, target2d_invalid.float().unsqueeze(3)], 3) #cannydisc
                    target2d = torch.cat([target2d, normals], 3)
                else:
                    target2d = normals
                target2d = target2d.permute(0, 3, 1, 2).contiguous()
                target2d_g = target2d_g.permute(0, 3, 1, 2).contiguous() #cannydisc
            color = None
            if pred_color:
                color = (output_color[1] + 1) * 0.5
            else:
                color = torch.zeros(output_sdf[0].shape[0], 3).cuda()
            # raycast prediction
            raycast_color, raycast_depth, raycast_normal = raycaster_rgbd(output_sdf[0].cuda(), output_sdf[1], color, output_normals, view_matrix, intrinsics)
            if args.weight_by_percent_pixels:
                weight_sample_pred2d = torch.sum((raycast_color[:,:,:,0] != -float('inf')).view(raycast_color.shape[0],-1),1).float()/float(args.style_width*args.style_height)
                weight_sample_pred2d = torch.clamp(weight_sample_pred2d, 0, 0.3)/0.3
            # geo loss
            raycast_depth = raycast_depth.unsqueeze(1)*args.voxelsize
            valid = (raycast_depth != -float('inf')) & (images_depth != 0)
            
            loss_depth = torch.mean(torch.abs(raycast_depth[valid] - images_depth[valid]))
            loss += args.weight_depth_loss * loss_depth
            pred_depth = raycast_depth.detach()
            target_depth = images_depth
            train_lossdepth.append(loss_depth.item())
            if args.weight_color_loss > 0: # color loss
                loss_color = loss_util.compute_2dcolor_loss(raycast_color, images_color.permute(0, 2, 3, 1), weight_color)
                loss += args.weight_color_loss * loss_color
                train_losscolor.append(loss_color.item())
                synth = raycast_color.detach().permute(0, 3, 1, 2)
                target = images_color
            if pred_color:
                raycast_g = raycast_color.clone() #labdisc
                raycast = torch.cat([raycast_color, raycast_normal],3)
            else:
                raycast = raycast_normal
            raycast = raycast.permute(0, 3, 1, 2).contiguous()
            raycast_g = raycast_g.permute(0, 3, 1, 2).contiguous() #cannydisc
            valid = raycast.detach() != -float('inf')
            num_valid = torch.sum(valid).item()
            if use_disc and args.patch_disc and args.patch_size < args.style_height:
                if args.parallel:
                    valid = (disc.module.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > args.valid_thresh).squeeze(1) #dpl
                else:
                    valid = (disc.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > args.valid_thresh).squeeze(1) #dpl
                
                weight_color_disc = None 
                if weight_color is not None:
                    if args.parallel:
                        weight_color_disc = disc.module.compute_valids(weight_color) #dpl
                    else:
                        weight_color_disc = disc.compute_valids(weight_color) #dpl
                    
                    weight_color_disc = args.weight_missing_color * weight_color_disc / torch.max(weight_color_disc).item()
            if num_valid > 2000 and (not args.patch_disc or torch.sum(valid).item() > 0):
                if compute_2dstyle or compute_2dcontent:
                    if not use_disc:
                        synth = raycast_color.detach().permute(0,3,1,2)
                        target = images_color.clone()
                    style_pred_color = raycast_color #[0,1]
                    style_tgt_color = images_color.permute(0,2,3,1).contiguous()
                    if args.color_space == 'lab': # convert to rgb
                        invalid = style_pred_color == -float('inf')
                        style_pred_color = data_util.convert_lab01_to_rgb_pt(style_pred_color)
                        style_pred_color[invalid] = -float('inf')
                        style_pred_color = style_pred_color.permute(0,3,1,2).contiguous()
                        style_tgt_color = data_util.convert_lab01_to_rgb_pt(style_tgt_color).permute(0,3,1,2).contiguous()
                    else:
                        style_pred_color = style_pred_color.permute(0,3,1,2).contiguous()
                        style_tgt_color = style_tgt_color.permute(0,3,1,2).contiguous()
                    #loss_style, loss_content = style.compute_style_loss(style_pred_color, style_tgt_color, model_style, compute_2dstyle, compute_2dcontent, missing_mask) #CTSDG
                    loss_style, loss_content = style.compute_style_loss(style_pred_color, style_tgt_color, model_style, compute_2dstyle, compute_2dcontent, None)
                    loss = loss + args.weight_style_loss * loss_style + args.weight_content_loss * loss_content
                    if compute_2dstyle:
                        train_lossstyle.append(loss_style.item())
                    if compute_2dcontent:
                        train_losscontent.append(loss_content.item())
                
                if use_disc: # train disc
                    if pred_color:
                        synth_invalid_3 = raycast_g == -float('inf') #cannydisc
                        synth_invalid = synth_invalid_3[:,0,:,:].clone() #cannydisc
                        synth, target = style.preprocess_rendered_target_images(raycast, torch.cat([images_color, images_normals],1))
                        synth[:,:3] = synth[:,:3] * 2 - 1 # normalize
                        synth_g, target_g = style.preprocess_rendered_target_images(raycast_g, images_color) #cannydisc #labdisc
                        synth_g[:,:3] = synth_g[:,:3] * 2 - 1 # normalize #cannydisc #labdisc
                        synth_g = torch.cat([synth_g, synth_invalid.float().unsqueeze(1)], dim=1) #cannydisc
                    else:
                        synth, target = style.preprocess_rendered_target_images(raycast, images_normals)
                    if args.project_targets:
                        target = target2d
                        target_g = target2d_g #cannydisc
                    elif pred_color:
                        target[:,:3] = target[:,:3] * 2 - 1 # normalize
                        target_g[:,:3] = target_g[:,:3] * 2 - 1 # normalize #cannydisc #labdisc

                    synth = torch.cat([input2d, synth], dim=1)
                    target = torch.cat([input2d, target], dim=1)
                    synth_g = torch.cat([input2d_g, synth_g], dim=1) #cannydisc
                    target_g = torch.cat([input2d_g, target_g], dim=1) #cannydisc
                    real_loss, fake_loss, penalty = gan_loss.compute_discriminator_loss(disc, target, synth.detach(), None if not args.patch_disc else valid, None if not args.patch_disc else weight_color_disc)
                    realedge_loss, fakeedge_loss, _ = gan_loss.compute_discriminator_loss(discedge, target_g, synth_g.detach(), None if not args.patch_disc else valid, None if not args.patch_disc else weight_color_disc) #cannydisc
                    if args.weight_by_percent_pixels:
                        if args.disc_loss_type != 'hinge':
                            repeats = torch.sum(valid.view(valid.shape[0],-1), 1).cpu().numpy()
                            if weight_sample_tgt2d is not None:
                                weight_sample_tgt2d = torch.from_numpy(np.repeat(weight_sample_tgt2d.cpu().numpy(), repeats)).cuda()
                            weight_sample_pred2d = torch.from_numpy(np.repeat(weight_sample_pred2d.cpu().numpy(), repeats)).cuda()
                        if weight_sample_tgt2d is not None:
                            real_loss = real_loss * weight_sample_tgt2d
                        fake_loss = fake_loss * weight_sample_pred2d
                    real_loss = torch.mean(real_loss)
                    fake_loss = torch.mean(fake_loss)
                    realedge_loss = torch.mean(realedge_loss) #cannydisc
                    fakeedge_loss = torch.mean(fakeedge_loss) #cannydisc
                    disc_loss = 0
                    discedge_loss = 0 #cannydisc
                    if args.disc_loss_type.startswith('wgan'):
                        disc_loss = args.weight_disc_loss * 0.005 * (real_loss + fake_loss) + 10 * penalty
                    else:
                        disc_loss = args.weight_disc_loss * (real_loss + fake_loss)
                        discedge_loss = args.weight_discedge_loss * (realedge_loss + fakeedge_loss) #cannydisc
                    train_lossdisc.append(disc_loss.item())
                    train_lossdisc_real.append(real_loss.item())
                    train_lossdisc_fake.append(fake_loss.item())
                    train_lossdiscedge.append(discedge_loss.item()) #cannydisc
                    train_lossdiscedge_real.append(realedge_loss.item()) #cannydisc
                    train_lossdiscedge_fake.append(fakeedge_loss.item()) #cannydisc
                    
                    disc_loss.backward()
                    optimizer_disc.step()
                    if args.weight_discedge_loss > 0: #cannydisc
                        discedge_loss.backward() #cannydisc
                        optimizer_discedge.step() #cannydisc
                    
                    # train gen
                    gen_loss = gan_loss.compute_generator_loss(disc, synth)
                    loss += args.weight_discgen_loss * gen_loss
                    train_lossgen.append(gen_loss.item())
                    genedge_loss = gan_loss.compute_generator_loss(discedge, synth_g) #cannydisc
                    loss += args.weight_discgenedge_loss * genedge_loss #cannydisc
                    train_lossgenedge.append(genedge_loss.item()) #cannydisc
            else:
                synth = None
                target = None
        loss.backward()
        optimizer.step()
        
        if output_color is not None:
            output_color = (output_color[1] + 1) * 0.5
        if use_disc and synth is not None:
            synth = synth.detach()
            synth = synth[:,synth.shape[1]//2:]
            target = target[:,target.shape[1]//2:]
            if output_color is not None:
                synth[:,:3] = (synth[:,:3] + 1) * 0.5
                target[:,:3] = (target[:,:3] + 1) * 0.5
        
        output_visual = output_save and (t + 2 == num_batches or (iter > 0 and iter % 10000 == 0))
        
        train_losses.append(loss.item())
        iter += 1
        if iter % 20 == 0:
            took = time.time() - start
            print_log(log_file, epoch, iter, train_losses, train_lossocc, train_iouocc, train_losssdf, train_lossdepth, train_losscolor, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, train_lossdiscedge_fake, train_lossgenedge, train_lossstyle, train_losscontent, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, took) #cannydisc
        if iter % 10000 == 0:
            if args.parallel:
                torch.save({'epoch': epoch,'state_dict': model.module.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch))) #dpl
                if use_disc:
                    torch.save({'epoch': epoch,'state_dict': disc.module.state_dict(),'optimizer' : optimizer_disc.state_dict()}, os.path.join(args.save, 'disc-iter%s-epoch%s.pth' % (iter, epoch))) #dpl
                if args.weight_discedge_loss >0: #cannydisc
                    torch.save({'epoch': epoch,'state_dict': discedge.module.state_dict(),'optimizer' : optimizer_discedge.state_dict()}, os.path.join(args.save, 'discedge-iter%s-epoch%s.pth' % (iter, epoch)))#cannydisc #dpl
            else:
                torch.save({'epoch': epoch,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch))) #dpl            
                if use_disc:
                    torch.save({'epoch': epoch,'state_dict': disc.state_dict(),'optimizer' : optimizer_disc.state_dict()}, os.path.join(args.save, 'disc-iter%s-epoch%s.pth' % (iter, epoch))) #dpl
                if args.weight_discedge_loss >0: #cannydisc
                    torch.save({'epoch': epoch,'state_dict': discedge.state_dict(),'optimizer' : optimizer_discedge.state_dict()}, os.path.join(args.save, 'discedge-iter%s-epoch%s.pth' % (iter, epoch)))#cannydisc #dpl
                
        if iter % 100 == 0:
            gc.collect()
        #print("train output_visual", output_visual)
#         if output_visual:
#             vis_pred_sdf = [None] * args.batch_size
#             vis_pred_color = [None] * args.batch_size
#             if output_color is not None: # convert colors to vec3uc
#                 output_color = torch.clamp(output_color.detach() * 255, 0, 255)
#             if output_sdf is not None and len(output_sdf[0]) > 0:
#                 for b in range(args.batch_size):
#                     mask = output_sdf[0][:,-1] == b
#                     if len(mask) > 0:
#                         vis_pred_sdf[b] = [output_sdf[0].detach()[mask].cpu().numpy(), output_sdf[1].detach()[mask].squeeze().cpu().numpy()]
#                     if output_color is not None:
#                         vis_pred_color[b] = output_color[mask].cpu().numpy()
#             inputs = inputs.cpu().numpy()
#             target_for_colors = target_for_colors.cpu().numpy()
#             vis_pred_images_color = None
#             vis_tgt_images_color = None
#             vis_input_images_color = None
#             vis_pred_depth = None
#             vis_target_depth = None
#             if synth is not None:
#                 vis_pred_images_color = synth.detach().cpu().numpy()
#                 vis_pred_images_color = np.transpose(vis_pred_images_color, [0,2,3,1])
#                 vis_tgt_images_color = target.cpu().numpy()
#                 vis_tgt_images_color = np.transpose(vis_tgt_images_color, [0,2,3,1])
#             if input2d is not None:
#                 vis_input_images_color = input2d.cpu().numpy()
#                 if pred_color:
#                     vis_input_images_color[:,:3] = (vis_input_images_color[:,:3] + 1) * 0.5
#                 vis_input_images_color = np.transpose(vis_input_images_color, [0,2,3,1])
#             if pred_depth is not None:            
#                 vis_pred_depth = pred_depth.cpu().numpy()[:,0,:,:]
#                 vis_pred_depth = data_util.vis_depth_as_hsv(vis_pred_depth, raycast_depth_max)
#                 vis_target_depth = target_depth.cpu().numpy()[:,0,:,:]
#                 vis_target_depth = data_util.vis_depth_as_hsv(vis_target_depth, raycast_depth_max)
#             pred_occ = None
#             if output_occ is not None:
#                 if isinstance(output_occ, tuple):
#                     pred_occ = None # not supported
#                 else:
#                     pred_occ = (torch.nn.Sigmoid()(output_occ) > 0.5).cpu().numpy().astype(np.float32)
#             print("no train saving")
            #data_util.save_predictions(os.path.join(args.save, 'iter%d-epoch%d' % (iter, epoch), 'train'), np.arange(sdfs.shape[0]), sample['name'], inputs, target_for_sdf.cpu().numpy(), target_for_colors, None, vis_tgt_images_color, vis_pred_sdf, vis_pred_color, None, vis_pred_images_color, sample['world2grid'].numpy(), args.truncation, args.color_space, input_images=vis_input_images_color, pred_depth=vis_pred_depth, target_depth=vis_target_depth, pred_occ=pred_occ)
    print("no train saving")

    return train_losses, train_lossocc, train_iouocc, train_losssdf, train_lossdepth, train_losscolor, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, train_lossdiscedge_fake, train_lossgenedge, train_lossstyle, train_losscontent, iter #cannydisc


def test(epoch, iter, dataloader, log_file, output_save):
    val_losses = []
    val_lossocc = []
    val_iouocc = []
    val_losssdf = []
    val_lossdepth = []
    val_losscolor = []
    val_lossdisc = []
    val_lossdisc_real = []
    val_lossdisc_fake = []
    val_lossgen = []
    val_lossdiscedge = [] #cannydisc
    val_lossdiscedge_real = [] #cannydisc
    val_lossdiscedge_fake = [] #cannydisc
    val_lossgenedge = [] #cannydisc
    val_lossstyle = []
    val_losscontent = []
    model.eval()
    disc.eval()
    discedge.eval() #cannydisc

    use_disc = args.weight_disc_loss > 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            sdfs = sample['sdf']
            if sdfs is None:
                #torch.cuda.empty_cache()
                print("continue 6")
                continue
            if sdfs.shape[0] < args.batch_size:
                #torch.cuda.empty_cache()
                print("continue 7")
                continue  # maintain same batch size
            if iter > args.num_iters_geo_only and (use_disc or args.weight_depth_loss > 0) and sample['images_color'] is None:
                print('skipping no frames')
                #torch.cuda.empty_cache()
                print("continue 8")
                continue
            inputs = sample['input'].cuda() #myyyyy
            known = sample['known']
            colors = sample['colors'].cuda()
            if args.use_loss_masking:
                #known = (known <= 1).cuda() #myyyyy
                known = (known <= args.known).cuda() #myyyyy
            #inputs[0] = inputs[0].cuda()
            #inputs[1] = inputs[1].cuda()
            
            target_for_sdf, target_for_colors = loss_util.compute_targets(sdfs.cuda(), args.truncation, args.use_loss_masking, known, colors)
            pred_sdf = [iter > args.num_iters_geo_only//3 and args.weight_sdf_loss > 0, iter > args.num_iters_geo_only*2//3 and args.weight_sdf_loss > 0]
            pred_color = iter > args.num_iters_geo_only and (args.weight_color_loss > 0 or args.weight_style_loss > 0 or args.weight_content_loss > 0)
            compute_2dstyle = iter > args.num_iters_before_content and args.weight_style_loss > 0
            compute_2dcontent = iter > args.num_iters_before_content and args.weight_content_loss > 0

            mask = sample['mask'].cuda()
            mask_hole = sample['mask_hole'].cuda() #gated
            output_occ = None
            output_occ, output_sdf, output_color = model(inputs, mask, mask_hole, pred_sdf=pred_sdf, pred_color=pred_color) #gated
            output_coarse_sdf = None
            if args.weight_depth_loss == 0:
                output_sdf = [[],[]]
            
            loss = 0.0
            input_occ = torch.abs(inputs[:,:1]) < (args.truncation-0.01)
            #input_occ = torch.abs(inputs[:,:1]) < args.truncation #empty
            #input_occ = loss_util.compute_dense_occ(inputs[0], inputs[1][:,0], target_for_sdf.shape, args.truncation)
            weight_occ_loss = 1 if iter <= args.num_iters_geo_only else args.weight_occ_loss #myyyyy
            weight = loss_util.compute_dense_geo_weights(target_for_sdf, input_occ, args.truncation, args.weight_surf_geo, args.weight_missing_geo) #myyyyy
            if output_occ is not None:
                empty = torch.nn.Sigmoid()(output_occ.detach()) < 0.5
                weight[empty] = 0
                loss_occ = loss_util.compute_geo_occ_loss(target_for_sdf, output_occ, known, weight, args.truncation)
                loss += weight_occ_loss * loss_occ
                val_lossocc.append(loss_occ.item())
                iou_occ = loss_util.compute_iou_occ(target_for_sdf, output_occ, known, args.truncation)
                val_iouocc.append(iou_occ)       
            if pred_sdf[1] and (args.weight_sdf_loss > 0 or args.weight_depth_loss > 0):
                if args.weight_sdf_loss > 0:
                    loss_sdf = loss_util.compute_geo_loss(target_for_sdf, output_coarse_sdf, output_sdf, known, weight, args.logweight_sdf)
                    loss += args.weight_sdf_loss * loss_sdf
                    val_losssdf.append(loss_sdf.item())
                if output_occ is not None:
                    locs = torch.nonzero((torch.abs(output_sdf.detach()[:,0]) < args.truncation) & ~empty[:,0])
                else:
                    locs = torch.nonzero(torch.abs(output_sdf.detach()[:,0]) < args.truncation)
                locs = torch.cat([locs[:,1:], locs[:,:1]],1)
                output_sdf = [locs, output_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
            else:
                output_sdf = None
            if pred_color:
                if output_coarse_sdf is not None:
                    locs_coarse = torch.nonzero(torch.abs(output_coarse_sdf[:,0]) < args.truncation)
                    locs_coarse = torch.cat([locs_coarse[:,1:], locs_coarse[:,:1]],1)
                output_color = [locs, output_color[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
            elif not pred_color:
                output_color = None

            missing_mask = None
            synth = None
            target = None
            input2d = None
            target2d = None #myyy
            target_depth = None
            pred_depth = None
            
            if iter > args.num_iters_geo_only and len(output_sdf[0]) > 0 and (args.weight_depth_loss > 0 or compute_2dstyle or compute_2dcontent or use_disc):
                if len(output_sdf[0]) > raycaster_rgbd.get_max_num_locs_per_sample()*args.batch_size:
                    print('too many voxels for raycast')
                    #torch.cuda.empty_cache()
                    print("continue 9")
                    continue
                images_color = sample['images_color'].cuda()
                #images_graycolor = sample['images_graycolor'].cuda() #cannydisc
                images_edge = sample['images_edge'].cuda() #cannydisc
                images_depth = sample['images_depth'].cuda()
                poses = sample['images_pose'].cuda()
                intrinsics = sample['images_intrinsic'].cuda()
                view_matrix = style.compute_view_matrix(sample['world2grid'].cuda(), poses)
            
                images_depth = images_depth.unsqueeze(1)
                images_normals = depth2normals(images_depth.clone(), intrinsics.clone()) #dpl
                if use_disc and images_normals is None:
                    #torch.cuda.empty_cache()
                    print("continue 10")
                    continue
                output_normals = loss_util.compute_normals_sparse(output_sdf[0].cuda(), output_sdf[1], target_for_sdf.shape[2:], transform=torch.inverse(view_matrix))
            
                weight_color = None
                if args.weight_missing_color > 1:
                    #missing_mask_3d = loss_util.compute_missing_geo_mask(inputs[0], target_for_sdf, args.truncation)
                    #missing_mask = raycaster_occ(missing_mask_3d.byte(), view_matrix, intrinsics).bool().clone()
                    #target_mask_3d = torch.abs(target_for_sdf) < 1
                    #target_mask_2d = raycaster_occ(target_mask_3d.byte(), view_matrix, intrinsics).bool()
                    missing_mask_3d = loss_util.compute_missing_geo_mask(input_occ, target_for_sdf, args.truncation) #myyy     
                    missing_mask = raycaster_occ(missing_mask_3d.byte(), view_matrix, intrinsics).bool().clone() #myyy     
                    target_mask_3d = torch.abs(target_for_sdf) < 1 #myyy     
                    target_mask_2d = raycaster_occ(target_mask_3d.byte(), view_matrix, intrinsics).bool() #myyy                
                    weight_color = (target_mask_2d & missing_mask).float()
                    weight_color[weight_color > 0] = args.weight_missing_color
                    weight_color[weight_color == 0] = 1
                
                #target2d = None
                if True:
                    input_locs = torch.nonzero(torch.abs(inputs[:,0]) < args.truncation)
                    input_locs = torch.cat([input_locs[:,1:], input_locs[:,:1]], 1) #myyy     
                    input_vals = inputs[input_locs[:,-1], :, input_locs[:,0], input_locs[:,1], input_locs[:,2]] #myyy     
                    input_normals = loss_util.compute_normals(inputs[:,:1], input_locs, transform=torch.inverse(view_matrix)) #myyy     
                    raycast_color, _, raycast_normal = raycaster_rgbd(input_locs.cuda(), input_vals[:,:1].contiguous(), input_vals[:,1:].contiguous(), input_normals, view_matrix, intrinsics) #myyy     
                    #input_normals = loss_util.compute_normals_sparse(inputs[0].cuda(), inputs[1][:,:1], target_for_sdf.shape[2:], transform=torch.inverse(view_matrix))
                    #raycast_color, _, raycast_normal = raycaster_rgbd(inputs[0].cuda(), inputs[1][:,:1].contiguous(), inputs[1][:,1:].contiguous(), input_normals, view_matrix, intrinsics)
                    if pred_color:
                        invalid = raycast_color == -float('inf')
                        input2d = raycast_color.clone()*2-1
                        input2d[invalid] = 0
                    normals = raycast_normal.clone()
                    invalid = raycast_normal == -float('inf')
                    normals[invalid] = 0
                    if pred_color:
                        #assert args.color_space == 'rgb' #cannydisc #labdisc
                        input2d_g = input2d.clone() #labdisc
                        input2d_g = torch.cat([input2d_g, images_edge.float().unsqueeze(3)], 3) #cannydisc
                        input2d = torch.cat([input2d, normals], 3)
                    else:
                        input2d = normals
                    input2d = input2d.permute(0, 3, 1, 2).contiguous()
                    input2d_g = input2d_g.permute(0, 3, 1, 2).contiguous() #cannydisc
                weight_sample_tgt2d = None
                weight_sample_pred2d = None
                if args.project_targets:
                    locs = torch.nonzero(torch.abs(target_for_sdf[:,0]) < args.truncation)
                    locs = torch.cat([locs[:,1:], locs[:,:1]],1).contiguous()
                    vals = target_for_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].contiguous()
                    colors = target_for_colors[locs[:,-1],locs[:,0],locs[:,1],locs[:,2],:].float()/255.0
                    target_normals = loss_util.compute_normals_sparse(locs, vals, target_for_sdf.shape[2:], transform=torch.inverse(view_matrix))
                    
                    raycast_color, _, raycast_normal = raycaster_rgbd(locs, vals, colors.contiguous(), target_normals, view_matrix, intrinsics)
                    if args.filter_proj_tgt:
                        invalid = loss_util.filter_proj_target(raycast_color, args.color_thresh, args.color_space)
                        invalid = invalid.unsqueeze(3).repeat(1,1,1,3) | (raycast_color == -float('inf'))
                    else:
                        invalid = raycast_color == -float('inf')
                    if args.weight_by_percent_pixels:
                        weight_sample_tgt2d = 1.0 - torch.sum(invalid[:,:,:,0].view(invalid.shape[0],-1),1).float()/float(args.style_width*args.style_height)
                        weight_sample_tgt2d = torch.clamp(weight_sample_tgt2d, 0, 0.3)/0.3
                    if args.weight_color_loss > 0:
                        target2d = raycast_color.clone()
                        target2d[invalid] = images_color.permute(0,2,3,1)[invalid] # copy to image
                        target2d = target2d*2-1
                        target2d_invalid = invalid[:,:,:,0].clone() #cannydisc
                    invalid = raycast_normal == -float('inf')
                    normals = raycast_normal.clone()
                    normals[invalid] = images_normals.permute(0,2,3,1)[invalid] # copy to image
                    if pred_color:
                        target2d_g = target2d.clone() #labdisc
                        target2d_g = torch.cat([target2d_g, target2d_invalid.float().unsqueeze(3)], 3) #cannydisc
                        target2d = torch.cat([target2d, normals], 3)
                    else:
                        target2d = normals
                    target2d = target2d.permute(0, 3, 1, 2).contiguous()
                    target2d_g = target2d_g.permute(0, 3, 1, 2).contiguous() #cannydisc
                color = None
                if pred_color:
                    color = (output_color[1] + 1) * 0.5
                else:
                    color = torch.zeros(output_sdf[0].shape[0], 3).cuda()
                # raycast prediction
                raycast_color, raycast_depth, raycast_normal = raycaster_rgbd(output_sdf[0].cuda(), output_sdf[1], color, output_normals, view_matrix, intrinsics)
                if args.weight_by_percent_pixels:
                    weight_sample_pred2d = torch.sum((raycast_color[:,:,:,0] != -float('inf')).view(raycast_color.shape[0],-1),1).float()/float(args.style_width*args.style_height)
                    weight_sample_pred2d = torch.clamp(weight_sample_pred2d, 0, 0.3)/0.3
                # geo loss
                raycast_depth = raycast_depth.unsqueeze(1)*args.voxelsize            
                valid = (raycast_depth != -float('inf')) & (images_depth != 0)
                loss_depth = torch.mean(torch.abs(raycast_depth[valid] - images_depth[valid]))
                loss += args.weight_depth_loss * loss_depth
                pred_depth = raycast_depth.detach()
                target_depth = images_depth            
                val_lossdepth.append(loss_depth.item())
                if args.weight_color_loss > 0: # color loss
                    loss_color = loss_util.compute_2dcolor_loss(raycast_color, images_color.permute(0, 2, 3, 1), weight_color)
                    loss += args.weight_color_loss * loss_color
                    val_losscolor.append(loss_color.item()) #myyy
                    synth = raycast_color.detach().permute(0, 3, 1, 2) #myyy
                    target = images_color
                if pred_color:
                    raycast_g = raycast_color.clone() #labdisc
                    raycast = torch.cat([raycast_color, raycast_normal],3)
                else:
                    raycast = raycast_normal
                raycast = raycast.permute(0, 3, 1, 2).contiguous()
                raycast_g = raycast_g.permute(0, 3, 1, 2).contiguous() #cannydisc
                valid = raycast.detach() != -float('inf')
                num_valid = torch.sum(valid).item()
                if use_disc and args.patch_disc and args.patch_size < args.style_height:
                    if args.parallel:
                        valid = (disc.module.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > args.valid_thresh).squeeze(1) #dpl
                    else:
                        valid = (disc.compute_valids(valid[:,-1,:,:].float().unsqueeze(1)) > args.valid_thresh).squeeze(1) #dpl
                    
                    weight_color_disc = None
                    if weight_color is not None:
                        if args.parallel:
                            weight_color_disc = disc.module.compute_valids(weight_color) #dpl
                        else:
                            weight_color_disc = disc.compute_valids(weight_color) #dpl
                        
                        weight_color_disc = args.weight_missing_color * weight_color_disc / torch.max(weight_color_disc).item()
                if num_valid > 2000 and (not args.patch_disc or torch.sum(valid).item() > 0):
                    if compute_2dstyle or compute_2dcontent:
                        if not use_disc:
                            synth = raycast_color.detach().permute(0,3,1,2)
                            target = images_color.clone()
                        style_pred_color = raycast_color #[0,1]
                        style_tgt_color = images_color.permute(0,2,3,1).contiguous()                    
                        if args.color_space == 'lab': # convert to rgb
                            invalid = style_pred_color == -float('inf')
                            style_pred_color = data_util.convert_lab01_to_rgb_pt(style_pred_color)
                            style_pred_color[invalid] = -float('inf')
                            style_pred_color = style_pred_color.permute(0,3,1,2).contiguous()
                            style_tgt_color = data_util.convert_lab01_to_rgb_pt(style_tgt_color).permute(0,3,1,2).contiguous()
                        else:
                            style_pred_color = style_pred_color.permute(0,3,1,2).contiguous()
                            style_tgt_color = style_tgt_color.permute(0,3,1,2).contiguous()
                        #loss_style, loss_content = style.compute_style_loss(style_pred_color, style_tgt_color, model_style, compute_2dstyle, compute_2dcontent, missing_mask) #CTSDG
                        loss_style, loss_content = style.compute_style_loss(style_pred_color, style_tgt_color, model_style, compute_2dstyle, compute_2dcontent, None)
                        loss = loss + args.weight_style_loss * loss_style + args.weight_content_loss * loss_content
                        if compute_2dstyle:
                            val_lossstyle.append(loss_style.item())
                        if compute_2dcontent:
                            val_losscontent.append(loss_content.item())
                        
                    if use_disc:
                        # val disc
                        if pred_color:
                            synth, target = style.preprocess_rendered_target_images(raycast, torch.cat([images_color, images_normals],1))
                            synth[:,:3] = synth[:,:3] * 2 - 1 # normalize
                            synth_g, target_g = style.preprocess_rendered_target_images(raycast_g, images_color) #cannydisc #labdisc
                            synth_g[:,:3] = synth_g[:,:3] * 2 - 1 # normalize #cannydisc #labdisc
                            synth_invalid_3 = raycast == -float('inf') #cannydisc
                            synth_invalid = synth_invalid_3[:,0,:,:].clone() #cannydisc
                            synth_g = torch.cat([synth_g, synth_invalid.float().unsqueeze(1)], dim=1) #cannydisc
                        else:
                            synth, target = style.preprocess_rendered_target_images(raycast, images_normals)
                        if args.project_targets:
                            target = target2d
                            target_g = target2d_g #cannydisc
                        elif pred_color:
                            target[:,:3] = target[:,:3] * 2 - 1 # normalize
                            target_g[:,:3] = target_g[:,:3] * 2 - 1 # normalize #cannydisc #labdisc
                        synth = torch.cat([input2d, synth], dim=1)
                        target = torch.cat([input2d, target], dim=1)
                        synth_g = torch.cat([input2d_g, synth_g], dim=1) #cannydisc
                        target_g = torch.cat([input2d_g, target_g], dim=1) #cannydisc
                        real_loss, fake_loss, penalty = gan_loss.compute_discriminator_loss(disc, target, synth.detach(), None if not args.patch_disc else valid, None if not args.patch_disc else weight_color_disc, val_mode=True)
                        realedge_loss, fakeedge_loss, _ = gan_loss.compute_discriminator_loss(discedge, target_g, synth_g.detach(), None if not args.patch_disc else valid, None if not args.patch_disc else weight_color_disc) #cannydisc

                        if args.weight_by_percent_pixels:
                            if args.disc_loss_type != 'hinge':
                                repeats = torch.sum(valid.view(valid.shape[0],-1), 1).cpu().numpy()
                                if weight_sample_tgt2d is not None:
                                    weight_sample_tgt2d = torch.from_numpy(np.repeat(weight_sample_tgt2d.cpu().numpy(), repeats)).cuda()
                                weight_sample_pred2d = torch.from_numpy(np.repeat(weight_sample_pred2d.cpu().numpy(), repeats)).cuda()
                            if weight_sample_tgt2d is not None:
                                real_loss = real_loss * weight_sample_tgt2d
                            fake_loss = fake_loss * weight_sample_pred2d
                        real_loss = torch.mean(real_loss)
                        fake_loss = torch.mean(fake_loss)
                        realedge_loss = torch.mean(realedge_loss) #cannydisc
                        fakeedge_loss = torch.mean(fakeedge_loss) #cannydisc
                        disc_loss = 0
                        discedge_loss = 0 #cannydisc
                        if args.disc_loss_type.startswith('wgan'):
                            disc_loss = args.weight_disc_loss * 0.005 * (real_loss + fake_loss) + 10 * penalty
                        else:
                            disc_loss = args.weight_disc_loss * (real_loss + fake_loss)
                            discedge_loss = args.weight_discedge_loss * (realedge_loss + fakeedge_loss) #cannydisc
                        val_lossdisc.append(disc_loss.item())
                        val_lossdisc_real.append(real_loss.item())
                        val_lossdisc_fake.append(fake_loss.item())
                        val_lossdiscedge.append(discedge_loss.item()) #cannydisc
                        val_lossdiscedge_real.append(realedge_loss.item()) #cannydisc
                        val_lossdiscedge_fake.append(fakeedge_loss.item()) #cannydisc
                        # val gen
                        gen_loss = gan_loss.compute_generator_loss(disc, synth)
                        loss += args.weight_discgen_loss * gen_loss
                        val_lossgen.append(gen_loss.item())
                        genedge_loss = gan_loss.compute_generator_loss(discedge, synth_g) #cannydisc
                        loss += args.weight_discgenedge_loss * genedge_loss #cannydisc
                        val_lossgenedge.append(genedge_loss.item()) #cannydisc
                else:
                    synth = None
                    target = None
            
#             if output_color is not None:
#                 output_color = (output_color[1] + 1) * 0.5
#             if use_disc and synth is not None:
#                 synth = synth.detach()
#                 synth = synth[:,synth.shape[1]//2:]
#                 target = target[:,target.shape[1]//2:]
#                 if output_color is not None:
#                     synth[:,:3] = (synth[:,:3] + 1) * 0.5
#                     target[:,:3] = (target[:,:3] + 1) * 0.5

#             output_visual = output_save and t + 2 == num_batches
            #output_visual = output_save and t <= 100 # vis_dataset
            val_losses.append(loss.item())
            if (t+1) % 100 == 0:
                gc.collect()
            #print("test output_visual", output_visual)
#             if output_visual:
#                 vis_pred_sdf = [None] * args.batch_size
#                 vis_pred_color = [None] * args.batch_size
#                 if output_color is not None: # convert colors to vec3uc
#                     output_color = torch.clamp(output_color.detach() * 255, 0, 255)
#                 if output_sdf is not None and len(output_sdf[0]) > 0:
#                     for b in range(args.batch_size):
#                         mask = output_sdf[0][:,-1] == b
#                         if len(mask) > 0:
#                             vis_pred_sdf[b] = [output_sdf[0][mask].detach().cpu().numpy(), output_sdf[1].detach()[mask].squeeze().cpu().numpy()]
#                         if output_color is not None:
#                             vis_pred_color[b] = output_color[mask].cpu().numpy()
#                 #inputs = [inputs[0].cpu().numpy(), inputs[1].cpu().numpy()]
#                 inputs = inputs.cpu().numpy()
#                 target_for_colors = target_for_colors.cpu().numpy()
#                 vis_pred_images_color = None
#                 vis_tgt_images_color = None
#                 vis_pred_depth = None
#                 vis_target_depth = None
#                 if synth is not None:
#                     vis_pred_images_color = synth.detach().cpu().numpy()
#                     vis_pred_images_color = np.transpose(vis_pred_images_color, [0,2,3,1])
#                     vis_tgt_images_color = target.cpu().numpy()
#                     vis_tgt_images_color = np.transpose(vis_tgt_images_color, [0,2,3,1])
#                 if pred_depth is not None:            
#                     vis_pred_depth = pred_depth.cpu().numpy()[:,0,:,:]
#                     vis_pred_depth = data_util.vis_depth_as_hsv(vis_pred_depth, raycast_depth_max)
#                     vis_target_depth = target_depth.cpu().numpy()[:,0,:,:]
#                     vis_target_depth = data_util.vis_depth_as_hsv(vis_target_depth, raycast_depth_max)
#                 pred_occ = None
#                 if output_occ is not None:
#                     if isinstance(output_occ, tuple):
#                         pred_occ = None # not supported
#                     else:
#                         pred_occ = (torch.nn.Sigmoid()(output_occ) > 0.5).cpu().numpy().astype(np.float32)
#                 print("no test saving")
                #data_util.save_predictions(os.path.join(args.save, 'iter%d-epoch%d' % (iter, epoch), 'val'), np.arange(sdfs.shape[0]), sample['name'], inputs, target_for_sdf.cpu().numpy(), target_for_colors, None, vis_tgt_images_color, vis_pred_sdf, vis_pred_color, None, vis_pred_images_color, sample['world2grid'], args.truncation, args.color_space, pred_depth=vis_pred_depth, target_depth=vis_target_depth, pred_occ=pred_occ)
    print("no test saving")
    return val_losses, val_lossocc, val_iouocc, val_losssdf, val_lossdepth, val_losscolor, val_lossdisc, val_lossdisc_real, val_lossdisc_fake, val_lossgen, val_lossdiscedge, val_lossdiscedge_real, val_lossdiscedge_fake, val_lossgenedge, val_lossstyle, val_losscontent #cannydisc

def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        raw_input('warning: save dir %s exists, press key to overwrite and continue' % args.save) #python2
        #input('warning: save dir %s exists, press key to overwrite and continue' % args.save) #python3
    data_util.dump_args_txt(args, os.path.join(args.save, 'args.txt'))
    log_file = open(os.path.join(args.save, 'log.csv'), 'w')
    has_val = len(val_files) > 0
    log_file_val = None
    if has_val:
        log_file_val = open(os.path.join(args.save, 'log_val.csv'), 'w')
    write_header(log_file, log_file_val)
    
    # start training
    iter = args.start_iter if args.start_iter > 0 else args.start_epoch * (len(train_dataset) // args.batch_size)
    print('starting training (iter %d)...' % iter)
    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()

        train_losses, train_lossocc, train_iouocc, train_losssdf, train_lossdepth, train_losscolor, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, train_lossdiscedge_fake, train_lossgenedge, train_lossstyle, train_losscontent, iter = train(epoch, iter, train_dataloader, log_file, output_save=(epoch % args.save_epoch == 0)) #cannydisc
        if has_val:
            val_losses, val_lossocc, val_iouocc, val_losssdf, val_lossdepth, val_losscolor, val_lossdisc, val_lossdisc_real, val_lossdisc_fake, val_lossgen, val_lossdiscedge, val_lossdiscedge_real, val_lossdiscedge_fake, val_lossgenedge, val_lossstyle, val_losscontent = test(epoch, iter, val_dataloader, log_file_val, output_save=(epoch % args.save_epoch == 0)) #cannydisc
        took = time.time() - start
        if has_val:            
            print_log(log_file_val, epoch, iter, train_losses, train_lossocc, train_iouocc, train_losssdf, train_lossdepth, train_losscolor, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, val_lossdepth, val_losscolor, val_lossdisc, val_lossdisc_real, val_lossdisc_fake, val_lossgen, val_lossdiscedge, val_lossdiscedge_real, val_lossdiscedge_fake, val_lossgenedge, val_lossstyle, val_losscontent, took) #cannydisc
        else:
            print_log(log_file, epoch, iter, train_losses, train_lossocc, train_iouocc, train_losssdf, train_lossdepth, train_losscolor, train_lossdisc, train_lossdisc_real, train_lossdisc_fake, train_lossgen, train_lossdiscedge, train_lossdiscedge_real, train_lossdiscedge_fake, train_lossgenedge, train_lossstyle, train_losscontent, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, took) #cannydisc
        if args.parallel:
            torch.save({'epoch': epoch + 1,'state_dict': model.module.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-epoch-%s.pth' % epoch)) #dpl
            if args.weight_disc_loss > 0:
                torch.save({'epoch': epoch + 1,'state_dict': disc.module.state_dict(),'optimizer' : optimizer_disc.state_dict()}, os.path.join(args.save, 'disc-epoch-%s.pth' % epoch)) #dpl
            if args.weight_discedge_loss >0: #cannydisc
                torch.save({'epoch': epoch + 1,'state_dict': discedge.module.state_dict(),'optimizer' : optimizer_discedge.state_dict()}, os.path.join(args.save, 'discedge-epoch-%s.pth' % epoch)) #cannydisc #dpl
        else:
            torch.save({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-epoch-%s.pth' % epoch)) #dpl
            if args.weight_disc_loss > 0:
                torch.save({'epoch': epoch + 1,'state_dict': disc.state_dict(),'optimizer' : optimizer_disc.state_dict()}, os.path.join(args.save, 'disc-epoch-%s.pth' % epoch)) #dpl
            if args.weight_discedge_loss >0: #cannydisc
                torch.save({'epoch': epoch + 1,'state_dict': discedge.state_dict(),'optimizer' : optimizer_discedge.state_dict()}, os.path.join(args.save, 'discedge-epoch-%s.pth' % epoch)) #cannydisc #dpl
            
    log_file.close()
    if has_val:
        log_file_val.close()
    return



if __name__ == '__main__':
    all_start_time = time.time()
    main()
    print("all time:", time.strftime("%d:%H:%M:%S", time.gmtime(time.time() - all_start_time)))



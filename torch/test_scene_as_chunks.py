from __future__ import division
from __future__ import print_function

"""
compute iou and recall loss
--num_to_vis_known : #save known mask mesh
--num_to_vis : #save all mesh
--max_to_process : maximum of # total file
###(num_to_vis_known >= num_to_vis)###
if Denominator==0, assert error
"""

import argparse
import os, sys
import shutil
import random
import torch
import numpy as np
import gc
import time

import data_util
import scene_dataloader
import scene_dataloader_gated as scene_dataloader
import model_gated_dual_stream_GAN as model_util
import loss as loss_util


COLOR_SPACES = ['rgb', 'lab']

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', required=True, help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
# model params
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=0, help='#points / voxel dim.')
parser.add_argument('--nf_gen', type=int, default=20, help='controls #channels of generator')
parser.add_argument('--no_pass_geo_feats', dest='pass_geo_feats', action='store_false')
parser.add_argument('--input_mask', type=int, default=1, help='input mask')
# test params
parser.add_argument('--num_to_vis', type=int, default=400, help='max num to vis')
parser.add_argument('--num_to_vis_known', type=int, default=400, help='max num to vis')
parser.add_argument('--max_to_process', type=int, default=400, help='max num to process')
parser.add_argument('--vis_only', dest='vis_only', action='store_true')
parser.add_argument('--weight_color_loss', type=float, default=1.0, help='weight color loss vs rest (0 to disable).')
parser.add_argument('--color_thresh', type=float, default=15.0, help='mask colors with all values < color_thresh')
parser.add_argument('--color_truncation', type=float, default=0, help='truncation in voxels for color')
parser.add_argument('--augment_rgb_scaling', dest='augment_rgb_scaling', action='store_true')
parser.add_argument('--augment_scale_min', type=float, default=0.5, help='for color augmentation')
parser.add_argument('--augment_scale_max', type=float, default=1.5, help='for color augmentation')
parser.add_argument('--color_space', type=str, default='lab', help='[rgb, lab]')
parser.add_argument('--stride', type=int, default=32, help='stride for chunking (0 - chunk size)')
# retest
parser.add_argument('--remain_num_proc', type=int, default=0, help='previous num_proc, default 0.')
#known
parser.add_argument('--known', type=float, default=3, help='known truncation for sdf')

parser.set_defaults(vis_only=False, augment_rgb_scaling=False, pass_geo_feats=True)
args = parser.parse_args()
assert(args.color_space in COLOR_SPACES)
if args.input_dim == 0: # set default values
    args.input_dim = (128,64,64)
args.input_nf = 1 + 3
UP_AXIS = 0


print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
print("fixed random seed:", random_seed)

# create model
nf_in_color = 3+1 if args.input_mask == 0 else 4+1 #gated
print("nf_in_color", nf_in_color)
#model = model_util.Generator(nf_in_geo=1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats, truncation=args.truncation, max_data_size=args.input_dim).cuda() #no_ff_mask
model = model_util.Generator(nf_in_geo=1+1, nf_in_color=nf_in_color, nf=args.nf_gen, pass_geo_feats=args.pass_geo_feats, truncation=args.truncation, max_data_size=args.input_dim).cuda() #gated
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)

def test(dataloader, output_vis, output_vis_known, output_vis_known_miss, output_vis_known_input, num_to_vis, num_to_vis_known):  #miss  
    all_start_time = time.time()
    model.eval()
    
    chunk_dim = args.input_dim
    args.max_input_height = chunk_dim[0]
    if args.stride == 0:
        args.stride = chunk_dim[1]
    pad = 2

    num_proc = 0
    num_vis = 0
    total_num_proc = args.remain_num_proc + 1
    num_batches = len(dataloader)
    print('starting testing...')
    with torch.no_grad():
        for t, sample in enumerate(dataloader): 
            start_time = time.time()
            inputs = sample['input']
            sdfs = sample['sdf']
            mask = sample['mask']
            colors = sample['colors']
            known = sample['known']
            #known = (known <= 1).cuda()
            known = (known <= args.known).cuda()
            mask_hole = sample['mask_hole'] #gated
            mask_hole = mask_hole.cuda() #miss
            
            max_input_dim = np.array(sdfs.shape[2:])
            #print("sdfs.shape", sdfs.shape) #sdfs.shape (1, 1, 149, 91, 60)
            if args.max_input_height > 0 and max_input_dim[UP_AXIS] > args.max_input_height:
                max_input_dim[UP_AXIS] = args.max_input_height
                inputs = inputs[:,:,:args.max_input_height]
                if mask is not None:
                    mask = mask[:,:,:args.max_input_height]
                if sdfs is not None:
                    sdfs = sdfs[:,:,:args.max_input_height]
                if colors is not None:
                    colors = colors[:,:args.max_input_height]
                if known is not None:
                    known = known[:,:,:args.max_input_height]
                if mask_hole is not None:
                    mask_hole = mask_hole[:,:,:args.max_input_height] #gated
            sys.stdout.write('[ %d | %d ] %s (%d, %d, %d)                ' % (num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1], max_input_dim[2]))
            
            output_colors = torch.zeros(colors.shape)
            output_sdfs = torch.zeros(sdfs.shape)
            output_norms = torch.zeros(sdfs.shape)
            output_occs = torch.zeros(sdfs.shape, dtype=torch.uint8)
            
            # chunk up the scene
            chunk_input = torch.ones(1, args.input_nf, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_mask = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_target_sdf = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda()
            chunk_target_colors = torch.zeros(1, chunk_dim[0], chunk_dim[1], chunk_dim[2], 3, dtype=torch.uint8).cuda()
            chunk_mask_hole = torch.ones(1, 1, chunk_dim[0], chunk_dim[1], chunk_dim[2]).cuda() #gated
            
            #print("inputs.shape", inputs.shape) #inputs.shape (1, 4, 128, 91, 60)
            
            for y in range(0, max_input_dim[1], args.stride):
                for x in range(0, max_input_dim[2], args.stride):
                    chunk_input_mask = torch.abs(inputs[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]]) < args.truncation
                    #print("chunk_input_mask.shape", chunk_input_mask.shape) # chunk_input_mask.shape (1, 4, 128, 64, 60)
                    if torch.sum(chunk_input_mask).item() == 0:
                        continue
                    sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d) (%d, %d)    ' % (num_proc, args.max_to_process, sample['name'], max_input_dim[0], max_input_dim[1], max_input_dim[2], y, x))

                    fill_dim = [min(sdfs.shape[2], chunk_dim[0]), min(sdfs.shape[3] - y, chunk_dim[1]), min(sdfs.shape[4] - x, chunk_dim[2])]
                    chunk_target_sdf.fill_(float('inf')) ### -inf???
                    chunk_target_colors.fill_(0)
                    chunk_input[:,0].fill_(-args.truncation)
                    chunk_input[:,1:].fill_(0)
                    chunk_mask.fill_(0) 
                    chunk_mask_hole.fill_(1) #gated
                    chunk_input[:,:,:fill_dim[0],:fill_dim[1],:fill_dim[2]] = inputs[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]]
                    chunk_mask[:,:,:fill_dim[0],:fill_dim[1],:fill_dim[2]] = mask[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]]
                    chunk_target_sdf[:,:,:fill_dim[0],:fill_dim[1],:fill_dim[2]] = sdfs[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]]
                    chunk_target_colors[:,:fill_dim[0],:fill_dim[1],:fill_dim[2],:] = colors[:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]]
                    chunk_mask_hole[:,:,:fill_dim[0],:fill_dim[1],:fill_dim[2]] = mask_hole[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]] #gated
                    
                    target_for_sdf, target_for_colors = loss_util.compute_targets(chunk_target_sdf.cuda(), args.truncation, False, None, chunk_target_colors.cuda())
                    
                    
                    output_occ = None
                    output_occ, output_sdf, output_color = model(chunk_input, chunk_mask, chunk_mask_hole, pred_sdf=[True, True], pred_color=args.weight_color_loss > 0)

                    if output_occ is not None:
                        occ = torch.nn.Sigmoid()(output_occ.detach()) > 0.5
                        locs = torch.nonzero((torch.abs(output_sdf.detach()[:,0]) < args.truncation) & occ[:,0]).cpu()
                    else:
                        locs = torch.nonzero(torch.abs(output_sdf[:,0]) < args.truncation).cpu()
                    locs = torch.cat([locs[:,1:], locs[:,:1]],1)
                    output_sdf = [locs, output_sdf[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]].detach().cpu()]
                    if args.weight_color_loss == 0:
                        output_color = None
                    else:
                        output_color = [locs, output_color[locs[:,-1],:,locs[:,0],locs[:,1],locs[:,2]]]
                    
                    output_locs = output_sdf[0] + torch.LongTensor([0,y,x,0])                    
                    if args.stride < chunk_dim[1]:
                        min_dim = [0, y, x]
                        max_dim = [0 + chunk_dim[0], y + chunk_dim[1], x + chunk_dim[2]]
                        if y > 0:
                            min_dim[1] += pad
                        if x > 0:
                            min_dim[2] += pad
                        if y + chunk_dim[1] < max_input_dim[1]:
                            max_dim[1] -= pad
                        if x + chunk_dim[2] < max_input_dim[2]:
                            max_dim[2] -= pad
                        for k in range(3):
                            max_dim[k] = min(max_dim[k], sdfs.shape[k+2])                        
                        outmask = (output_locs[:,0] >= min_dim[0]) & (output_locs[:,1] >= min_dim[1]) & (output_locs[:,2] >= min_dim[2]) & (output_locs[:,0] < max_dim[0]) & (output_locs[:,1] < max_dim[1]) & (output_locs[:,2] < max_dim[2])
                    else:
                        outmask = (output_locs[:,0] < output_sdfs.shape[2]) & (output_locs[:,1] < output_sdfs.shape[3]) & (output_locs[:,2] < output_sdfs.shape[4])
                    output_locs = output_locs[outmask]
                    output_sdf = [output_sdf[0][outmask], output_sdf[1][outmask]]
                    if output_color is not None:
                        output_color = [output_color[0][outmask], output_color[1][outmask]]
                        output_color = (output_color[1] + 1) * 0.5
                        
                        output_colors[0,output_locs[:,0],output_locs[:,1],output_locs[:,2],:] += output_color.detach().cpu()
                    if output_occ is not None:
                        output_occs[:,:,:chunk_dim[0],y:y+chunk_dim[1],x:x+chunk_dim[2]] = occ[:,:,:fill_dim[0],:fill_dim[1],:fill_dim[2]]
                        
                    output_sdfs[0,0,output_locs[:,0],output_locs[:,1],output_locs[:,2]] += output_sdf[1][:,0].detach().cpu()
                    output_norms[0,0,output_locs[:,0],output_locs[:,1],output_locs[:,2]] += 1
            
            # normalize
            mask = output_norms > 0
            output_norms = output_norms[mask]
            output_sdfs[mask] = output_sdfs[mask] / output_norms # avg overlap part
            output_sdfs[~mask] = -float('inf')
            mask = mask.view(1, mask.shape[2], mask.shape[3], mask.shape[4])
            output_colors[mask,:] = output_colors[mask,:] / output_norms.view(-1,1)
            output_colors = torch.clamp(output_colors*255, 0, 255)
            
            sdfs = torch.clamp(sdfs, -args.truncation, args.truncation)
            output_sdfs = torch.clamp(output_sdfs, -args.truncation, args.truncation)
            mask_hole = mask_hole.bool()
            
            # save miss and input visualize #miss
            if known is None:
                known_input = mask_hole
                known_miss = ~mask_hole
            else:
                known_input = known & mask_hole
                known_miss = known & (~mask_hole)
            # miss part
            sdfs_known_miss = sdfs.clone()
            sdfs_known_miss[~known_miss] = -float('inf')
            colors_known_miss = colors.clone() #delv4
            if args.color_space == 'lab':
                colors_known_miss[~known_miss[0]] = torch.tensor([0,0,0]).to(torch.float32) #delv4
            elif args.color_space == 'rgb':
                colors_known_miss[~known_miss[0]] = torch.tensor([0,0,0]).to(torch.uint8) #delv4
            output_sdfs_known_miss = output_sdfs.clone()
            output_sdfs_known_miss[~known_miss] = -float('inf')
            locs_known = torch.nonzero(torch.abs(output_sdfs_known_miss[0,0]) < args.truncation)
            vis_pred_sdf_known = [None]
            vis_pred_color_known = [None]
            sdf_vals_known = output_sdfs_known_miss[0,0,locs_known[:,0],locs_known[:,1],locs_known[:,2]].view(-1)
            vis_pred_sdf_known[0] = [locs_known.cpu().numpy(), sdf_vals_known.cpu().numpy()]
            vals_known = output_colors[0,locs_known[:,0],locs_known[:,1],locs_known[:,2]]
            vis_pred_color_known[0] = vals_known.cpu().numpy()
            data_util.save_predictions(output_vis_known_miss, np.arange(1), sample['name'], None, sdfs_known_miss.cpu().numpy(), colors_known_miss.cpu().numpy(), None, None, vis_pred_sdf_known, vis_pred_color_known, None, None, sample['world2grid'], args.truncation, args.color_space) # don't save input mesh
            # input part
            sdfs_known_input = sdfs.clone()
            sdfs_known_input[~known_input] = -float('inf')
            colors_known_input = colors.clone() #delv4
            if args.color_space == 'lab':
                colors_known_input[~known_input[0]] = torch.tensor([0,0,0]).to(torch.float32) #delv4
            elif args.color_space == 'rgb':
                colors_known_input[~known_input[0]] = torch.tensor([0,0,0]).to(torch.uint8) #delv4
            output_sdfs_known_input = output_sdfs.clone()
            output_sdfs_known_input[~known_input] = -float('inf')
            locs_known = torch.nonzero(torch.abs(output_sdfs_known_input[0,0]) < args.truncation)
            vis_pred_sdf_known = [None]
            vis_pred_color_known = [None]
            sdf_vals_known = output_sdfs_known_input[0,0,locs_known[:,0],locs_known[:,1],locs_known[:,2]].view(-1)
            vis_pred_sdf_known[0] = [locs_known.cpu().numpy(), sdf_vals_known.cpu().numpy()]
            vals_known = output_colors[0,locs_known[:,0],locs_known[:,1],locs_known[:,2]]
            vis_pred_color_known[0] = vals_known.cpu().numpy()
            data_util.save_predictions(output_vis_known_input, np.arange(1), sample['name'], None, sdfs_known_input.cpu().numpy(), colors_known_input.cpu().numpy(), None, None, vis_pred_sdf_known, vis_pred_color_known, None, None, sample['world2grid'], args.truncation, args.color_space) # don't save input mesh
            #miss
            
            # known mask
            if known is not None:
                sdfs_known = sdfs.clone()
                sdfs_known[~known] = -float('inf')
                colors_known = colors.clone() #delv4
                if args.color_space == 'lab':
                    colors_known[~known[0]] = torch.tensor([0,0,0]).to(torch.float32) #delv4
                elif args.color_space == 'rgb':
                    colors_known[~known[0]] = torch.tensor([0,0,0]).to(torch.uint8) #delv4
                output_sdfs_known = output_sdfs.clone()
                output_sdfs_known[~known] = -float('inf')
            
            
            if num_vis < num_to_vis_known:
                locs_known = torch.nonzero(torch.abs(output_sdfs_known[0,0]) < args.truncation)
                vis_pred_sdf_known = [None]
                vis_pred_color_known = [None]
                sdf_vals_known = output_sdfs_known[0,0,locs_known[:,0],locs_known[:,1],locs_known[:,2]].view(-1)
                vis_pred_sdf_known[0] = [locs_known.cpu().numpy(), sdf_vals_known.cpu().numpy()]
                if args.weight_color_loss > 0:
                    vals_known = output_colors[0,locs_known[:,0],locs_known[:,1],locs_known[:,2]]
                    vis_pred_color_known[0] = vals_known.cpu().numpy()
                if output_occs is not None:
                    pred_occ = output_occs.cpu().numpy().astype(np.float32)
                
                data_util.save_predictions(output_vis_known, np.arange(1), sample['name'], None, sdfs_known.cpu().numpy(), colors_known.cpu().numpy(), None, None, vis_pred_sdf_known, vis_pred_color_known, None, None, sample['world2grid'], args.truncation, args.color_space, pred_occ=pred_occ) # don't save input and target mesh
                if num_vis < num_to_vis:
                    inputs = inputs.cpu().numpy() #delv4
                    locs = torch.nonzero(torch.abs(output_sdfs[0,0]) < args.truncation)
                    vis_pred_sdf = [None]
                    vis_pred_color = [None]
                    sdf_vals = output_sdfs[0,0,locs[:,0],locs[:,1],locs[:,2]].view(-1)
                    vis_pred_sdf[0] = [locs.cpu().numpy(), sdf_vals.cpu().numpy()]
                    if args.weight_color_loss > 0:
                        vals = output_colors[0,locs[:,0],locs[:,1],locs[:,2]]
                        vis_pred_color[0] = vals.cpu().numpy()
                    data_util.save_predictions(output_vis, np.arange(1), sample['name'], inputs, sdfs.cpu().numpy(), colors.cpu().numpy(), None, None, vis_pred_sdf, vis_pred_color, None, None, sample['world2grid'], args.truncation, args.color_space, pred_occ=pred_occ)
                num_vis += 1
            if num_proc%50 == 0:
                print("all time:", time.strftime("%d:%H:%M:%S", time.gmtime(time.time() - all_start_time)))
            num_proc += 1
            total_num_proc += 1
            
            #del t, sample, inputs, sdfs, mask, mask_hole, colors, known, output_colors, output_sdfs, output_norms, output_occs, chunk_input, chunk_mask, chunk_target_sdf, chunk_target_colors, chunk_input_mask, target_for_sdf, target_for_colors, output_occ, output_sdf, output_color, occ, locs, output_locs, outmask, target_occ_iou, pred_occ_iou, loss_iou_occ, loss_recall_occ, sdfs_known, output_sdfs_known, colors_known, locs_known, vis_pred_sdf_known, vis_pred_color_known, sdf_vals_known, vals_known, pred_occ, vis_pred_sdf, vis_pred_color, sdf_vals, vals
            #del sdfs_known_miss, colors_known_miss, output_sdfs_known_miss, sdfs_known_input, colors_known_input, output_sdfs_known_input #miss
            gc.collect()
    print("all time:", time.strftime("%d:%H:%M:%S", time.gmtime(time.time() - all_start_time)))

def main():
    # data files
    test_files, _, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '', 0)
    if len(test_files) > args.max_to_process:
        test_files = test_files[:args.max_to_process]
    else:
        args.max_to_process = len(test_files)
    #random.shuffle(test_files)
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, True, args.augment_rgb_scaling, (args.augment_scale_min, args.augment_scale_max), args.color_truncation, args.color_space, target_path=args.target_data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate_voxels)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=scene_dataloader.collate_voxels)
    print('test_dataset', len(test_dataloader))

    if os.path.exists(args.output):
        if args.vis_only:
            print('warning: output dir %s exists, will overwrite any existing files')
        else:
            #input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
            raw_input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
            #shutil.rmtree(args.output) #delete the existed director
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    output_vis_path = os.path.join(args.output, 'vis')
    if not os.path.exists(output_vis_path):
        os.makedirs(output_vis_path)
    output_vis_known_path = os.path.join(args.output, 'vis_known')
    if not os.path.exists(output_vis_known_path):
        os.makedirs(output_vis_known_path)
    output_vis_known_miss_path = os.path.join(args.output, 'vis_known_miss') #miss
    if not os.path.exists(output_vis_known_miss_path):
        os.makedirs(output_vis_known_miss_path)
    output_vis_known_input_path = os.path.join(args.output, 'vis_known_input') #miss
    if not os.path.exists(output_vis_known_input_path):
        os.makedirs(output_vis_known_input_path)

    # start testing
    print('starting testing...')
    #test(test_dataloader, output_vis_path, output_vis_known_path, args.num_to_vis, args.num_to_vis_known) #miss
    test(test_dataloader, output_vis_path, output_vis_known_path, output_vis_known_miss_path, output_vis_known_input_path, args.num_to_vis, args.num_to_vis_known) #miss


if __name__ == '__main__':
    main()



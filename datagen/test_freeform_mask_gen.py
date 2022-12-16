"""
# default value
hole style: free-form
diameter: 15 = (30cm)
length: 64 = (step)
division = 1
"""

import os,struct
import numpy as np
import random
import time
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to input data (mp_sdf_2cm_target)')
    parser.add_argument('--output', required=True, help='path to output data')
    parser.add_argument('--list_val', default='../filelists/mp-rooms_val_ff-scenes.txt', help='only generate the scenes in test/val list') #52
    parser.add_argument('--list_test', default='../filelists/mp-rooms_test_ff-scenes.txt', help='only generate the scenes in test/val list') #394
    parser.add_argument('--diameter', default=15, help='diameter of sphere')
    parser.add_argument('--max_length', default=64, help='total steps of all strokes')
    parser.add_argument('--div', default=1, help='1: be proportioned to the training dataset')
    opt = parser.parse_args()
    return opt

def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense #.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])

def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values

def dense_to_sparse_np_color(grid, color, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = color[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values

def main(config):
    all_start_time = time.time()
    # output file setting
    out_path_data = config.output
    print("out_path_data", out_path_data)
    direct = np.zeros(3)
    
    # To generate free-form mask on all the scenes in dataset, uncommment the code belowed
#     sdf_list = os.listdir(config.input)
#     sdf_list.sort()
    # all the scenes in dataset
    
    # To generate free-form mask on the test/val scenes in dataset, uncommment the code belowed
    sdf_list = []
    f = open(config.list_val,'r')
    lines = f.readlines()
    for line in lines:
        sdf_name = line.split('\n')[0]
        sdf_list.append(sdf_name)
    f.close()
    f = open(config.list_test,'r')
    lines = f.readlines()
    for line in lines:
        sdf_name = line.split('\n')[0]
        sdf_list.append(sdf_name)
    f.close()
    # the test/val scenes in dataset
    
    print("sdf_list:", len(sdf_list)) # 52+394 = 446
    count = 0
    for sdf_name in sdf_list:
        if '.sdf' in sdf_name:
            if sdf_name in sdf_list:
                
                if sdf_name == 'gYvKGZ5eRqb_room0__0__.sdf':
                    count += 1
                    continue
                print(count, sdf_name, "time spent:", time.strftime("%H:%M:%S", time.gmtime(time.time() - all_start_time)))
                
                # reading target sdf file
                try:
                    fin = open(os.path.join(config.input, sdf_name), 'rb')
                    dimx = struct.unpack('Q', fin.read(8))[0]
                    dimy = struct.unpack('Q', fin.read(8))[0]
                    dimz = struct.unpack('Q', fin.read(8))[0]
                    voxelsize = struct.unpack('f', fin.read(4))[0]
                    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
                except:
                    print('failed to read file:', sdf_name)
                    continue
                world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
                # load sdf
                num = struct.unpack('Q', fin.read(8))[0]
                locs = struct.unpack('I'*num*3, fin.read(num*3*4))
                locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
                locs = np.flip(locs,1).copy() # convert to zyx ordering
                sdf = struct.unpack('f'*num, fin.read(num*4))
                sdf = np.asarray(sdf, dtype=np.float32)
                sdf /= voxelsize
                fin.close()

                # load color
                file_color = os.path.join(config.input, sdf_name[:-4]+".colors")
                cfin = open(file_color, 'rb')
                colors = None
                cdimx = struct.unpack('Q', cfin.read(8))[0]
                cdimy = struct.unpack('Q', cfin.read(8))[0]
                cdimz = struct.unpack('Q', cfin.read(8))[0]
                assert cdimx == dimx and cdimy == dimy and cdimz == dimz
                num = struct.unpack('Q', cfin.read(8))[0]
                colors = struct.unpack('B'*num*3, cfin.read(num*3))
                colors = np.asarray(colors, dtype=np.uint8).reshape(num, 3)
                #TODO always loads dense
                #colors = sparse_to_dense_np(locs, colors, cdimx, cdimy, cdimz, 0)
                cfin.close()

                # To range sdf value between (-4, 4)
                # trunc_id = np.abs(sdf) > 4
                # sdf[trunc_id] = -float('inf')
                # colors[trunc_id,:] = (0, 0, 0)
                # To clamp range sdf value between (-5, 5)
                # sdf = np.clip(sdf, -5, 5)
                
                sdf_sparse = sdf
                sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
                colors = sparse_to_dense_np(locs, colors, cdimx, cdimy, cdimz, 0)
                
                # scale the length
                num_hole = max((dimx//64)*(dimy//64)//config.div, 1)
                mask = np.ones((dimz, dimy, dimx)) # z,y,x

                # cutting hole
                for i in range(num_hole):
                    occ = np.where(np.abs(sdf_sparse) <= 1)[0] ### z,y,x
                    idx = random.randrange(occ.shape[0])
                    idx2 = occ[idx] ### z,y,x
                    cen = locs[idx2]
                    ran_step = random.randrange(10) + 20
                    step = 0
                    count_random = 0
                    half = config.diameter//2
                    while step <= config.max_length:
                        step += 1
                        ran_step -= 1

                        temp = np.ones((config.diameter, config.diameter, config.diameter))*100
                        x_min = int(max(0,    cen[2]-half))
                        x_max = int(min(dimx, cen[2]+half+1))
                        y_min = int(max(0,    cen[1]-half))
                        y_max = int(min(dimy, cen[1]+half+1))
                        z_min = int(max(0,    cen[0]-half))
                        z_max = int(min(dimz, cen[0]+half+1))
                        for i in range(x_min,x_max):
                            for j in range(y_min,y_max):
                                for k in range(z_min,z_max):
                                    if math.sqrt((i-cen[2])**2+(j-cen[1])**2+(k-cen[0])**2) < half+1:
                                        mask[k,j,i] = 0
                                        temp[(k-cen[0]+half), (j-cen[1]+half), (i-cen[2]+half)] = sdf[k,j,i]
                                        sdf[k,j,i] = -float('inf')
                                        colors[k,j,i] = [0,0,0]
                        temp_small = temp[half-3:half+4, half-3:half+4, half-3:half+4]
                        arround = np.where(np.abs(temp_small) <= 1)
                        if ran_step <= 0:
                            #print("random")
                            ran_step = random.randrange(10) + 20
                            count_random += 1
                            idx = random.randrange(occ.shape[0])
                            idx2 = occ[idx] ### z,y,x
                            cen = locs[idx2]
                        elif len(arround[0]) == 0:
                            #print("end")
                            arround = np.where(np.abs(temp) <= 5)
                            if len(arround[0]) == 0:
                                #print("end random")
                                ran_step = random.randrange(10) + 20
                                count_random += 1
                                idx = random.randrange(occ.shape[0])
                                idx2 = occ[idx] ### z,y,x
                                cen = locs[idx2]
                            else:
                                #print("end bigger arround",len(arround[0]))
                                idx = random.randrange(len(arround[0]))
                                direct[0] = arround[0][idx]-half
                                direct[1] = arround[1][idx]-half
                                direct[2] = arround[2][idx]-half
                                cen[0] = max(0, min(dimz, cen[0]+direct[0]))
                                cen[1] = max(0, min(dimy, cen[1]+direct[1]))
                                cen[2] = max(0, min(dimx, cen[2]+direct[2]))
                        else:
                            #print("arround",len(arround[0]))
                            idx = random.randrange(len(arround[0]))
                            direct[0] = arround[0][idx]-3
                            direct[1] = arround[1][idx]-3
                            direct[2] = arround[2][idx]-3
                            cen[0] = max(0, min(dimz, cen[0]+direct[0]))
                            cen[1] = max(0, min(dimy, cen[1]+direct[1]))
                            cen[2] = max(0, min(dimx, cen[2]+direct[2]))
                    #print("count_random", count_random)
                    
                # writeing input chunk file
                fout = open(os.path.join(out_path_data, sdf_name), 'wb')
                fout.write(struct.pack('Q', dimx))
                fout.write(struct.pack('Q', dimy))
                fout.write(struct.pack('Q', dimz))
                fout.write(struct.pack('f', voxelsize))
                world2grid = world2grid.reshape([1, -1])
                fout.write(struct.pack('=%sf' % world2grid.size, *world2grid.flatten('F')))
                # locs, values = dense_to_sparse_np(sdf, 5)
                # locs, values = dense_to_sparse_np(sdf, 5.1)
                locs = locs.transpose()
                num = locs.shape[0]
                fout.write(struct.pack('Q', num))
                locs = np.flip(locs,1).copy()
                locs = locs.reshape([1, -1])
                fout.write(struct.pack('=%sI' % locs.size, *locs.flatten('F')))
                values *= voxelsize
                values = values.reshape([1, -1])
                fout.write(struct.pack('=%sf' % values.size, *values.flatten('F')))
                mask = mask.reshape([1, -1]).astype('bool_')
                fout.write(struct.pack('=%s?' % mask.size, *mask.flatten('F')))
                fout.close()
                
                fout = open(os.path.join(out_path_data, sdf_name[:-4]+".colors"), 'wb')
                fout.write(struct.pack('Q', dimx))
                fout.write(struct.pack('Q', dimy))
                fout.write(struct.pack('Q', dimz))
                # locs, values = dense_to_sparse_np_color(sdf, colors, 5)
                # locs, values = dense_to_sparse_np_color(sdf, colors, 5.1)
                locs = locs.transpose()
                num = locs.shape[0]
                fout.write(struct.pack('Q', num))
                values = values.reshape([1, -1])
                fout.write(struct.pack('=%sB' % values.size, *values.flatten('F')))
                fout.close()

                count +=1
                if count % 10 == 0:
                    print(count, "sdf done.", "time spent:", time.strftime("%H:%M:%S", time.gmtime(time.time() - all_start_time)))
    print("total #sdf:", count, "all time:", time.strftime("%d:%H:%M:%S", time.gmtime(time.time() - all_start_time)))

if __name__=='__main__':
    random.seed(2021)
    np.random.seed(2021)
    config = parse_args()
    os.makedirs(config.output, exist_ok=True) #exist_ok=True: won't repot error even if the dir exist
    main(config)

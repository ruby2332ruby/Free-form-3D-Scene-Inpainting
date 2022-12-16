# [v3]: no concate final feature inter output become last output
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv(nn.Module):
    """
    Gated Convlution layer with activation (default activation:None norm:None)
    Params: same as conv3d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 norm=None, 
                 activation=None):
        super(GatedConv, self).__init__()
        self.activation = activation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm(out_channels) if norm is not None else None
        
#         ##print
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
            
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        #print("Gated: ", self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        #print("input.shape", input.shape)
        x = self.conv(input)
        mask = self.mask_conv(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.norm is not None:
            #print("output.shape", x.shape)
            return self.norm(x)
        else:
            return x

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out
        


def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num


class SNConv2WithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv2WithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        return x

class Discriminator2D(nn.Module):
    def __init__(self, nf_in, nf, patch_size, image_dims, patch, use_bias, disc_loss_type='vanilla'):
        nn.Module.__init__(self)
        self.use_bias = use_bias
        approx_receptive_field_sizes = [4, 10, 22, 46, 94, 190, 382, 766]
        num_layers = len(approx_receptive_field_sizes)
        if patch:
            for k in range(len(approx_receptive_field_sizes)):
                if patch_size < approx_receptive_field_sizes[k]:
                    num_layers = k
                    break
        assert(num_layers >= 1)
        self.patch = patch
        self.nf = nf
        dim = min(image_dims[0], image_dims[1])
        num = int(math.floor(math.log(dim, 2)))
        num_layers = min(num, num_layers)
        activation = None if num_layers == 1 else torch.nn.LeakyReLU(0.2, inplace=True)
        self.discriminator_net = torch.nn.Sequential(
            SNConv2WithActivation(nf_in, 2*nf, 4, 2, 1, activation=activation, bias=self.use_bias),
        )
        if num_layers > 1:
            activation = None if num_layers == 2 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p1', SNConv2WithActivation(2*nf, 4*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        if num_layers > 2:
            activation = None if num_layers == 3 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p2', SNConv2WithActivation(4*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        for k in range(3, num_layers):
            activation = None if num_layers == k+1 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p%d' % k, SNConv2WithActivation(8*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        self.final = None
        if not patch or disc_loss_type != 'hinge': #hack
            self.final = torch.nn.Conv2d(nf*8, 1, 1, 1, 0)        
        num_params = count_num_model_params(self.discriminator_net)
        print('#params discriminator', count_num_model_params(self.discriminator_net))
        
        self.compute_valid = None
        if patch:
            self.compute_valid = torch.nn.Sequential(
                torch.nn.AvgPool2d(4, stride=2, padding=1),
            )
            for k in range(1, num_layers):
                self.compute_valid.add_module('p%d' % k, torch.nn.AvgPool2d(4, stride=2, padding=1))
    
    def compute_valids(self, valid):
        if self.compute_valid is None:
            return None
        valid = self.compute_valid(valid)
        return valid

    def forward(self, x, alpha=None):
        for k in range(len(self.discriminator_net)-1):
            x = self.discriminator_net[k](x)
        x = self.discriminator_net[-1](x) 
        
        if self.final is not None:
            x = self.final(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class Conv3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bn=True, activation=torch.nn.LeakyReLU(0.2, inplace=True), bn_before=False, norm_type='batch'):
        super(Conv3, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.bn = None 
        if bn and norm_type != 'none':
            if norm_type == 'batch':
                self.bn = torch.nn.BatchNorm3d(out_channels, momentum=0.8)
            elif norm_type == 'inst':
                self.bn = torch.nn.InstanceNorm3d(out_channels)#, momentum=0.8)
            else:
                raise
        self.bn_before = bn_before

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, verbose=False):    
        x = self.conv3d(x)
        if verbose:
            print(' (conv3)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<1]).item())
            for b in range(x.shape[0]):
                print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if self.bn_before and self.bn is not None:
            x = self.bn(x)
            if verbose:
                print(' (conv3-bn)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if self.activation is not None:
            x = self.activation(x)
            if verbose:
                print(' (conv3-act)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        if not self.bn_before and self.bn is not None:
            x = self.bn(x)
            if verbose:
                print(' (conv3-bn)x', x.shape, torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.mean(x[torch.abs(x)<=1]).item())
                for b in range(x.shape[0]):
                    print('     x(%d)'%b, x[b].shape, torch.min(x[b]).item(), torch.max(x[b]).item(), torch.mean(x[b]).item(), torch.mean(x[b][torch.abs(x[b])<=1]).item(), torch.sum(torch.abs(x[b])<=1).item())
        return x

class Generator(nn.Module):
    def __init__(self, nf_in_geo, nf_in_color, nf, pass_geo_feats, max_data_size, truncation, max_dilation=1):
        nn.Module.__init__(self)
        self.data_dim = 3
        self.nf = nf
        self.input_mask = nf_in_color > 3
        self.max_data_size = np.array(max_data_size)
        #self.use_bias = True
        self.use_bias = False
        self.pass_geo_feats = pass_geo_feats
        self.max_dilation = max_dilation
        self.truncation = truncation
        self.interpolate_mode = 'nearest'
        #nf_in_geo=1+1 nf_in_color=4+1 nf=20 pass_geo_feats=True max_data_size=(128, 64, 64) truncation=3

        #use_dilations = True
        #nz_in = max_data_size[0]
        #kz = [1] * 34 if nz_in == 1 else [5, 4,3, 4,3,3, 3,3,3,3,3, 3, 3,3, 3,3,3, 5, 4,4, 3,3,3, 3,3,3,3, 3,3, 3,3, 3,3,3]
        #dz = [1] * 34 if (not use_dilations or nz_in == 1) else [min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation), min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation)]
        #dyx = [1] * 34 if not use_dilations else [min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation), min(2,max_dilation),min(4,max_dilation),min(8,max_dilation),min(16,max_dilation)]
        
        # === geo net === 
        # encoder
        self.en_1 = GatedConv(nf_in_geo, self.nf, 5, 1, 2, bias=self.use_bias, norm=None, activation=nn.LeakyReLU(0.2, True))
        self.en_2 = GatedConv(self.nf, self.nf, 4, 2, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_3 = GatedConv(self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_4 = GatedConv(self.nf, 2*self.nf, 4, 2, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_5 = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_6 = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_7 = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        # decoder
        self.de_7 = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_6 = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_5 = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_4 = GatedConv(2*self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_3 = GatedConv(self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_2 = GatedConv(self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_1 = GatedConv(self.nf+nf_in_geo, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True)) #single_stream
        
        
        
        # === color net === 
        self.en_1_color = GatedConv(nf_in_color, self.nf, 5, 1, 2, bias=self.use_bias, norm=None, activation=nn.LeakyReLU(0.2, True))
        self.en_2_color = GatedConv(self.nf, self.nf, 4, 2, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_3_color = GatedConv(self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_4_color = GatedConv(self.nf, 2*self.nf, 4, 2, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True)) # pass_geo
        self.en_5_color = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_6_color = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.en_7_color = GatedConv(2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True)) #single_stream
        # decoder
        self.de_7_color = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_6_color = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_5_color = GatedConv(2*self.nf+2*self.nf, 2*self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_4_color = GatedConv(2*self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_3_color = GatedConv(self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_2_color = GatedConv(self.nf+self.nf, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        self.de_1_color = GatedConv(self.nf+nf_in_color, self.nf, 3, 1, 1, bias=self.use_bias, norm=nn.BatchNorm3d, activation=nn.LeakyReLU(0.2, True))
        
        # feature projection
        self.occ_resolver = Bottleneck(self.nf, self.nf//4)
        self.occ_feature_projection = nn.Sequential(
            nn.Conv3d(self.nf, 1, 3, 1, 1),
            #nn.Sigmoid() #write in train process
        )
        self.sdf_resolver = Bottleneck(self.nf, self.nf//4)
        self.sdf_feature_projection = nn.Sequential(
            nn.Conv3d(self.nf, 1, 3, 1, 1),
            nn.Tanh()
        )
        self.color_resolver = Bottleneck(self.nf, self.nf//4)
        self.color_feature_projection = nn.Sequential(
            nn.Conv3d(self.nf, 3, 3, 1, 1),
            nn.Tanh()
        )
        
#         # feature fusion #DUALv3
#         self.fusion_output = nn.Sequential(
#             nn.Conv3d(self.nf+self.nf, self.nf, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.output_layer_occ = nn.Sequential(
#             nn.Conv3d(self.nf, self.nf//2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(self.nf//2, 1, 3, 1, 1),
#         )
#         self.output_layer_sdfcolor = nn.Sequential(
#             nn.Conv3d(self.nf, self.nf//2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(self.nf//2, self.nf//2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(self.nf//2, 1+3, 3, 1, 1),
#             nn.Tanh()
#         )
        # single stream pass geo feature to color
#         self.pass_geo = nn.Sequential(
#                 nn.Conv3d(self.nf, self.nf, 4, 2, 1, bias=self.use_bias),
#                 nn.LeakyReLU(0.2, True),
#                 nn.BatchNorm3d(self.nf),
#             )
        
        
        num_params_geo = count_num_model_params(self.en_1) + count_num_model_params(self.en_2) + count_num_model_params(self.en_3) + count_num_model_params(self.en_4) + count_num_model_params(self.en_5) + count_num_model_params(self.en_6) + count_num_model_params(self.en_7) + count_num_model_params(self.de_7) + count_num_model_params(self.de_6) + count_num_model_params(self.de_5) + count_num_model_params(self.de_4)+ count_num_model_params(self.de_3) + count_num_model_params(self.de_2) + count_num_model_params(self.de_1) + count_num_model_params(self.occ_resolver) + count_num_model_params(self.occ_feature_projection)  + count_num_model_params(self.sdf_resolver) + count_num_model_params(self.sdf_feature_projection)
        num_params_color = count_num_model_params(self.en_1_color) + count_num_model_params(self.en_2_color) + count_num_model_params(self.en_3_color) + count_num_model_params(self.en_4_color) + count_num_model_params(self.en_5_color) + count_num_model_params(self.en_6_color) + count_num_model_params(self.en_7_color) + count_num_model_params(self.de_7_color) + count_num_model_params(self.de_6_color) + count_num_model_params(self.de_5_color) + count_num_model_params(self.de_4_color)+ count_num_model_params(self.de_3_color) + count_num_model_params(self.de_2_color) + count_num_model_params(self.de_1_color) + count_num_model_params(self.color_resolver) + count_num_model_params(self.color_feature_projection) #+ count_num_model_params(self.pass_geo)
        #num_params_fusion = count_num_model_params(self.fusion_output) + count_num_model_params(self.output_layer_occ) + count_num_model_params(self.output_layer_sdfcolor) #DUALv3
        print('#params(geo) = ', num_params_geo)
        print('#params(color) = ', num_params_color)
        #print('#params(fusion) = ', num_params_fusion) #DUALv3
        #print('#params(total) = ', num_params_geo+num_params_color+num_params_fusion) #DUALv3
        print('#params(total) = ', num_params_geo+num_params_color) #DUALv3
        

    def update_sizes(self, input_max_dim):
        self.max_data_size = input_max_dim

    def forward(self, x, mask, mask_hole, pred_color, pred_sdf): #DUALv3
    #def forward(self, x, mask, mask_hole): #DUALv3
        # preprocess input geo
        x_geo = x[:,:1,:,:,:].clone() #fixmiss
        x_geo[torch.abs(x_geo) >= self.truncation-0.01] = 0
        #x_geo[torch.abs(x_geo) >= self.truncation-0.01] = 3
        x_geo = x_geo/3 #resize range from [-2.99 2.99] to [-0.99 0.99]
        x_geo = torch.cat([x_geo, mask_hole], 1)
        #scale_factor = 2 if self.max_data_size[0] > 1 else (1,2,2)
        scale_factor = 2
        
        # preprocess input color
        x_color = x[:,1:4,:,:,:].clone() #fixmiss #sobel
#         x_sobel = x[:,4,:,:,:].clone().unsqueeze(1) #fixmiss #sobel
        x_color = x_color*2-1  #resize range from [0 1] to [-1 1]
        if self.input_mask:
            masked_x = x_color * (1 - mask) + mask
            x_color = torch.cat((x_color, mask), dim=1)
#         x_color = torch.cat((x_color, x_sobel), dim=1) #sobel
        x_color = torch.cat((x_color, mask_hole), dim=1)
        
        # geo encoder
        x_en1_geo = self.en_1(x_geo)
        x_en2_geo = self.en_2(x_en1_geo)
        x_en3_geo = self.en_3(x_en2_geo)
        x_en4_geo = self.en_4(x_en3_geo)
        x_en5_geo = self.en_5(x_en4_geo)
        x_en6_geo = self.en_6(x_en5_geo)
        x_en7_geo = self.en_7(x_en6_geo)
        
        # color encoder
        x_en1_color = self.en_1_color(x_color)
        x_en2_color = self.en_2_color(x_en1_color)
        x_en3_color = self.en_3_color(x_en2_color)
        x_en4_color = self.en_4_color(x_en3_color)
        x_en5_color = self.en_5_color(x_en4_color)
        x_en6_color = self.en_6_color(x_en5_color)
        x_en7_color = self.en_7_color(x_en6_color)
        
        # geo decoder
        x_en7_geo_cat = torch.cat((x_en7_color, x_en6_geo), dim=1) #skip connection
        x_de7_geo = self.de_7(x_en7_geo_cat)
        x_de7_geo = torch.cat((x_de7_geo, x_en5_geo), dim=1) #skip connection
        x_de6_geo = self.de_6(x_de7_geo)
        x_de6_geo = torch.cat((x_de6_geo, x_en4_geo), dim=1) #skip connection
        x_de5_geo = self.de_5(x_de6_geo)
        x_de5_geo = torch.nn.functional.interpolate(x_de5_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_de5_geo = torch.cat((x_de5_geo, x_en3_geo), dim=1) #skip connection
        x_de4_geo = self.de_4(x_de5_geo)
        x_de4_geo = torch.cat((x_de4_geo, x_en2_geo), dim=1) #skip connection
        x_de3_geo = self.de_3(x_de4_geo)
        x_de3_geo = torch.nn.functional.interpolate(x_de3_geo, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_de3_geo = torch.cat((x_de3_geo, x_en1_geo), dim=1) #skip connection
        x_de2_geo = self.de_2(x_de3_geo)
        x_de2_geo = torch.cat((x_de2_geo, x_geo), dim=1) #skip connection
        x_de1_geo = self.de_1(x_de2_geo)
        
        # color decoder
        x_en7_color = torch.cat((x_en7_geo, x_en6_color), dim=1) #skip connection
        x_de7_color = self.de_7_color(x_en7_color)
        x_de7_color = torch.cat((x_de7_color, x_en5_color), dim=1) #skip connection
        x_de6_color = self.de_6_color(x_de7_color)
        x_de6_color = torch.cat((x_de6_color, x_en4_color), dim=1) #skip connection
        x_de5_color = self.de_5_color(x_de6_color)
        x_de5_color = torch.nn.functional.interpolate(x_de5_color, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_de5_color = torch.cat((x_de5_color, x_en3_color), dim=1) #skip connection
        x_de4_color = self.de_4_color(x_de5_color)
        x_de4_color = torch.cat((x_de4_color, x_en2_color), dim=1) #skip connection
        x_de3_color = self.de_3_color(x_de4_color)
        x_de3_color = torch.nn.functional.interpolate(x_de3_color, scale_factor=scale_factor, mode=self.interpolate_mode)
        x_de3_color = torch.cat((x_de3_color, x_en1_color), dim=1) #skip connection
        x_de2_color = self.de_2_color(x_de3_color)
        x_de2_color = torch.cat((x_de2_color, x_color), dim=1) #skip connection
        x_de1_color = self.de_1_color(x_de2_color)
        
        # project function: from feature space to inter output
#         # geo inter output
#         x_occ_inter = self.occ_resolver(x_de1_geo)
#         x_occ_inter = self.occ_feature_projection(x_occ_inter)
#         x_sdf_inter = self.sdf_resolver(x_de1_geo)
#         x_sdf_inter = self.sdf_feature_projection(x_sdf_inter)
#         x_sdf_inter = x_sdf_inter*3  #resize range from [-1 1] to [-3 3]
#         # color inter output
#         x_rgb_inter = self.color_resolver(x_de1_color)
#         x_rgb_inter= self.color_feature_projection(x_rgb_inter)
        # geo final output
        x_occ = self.occ_resolver(x_de1_geo)
        x_occ = self.occ_feature_projection(x_occ)
        x_sdf = self.sdf_resolver(x_de1_geo)
        x_sdf = self.sdf_feature_projection(x_sdf)
        x_sdf = x_sdf*3  #resize range from [-1 1] to [-3 3]
        # color final output
        x_rgb = self.color_resolver(x_de1_color)
        x_rgb= self.color_feature_projection(x_rgb)
        
#         # BIGFF => concate #DUALv3
#         x_concate = torch.cat((x_de1_geo, x_de1_color), dim=1)
#         x_fusion = self.fusion_output(x_concate)
#         x_occ = self.output_layer_occ(x_fusion)
#         x_output = self.output_layer_sdfcolor(x_fusion)
#         x_sdf = x_output[:,0].unsqueeze(1)
#         x_sdf = x_sdf*3  #resize range from [-1 1] to [-3 3]
#         x_rgb = x_output[:,1:]
        
        #return x_occ_inter, x_sdf_inter, x_rgb_inter, x_occ, x_sdf, x_rgb
        return x_occ, x_sdf, x_rgb









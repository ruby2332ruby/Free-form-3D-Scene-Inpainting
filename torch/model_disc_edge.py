"""
#cannydisc: 2D Edge Discriminator (include 2d edge detection)
#labdisc: 3 channel color discedge + lab color space
#nonormal: discedge channel: (1+1) + (1+1) no nomral images
"""

# Edge Detection network from CTSDG github
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeDetector(nn.Module):

    def __init__(self, in_channels=1, mid_channels=16, out_channels=1):
        super(EdgeDetector, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image):
        
        image = self.projection(image)
        edge = self.res_layer(image)
        edge = self.relu(edge + image)
        edge = self.out_layer(edge)

        return edge
    
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

    
class Discriminator2DEdge(nn.Module):
    def __init__(self, nf_in, nf, patch_size, image_dims, patch, use_bias, disc_loss_type='vanilla'):
        nn.Module.__init__(self)
        self.use_bias = use_bias
        #self.edgedetec = EdgeDetector() #cannydisc #labdisc
        self.edgedetec = EdgeDetector(in_channels=3) #cannydisc #labdisc
        self.smallvalue = 1.e-17
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
#         # cannydisc #nonormal
#         #print("x", x.shape) #('x', (2, 4, 256, 320)) [graycolor(1), edge/invalidmask(1)]*2
#         x_new = x.clone()
#         # input2d_valid_e : invalid e == 0
#         input2d_g = x[:, 0, :, :].clone().unsqueeze(1)
#         input2d_e = x[:, 1, :, :].clone().unsqueeze(1)
#         invalid2d_invalid = input2d_g == 0
# #         print("invalid2d_invalid", invalid2d_invalid.sum().item())
#         input2d_valid_e = self.edgedetec(input2d_g)
#         input2d_valid_e = input2d_valid_e*((~invalid2d_invalid).float())
#         x_new[:, 1, :, :] = input2d_valid_e.squeeze(1)
#         # target_valid_e/synth_valid_e : invalid e == ground_truth e
#         target2d_g = x[:, 2, :, :].clone().unsqueeze(1)
#         target2d_invalid = x[:, 3, :, :].clone().unsqueeze(1).bool()
# #         print("target2d_invalid", target2d_invalid.sum().item())
#         target2d_valid_e = self.edgedetec(target2d_g)
#         target2d_valid_e = target2d_valid_e*((~target2d_invalid).float()) + input2d_e*target2d_invalid.float()
#         x_new[:, 3, :, :] = target2d_valid_e.squeeze(1)
#         # cannydisc #nonormal

        # cannydisc #nonormal #labdisc
        #print("x", x.shape) #('x', (2, 8, 256, 320)) [graycolor(3), edge/invalidmask(1)]*2
        x_new = x.clone()
        # input2d_valid_e : invalid e == 0
        input2d_g = x[:, 0:3, :, :].clone()
        input2d_e = x[:, 3, :, :].clone().unsqueeze(1)
        invalid2d_invalid = (input2d_g == 0)[:, 0, :, :].unsqueeze(1)
#         print("invalid2d_invalid", invalid2d_invalid.sum().item())
        input2d_valid_e = self.edgedetec(input2d_g)
        input2d_valid_e = input2d_valid_e*((~invalid2d_invalid).float())
        x_new[:, 3, :, :] = input2d_valid_e.squeeze(1)
        # target_valid_e/synth_valid_e : invalid e == ground_truth e
        target2d_g = x[:, 4:7, :, :].clone()
        target2d_invalid = x[:, 7, :, :].clone().unsqueeze(1).bool()
#         print("target2d_invalid", target2d_invalid.sum().item())
        target2d_valid_e = self.edgedetec(target2d_g)
        target2d_valid_e = target2d_valid_e*((~target2d_invalid).float()) + input2d_e*target2d_invalid.float()
        x_new[:, 3, :, :] = target2d_valid_e.squeeze(1)
        # cannydisc #nonormal #labdisc

        
        for k in range(len(self.discriminator_net)-1):
            x_new = self.discriminator_net[k](x_new)
        x_new = self.discriminator_net[-1](x_new) 
        
        if self.final is not None:
            x_new = self.final(x_new)
        x_new = x_new.permute(0, 2, 3, 1).contiguous()
        return x_new
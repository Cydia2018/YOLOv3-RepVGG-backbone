import torch
import torch.nn as nn
import numpy as np

import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

use_dict_old = {'stage0.rbr_dense':'module_list.1','stage0.rbr_1x1':'module_list.2',
            # ---------
            'stage1.0.rbr_dense':'module_list.5','stage1.0.rbr_1x1':'module_list.6',
            'stage1.1.rbr_dense':'module_list.9','stage1.1.rbr_1x1':'module_list.10','stage1.1.rbr_identity':'module_list.11',
            'stage1.2.rbr_dense':'module_list.13','stage1.2.rbr_1x1':'module_list.14','stage1.2.rbr_identity':'module_list.15',
            'stage1.3.rbr_dense':'module_list.17','stage1.3.rbr_1x1':'module_list.18','stage1.3.rbr_identity':'module_list.19',
            # ---------
            'stage2.0.rbr_dense':'module_list.21','stage2.0.rbr_1x1':'module_list.22',
            'stage2.1.rbr_dense':'module_list.25','stage2.1.rbr_1x1':'module_list.26','stage2.1.rbr_identity':'module_list.27',
            'stage2.2.rbr_dense':'module_list.29','stage2.2.rbr_1x1':'module_list.30','stage2.2.rbr_identity':'module_list.31',
            'stage2.3.rbr_dense':'module_list.33','stage2.3.rbr_1x1':'module_list.34','stage2.3.rbr_identity':'module_list.35',
            'stage2.4.rbr_dense':'module_list.37','stage2.4.rbr_1x1':'module_list.38','stage2.4.rbr_identity':'module_list.39',
            'stage2.5.rbr_dense':'module_list.41','stage2.5.rbr_1x1':'module_list.42','stage2.5.rbr_identity':'module_list.43',
            # ---------
            'stage3.0.rbr_dense':'module_list.45','stage3.0.rbr_1x1':'module_list.46',
            'stage3.1.rbr_dense':'module_list.49','stage3.1.rbr_1x1':'module_list.50','stage3.1.rbr_identity':'module_list.51',
            'stage3.2.rbr_dense':'module_list.53','stage3.2.rbr_1x1':'module_list.54','stage3.2.rbr_identity':'module_list.55',
            'stage3.3.rbr_dense':'module_list.57','stage3.3.rbr_1x1':'module_list.58','stage3.3.rbr_identity':'module_list.59',
            'stage3.4.rbr_dense':'module_list.61','stage3.4.rbr_1x1':'module_list.62','stage3.4.rbr_identity':'module_list.63',
            'stage3.5.rbr_dense':'module_list.65','stage3.5.rbr_1x1':'module_list.66','stage3.5.rbr_identity':'module_list.67',
            'stage3.6.rbr_dense':'module_list.69','stage3.6.rbr_1x1':'module_list.70','stage3.6.rbr_identity':'module_list.71',
            'stage3.7.rbr_dense':'module_list.73','stage3.7.rbr_1x1':'module_list.74','stage3.7.rbr_identity':'module_list.75',
            'stage3.8.rbr_dense':'module_list.77','stage3.8.rbr_1x1':'module_list.78','stage3.8.rbr_identity':'module_list.79',
            'stage3.9.rbr_dense':'module_list.81','stage3.9.rbr_1x1':'module_list.82','stage3.9.rbr_identity':'module_list.83',
            'stage3.10.rbr_dense':'module_list.85','stage3.10.rbr_1x1':'module_list.86','stage3.10.rbr_identity':'module_list.87',
            'stage3.11.rbr_dense':'module_list.89','stage3.11.rbr_1x1':'module_list.90','stage3.11.rbr_identity':'module_list.91',
            'stage3.12.rbr_dense':'module_list.93','stage3.12.rbr_1x1':'module_list.94','stage3.12.rbr_identity':'module_list.95',
            'stage3.13.rbr_dense':'module_list.97','stage3.13.rbr_1x1':'module_list.98','stage3.13.rbr_identity':'module_list.99',
            'stage3.14.rbr_dense':'module_list.101','stage3.14.rbr_1x1':'module_list.102','stage3.14.rbr_identity':'module_list.103',
            'stage3.15.rbr_dense':'module_list.105','stage3.15.rbr_1x1':'module_list.106','stage3.15.rbr_identity':'module_list.107',
            # ----------
            'stage4.0.rbr_dense':'module_list.109','stage4.0.rbr_1x1':'module_list.110'
            }

use_dict = {'stage0.rbr_dense': 'module_list.0', 'stage0.rbr_1x1': 'module_list.1', 
            'stage1.0.rbr_dense': 'module_list.4', 'stage1.0.rbr_1x1': 'module_list.5', 
            'stage1.1.rbr_dense': 'module_list.8', 'stage1.1.rbr_1x1': 'module_list.9', 'stage1.1.rbr_identity': 'module_list.10', 
            'stage1.2.rbr_dense': 'module_list.12', 'stage1.2.rbr_1x1': 'module_list.13', 'stage1.2.rbr_identity': 'module_list.14', 
            'stage1.3.rbr_dense': 'module_list.16', 'stage1.3.rbr_1x1': 'module_list.17', 'stage1.3.rbr_identity': 'module_list.18', 
            'stage2.0.rbr_dense': 'module_list.20', 'stage2.0.rbr_1x1': 'module_list.21', 
            'stage2.1.rbr_dense': 'module_list.24', 'stage2.1.rbr_1x1': 'module_list.25', 'stage2.1.rbr_identity': 'module_list.26', 
            'stage2.2.rbr_dense': 'module_list.28', 'stage2.2.rbr_1x1': 'module_list.29', 'stage2.2.rbr_identity': 'module_list.30', 
            'stage2.3.rbr_dense': 'module_list.32', 'stage2.3.rbr_1x1': 'module_list.33', 'stage2.3.rbr_identity': 'module_list.34', 
            'stage2.4.rbr_dense': 'module_list.36', 'stage2.4.rbr_1x1': 'module_list.37', 'stage2.4.rbr_identity': 'module_list.38', 
            'stage2.5.rbr_dense': 'module_list.40', 'stage2.5.rbr_1x1': 'module_list.41', 'stage2.5.rbr_identity': 'module_list.42', 
            'stage3.0.rbr_dense': 'module_list.44', 'stage3.0.rbr_1x1': 'module_list.45', 
            'stage3.1.rbr_dense': 'module_list.48', 'stage3.1.rbr_1x1': 'module_list.49', 'stage3.1.rbr_identity': 'module_list.50', 
            'stage3.2.rbr_dense': 'module_list.52', 'stage3.2.rbr_1x1': 'module_list.53', 'stage3.2.rbr_identity': 'module_list.54', 
            'stage3.3.rbr_dense': 'module_list.56', 'stage3.3.rbr_1x1': 'module_list.57', 'stage3.3.rbr_identity': 'module_list.58', 
            'stage3.4.rbr_dense': 'module_list.60', 'stage3.4.rbr_1x1': 'module_list.61', 'stage3.4.rbr_identity': 'module_list.62', 
            'stage3.5.rbr_dense': 'module_list.64', 'stage3.5.rbr_1x1': 'module_list.65', 'stage3.5.rbr_identity': 'module_list.66', 
            'stage3.6.rbr_dense': 'module_list.68', 'stage3.6.rbr_1x1': 'module_list.69', 'stage3.6.rbr_identity': 'module_list.70', 
            'stage3.7.rbr_dense': 'module_list.72', 'stage3.7.rbr_1x1': 'module_list.73', 'stage3.7.rbr_identity': 'module_list.74', 
            'stage3.8.rbr_dense': 'module_list.76', 'stage3.8.rbr_1x1': 'module_list.77', 'stage3.8.rbr_identity': 'module_list.78', 
            'stage3.9.rbr_dense': 'module_list.80', 'stage3.9.rbr_1x1': 'module_list.81', 'stage3.9.rbr_identity': 'module_list.82', 
            'stage3.10.rbr_dense': 'module_list.84', 'stage3.10.rbr_1x1': 'module_list.85', 'stage3.10.rbr_identity': 'module_list.86', 
            'stage3.11.rbr_dense': 'module_list.88', 'stage3.11.rbr_1x1': 'module_list.89', 'stage3.11.rbr_identity': 'module_list.90', 
            'stage3.12.rbr_dense': 'module_list.92', 'stage3.12.rbr_1x1': 'module_list.93', 'stage3.12.rbr_identity': 'module_list.94', 
            'stage3.13.rbr_dense': 'module_list.96', 'stage3.13.rbr_1x1': 'module_list.97', 'stage3.13.rbr_identity': 'module_list.98', 
            'stage3.14.rbr_dense': 'module_list.100', 'stage3.14.rbr_1x1': 'module_list.101', 'stage3.14.rbr_identity': 'module_list.102', 
            'stage3.15.rbr_dense': 'module_list.104', 'stage3.15.rbr_1x1': 'module_list.105', 'stage3.15.rbr_identity': 'module_list.106', 
            'stage4.0.rbr_dense': 'module_list.108', 'stage4.0.rbr_1x1': 'module_list.109'}

# for rep_name in use_dict_old:
#     # yolo_name = k.replace(rep_name,use_dict[rep_name])
#     yolo_name = use_dict_old[rep_name].split('.')[0]+'.'+str(int(use_dict_old[rep_name].split('.')[-1])-1)
#     use_dict[rep_name]=yolo_name

# print(use_dict)


def get_equivalent_kernel_bias2(weight):
    kernel3x3, bias3x3 = fuse_bn_tensor(weight[0:6])
    kernel1x1, bias1x1 = fuse_bn_tensor(weight[6:])
    return [kernel3x3 + pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1]

def get_equivalent_kernel_bias3(weight):
    kernel3x3, bias3x3 = fuse_bn_tensor(weight[0:6])
    kernel1x1, bias1x1 = fuse_bn_tensor(weight[6:12])
    kernelid, biasid = fuse_bn_tensor_bn(weight[12:])
    return [kernel3x3 + pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid]

def pad_1x1_to_3x3_tensor(kernel1x1):
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

def fuse_bn_tensor(branch):
    kernel = branch[0]
    gamma = branch[1]
    beta = branch[2]
    running_mean = branch[3]
    running_var = branch[4]
    eps = 1e-05
    
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

def fuse_bn_tensor_bn(branch):
    input_dim = list(branch[0].size())[0]
    kernel_value = np.zeros((input_dim, input_dim, 3, 3), dtype=np.float32)
    for i in range(input_dim):
        kernel_value[i, i % input_dim, 1, 1] = 1
    id_tensor = torch.from_numpy(kernel_value).to(branch[0].device)
    kernel = id_tensor
    gamma = branch[0]
    beta = branch[1]
    running_mean = branch[2]
    running_var = branch[3]
    eps = 1e-05

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

# def repvgg_convert():
#     kernel, bias = self.get_equivalent_kernel_bias()
#     return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()

def main():
    device = torch_utils.select_device('2')
    # cfg = 'cfg/yolov3-repvggB0-hand.cfg'
    cfg = 'cfg/yolov3-repvggB1-hand.cfg'
    img_size=416
    # weights = 'weights_repvgg/B0/best.pt'
    # weights = 'weights_repvgg/B0/last.pt'
    weights = 'weights_repvgg/B1/last.pt'
    model = Darknet(cfg, img_size).to(device)
    ck = torch.load(weights, map_location=device)
    if 'model' in ck:
        model_ = ck['model']
    else:
        model_ = ck
    # print(type(list(model_.items())[0][1]))
    convert_dict={}
    tmp2=[]
    tmp3=[]
    for k,v in model_.items():
        ik = int(k.split('.')[1])
        i = 2 * (int(k.split('.')[1]) // 4)

        if i==0 or i==2 or i==10 or i==22 or i==54:
            tmp2.append(v)
            if len(tmp2)==2*6:
                w,b = get_equivalent_kernel_bias2(tmp2)
                convert_dict['module_list.'+str(i)+'.conv.weight']=w
                convert_dict['module_list.'+str(i)+'.conv.bias']=b
                tmp2=[]
        elif i<=55:
            tmp3.append(v)
            if len(tmp3)==3*6-1:
                w,b = get_equivalent_kernel_bias3(tmp3)
                convert_dict['module_list.'+str(i)+'.conv.weight']=w
                convert_dict['module_list.'+str(i)+'.conv.bias']=b
                tmp3=[]
        elif i>55:
            convert_dict[k.replace(str(ik),str(ik-56))]=v

    # torch.save(convert_dict,'repB0_convert_last.pt')
    torch.save(convert_dict,'repB1_convert_last.pt')

main()
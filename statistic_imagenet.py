import os
import sys
import random
import argparse
import functools
import numpy as np
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from utils.counter import count_flops_params
from utils.train_util import load_model_pytorch
from cdp import acculumate_feature, calculate_cdp, \
    get_threshold_by_flops, get_threshold_by_sparsity, \
    TFIDFPruner
import nets

# We use single GPU to calculate statistic of model

parser = argparse.ArgumentParser(description='Get statistic of model on ImageNet')

# Model & Pretrain file
parser.add_argument('--model', required=True, type=str, default='resnet50', help='model name')
parser.add_argument('--pretrained_dir', type=str, default='', help="pretrained file path")
# Data setting
# '/gdata/ImageNet2012/train/'
parser.add_argument('--dataroot', required=True, metavar='PATH',
                    help='Path to ImageNet folder, which should contains train and val ')
parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for statistics (default: 128)')
parser.add_argument('--stop_batch', type=int, default=200, help="Sample batch number")
# Environment Setting
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='Number of data loading workers (default: 8)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Save setting
parser.add_argument('--save_statistic_file', action='store_true')
parser.add_argument('--save_statistic_path', type=str, default="", help="Save path of extracted features")

parser.add_argument('--save_model_file', type=bool, default=True)
parser.add_argument('--save_model_path', type=str, default='./ckpt/', help='model save directory')

# CDP setting
parser.add_argument('--coe', type=float, help='COE, ')
parser.add_argument('--sparsity', type=float, default=0.39,
                        help='target overall target sparsity')

args = parser.parse_args()

if not args.save_statistic_path:
    args.save_statistic_path= './feature_iit/fiit_{}_s{}.pth'.format(args.model,args.stop_batch)
    
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
trainset = torchvision.datasets.ImageFolder(
        os.path.join(args.dataroot,'train/'), 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomSizedCrop(size[0]), #224 , 299
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ]))
search_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

def create_model(model_name:str, pretrained:str):
    if model_name == 'resnet18':
        # model = nets.ResNet18()
        model = models.resnet18()
    if model_name == 'resnet34':
        model = nets.ResNet34()
    if model_name == 'resnet50':
        model = nets.ResNet50()
    if model_name == 'mobilenetv2':
        model = nets.MobileNetV2()
    if model_name in 'mobilenetv1':
        model = nets.MobileNet()
    model=model.cuda()
    load_model_pytorch(model, pretrained, model_name)
    return model
model = create_model(args.model, args.pretrained_dir)

flops, param, detail_flops = count_flops_params(model, (1,3,224,224))

# 可以函数化
print('fiit calculation start:')
feature_iit = acculumate_feature(model, search_loader, args.stop_batch)

tf_idf_map = calculate_cdp(feature_iit,args.coe)

if args.save_statistic_file:
    torch.save(feature_iit, args.save_statistic_path)

tf_idf_array=torch.cat([v for _,v in tf_idf_map.items()],0)
threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-args.sparsity)))[0].min()
print('threshold', threshold)

cdp_config={
    "threshold": threshold,
    "map": tf_idf_map
}
config_list = [{
    'sparsity': args.sparsity,  
    'op_types': ['Conv2d']
}]
pruner = TFIDFPruner(model, config_list, cdp_config=cdp_config)
_ = pruner.compress()

model_name="{}_s{}.pth".format(args.model,args.stop_batch) 
savepath=os.path.join(args.save_model_path, model_name)
torch.save(model.state_dict(), savepath)
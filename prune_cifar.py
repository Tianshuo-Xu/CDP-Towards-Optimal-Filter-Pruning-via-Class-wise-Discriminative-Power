import os
import random
import argparse
import functools
from math import e, log

import numpy as np
import torch
from torch.backends import cudnn

import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from nni.algorithms.compression.pytorch.pruning \
    import L1FilterPruner, L1FilterPrunerMasker

from utils.counter import count_flops_params
from utils.train_util import load_model_pytorch, train, test

from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR
from cdp import acculumate_feature, calculate_cdp, \
    get_threshold_by_flops, get_threshold_by_sparsity,  \
    TFIDFPruner

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False
     
parser = argparse.ArgumentParser(description='CDP Pruner')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or imagenet')

parser.add_argument('--model', type=str, default='resnet56',
                    help='model to use, only resnet56, resnet20')
parser.add_argument('--pretrained_dir', type=str, default=None,
                    help='pretrained file path')
# Data setting
# '/gdata/ImageNet2012/train/'
parser.add_argument('--dataroot', required=True, metavar='PATH',
                    help='Path to Dataset folder')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for statistics (default: 128)')
parser.add_argument('--stop_batch', type=int, default=200, help="Sample batch number")
parser.add_argument('--search_batch_size', type=int, default=10000,
                    help='input batch size for search (default: 256)')
parser.add_argument('--test_batch_size', type=int, default=10000,
                    help='input batch size for testing (default: 256)')
# Environment Setting
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='Number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# Training Setting
parser.add_argument('--epochs', type=int, default=300,
                        help='epochs to fine tune')

# CDP Setting
parser.add_argument('--coe', type=int,
                        help='whether to use balance coefficient')
parser.add_argument('--sparsity', type=float, default=0.39,
                        help='target overall target sparsity')
# Saver Setting
parser.add_argument('--save_acc', type=float, default=94.0,
                        help='save accuracy')
parser.add_argument('--savepath', type=str, default='./ckpt/',
                        help='model save directory')
args = parser.parse_args()
print(args)

#random.seed(args.seed)
#torch.manual_seed(args.seed)
setup_seed(args.seed)

sparsity = args.sparsity
save_file = '_'.join([str(args.model),
                      'coe{}'.format(args.coe),
                      'seed{}'.format(args.seed)
                      ])
args.savepath=os.path.join(args.savepath,args.model)
save_info = os.path.join(args.savepath, save_file)
save_acc = args.save_acc    


def load_model(args):
    if args.model == 'resnet56':
        net = ResNet_CIFAR(depth=56)
        model_path = '../resnet56_base/checkpoint/model_best.pth.tar'

    elif args.model == 'resnet20':
        net = ResNet_CIFAR(depth=20)
        model_path = './models/resnet20_base/checkpoint/model_best.pth.tar'

    elif args.model == 'vgg':
        net = VGG_CIFAR()
        model_path = './models/vgg_base/checkpoint/model_best.pth.tar'
    else:
        print('no model')
        return
    if args.pretrained_dir:
        model_path = args.pretrained_dir

    net = net.cuda()
    load_model_pytorch(net, model_path, args.model)
    return net

mean=[125.31 / 255, 122.95 / 255, 113.87 / 255]
std=[63.0 / 255, 62.09 / 255, 66.70 / 255]
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=args.search_batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


# load pretrained model
net = load_model(args)
feature_iit = acculumate_feature(net, train_all_loader, args.stop_batch)
tf_idf_map = calculate_cdp(feature_iit,args.coe)
threshold = get_threshold_by_sparsity(tf_idf_map,sparsity)
# threshold = get_threshold_by_flops(tf_idf_map,target_reduced_ratio,net)

print('threshold', threshold)

# pruning         
net = load_model(args)
flops, param, detail_flops = count_flops_params(net, (1, 3, 32, 32))
test(net, testloader)
cdp_config={ "threshold": threshold, "map": tf_idf_map }
config_list = [{
    'sparsity': sparsity,  
    'op_types': ['Conv2d']
}]
pruner = TFIDFPruner(net, config_list, cdp_config=cdp_config)
_ = pruner.compress()

flops, param, detail_flops = count_flops_params(net, (1, 3, 32, 32))
save_info += str(flops)[0:4]
print(save_info)

train(net, epochs=300, lr=0.1, train_loader=trainloader, test_loader=testloader, save_info=save_info, save_acc=save_acc)

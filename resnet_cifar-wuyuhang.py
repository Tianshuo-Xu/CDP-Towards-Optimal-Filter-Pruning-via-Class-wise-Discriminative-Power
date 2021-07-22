from torchvision import datasets, transforms, models
import torch
import torch.backends.cudnn as cudnn
import csv
import time, datetime

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L1FilterPrunerMasker
from utils.counter import count_flops_params
from utils.train_util import load_model_pytorch, train, test
import torchvision
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F
from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR
from math import e, log
import argparse
import random
from utils.scheduler import linear_warmup_scheduler
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

parser = argparse.ArgumentParser(description='AF-ISF Pruner')

parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 or imagenet')
parser.add_argument('--data_dir', type=str, default='/gdata/cifar10/',
                        help='dataset directory')
parser.add_argument('--model', type=str, default='resnet56',
                        help='model to use, only resnet56, resnet20')
parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='dataset directory')
parser.add_argument('--coe', type=int, default=0,
                        help='whether to use balance coefficient')
parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 256)')
parser.add_argument('--search_batch_size', type=int, default=10000,
                        help='input batch size for search (default: 10000)')
parser.add_argument('--test_batch_size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=300,
                        help='epochs to fine tune')
parser.add_argument('--sparsity', type=float, default=0.39,
                        help='target overall target sparsity')
parser.add_argument('--save_acc', type=float, default=94.0,
                        help='save accuracy')
parser.add_argument('--save_path', type=str, default='./ckpt/resnet56/',
                        help='dataset directory')
parser.add_argument('--seed', type=int, default=2021,
                        help='random seed')
parser.add_argument('--flops_rate',type=float,default=0.6,
                       help='denote float reduction ratio')
# Warmup
parser.add_argument('--warmup', type=int, default=0, help='Warm up epochs.')
parser.add_argument('--warm_lr', type=float, default=10e-6, help='Warm up learning rate.')
parser.add_argument('--warm_batches', type=int, default=0,help="Warm up batch num, 如果这个参数不为零，那就优先使用此参数而不是warmup")
# Label Smoothing
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing rate.')

args = parser.parse_args()
print(args)

# 设置种子，确保可重复
setup_seed(args.seed)

sparsity = args.sparsity
coe = args.coe
ltime=time.strftime("%H%M%S", time.localtime())
save_info = args.save_path+args.model+'_coe{}_'.format(args.coe)+ltime+"_"
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
trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=args.search_batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


feature_iit = {}
def hook(m, x, y, name):
    #print(name, y.shape)
    global feature_iit
    f = F.relu(y)
    #f = y
    feature = F.avg_pool2d(f, f.size(3))   # mean
    feature = feature.view(f.size()[0], -1)
    feature = feature.transpose(0, 1)
    if name not in feature_iit:
        feature_iit[name] = feature
    else:
        feature_iit[name] = torch.cat([feature_iit[name], feature], 1)

# load pretrained model
net = load_model(args)

# register hook
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d):
    #if not isinstance(m, nn.Linear):
        handler = m.register_forward_hook(functools.partial(hook, name=name))

# Run train dataset, record statics of feature maps
for batch_idx, (inputs, targets) in enumerate(train_all_loader):
    if(not batch_idx%100):
        print(batch_idx)
    net.eval()
    with torch.no_grad():
        net(inputs.cuda())

tf_idf_map = {}

def calc_tf_idf(feature, name):   # feature = [c, n] ([64, 10000])
    global tf_idf_map
    
    # calc tf
    if args.coe == 0:
        balance_coe = 1.0
    else:    
        balance_coe = log((feature.shape[0]/args.coe)*e)
        
#     print(balance_coe)
    # calc idf
    sample_quant = float(feature.shape[1])
    sample_mean = feature.mean(dim=1).view(feature.shape[0], 1)
    sample_inverse = (feature >= sample_mean).sum(dim=1).type(torch.FloatTensor)
    
    # calc tf mean
    feature_sum = feature.sum(dim=0)
    tf = (feature / feature_sum) * balance_coe
    tf_mean = (tf * (feature >= sample_mean)).sum(dim=1)   # Sa
    tf_mean /= (feature >= sample_mean).sum(dim=1)

    idf = torch.log(sample_quant / (sample_inverse + 1.0)).cuda()
    
    importance = tf_mean * idf
    tf_idf_map[name] = importance

for i in feature_iit:
    calc_tf_idf(feature_iit[i], i)


temp = ''
for i in tf_idf_map:
    temp = i
    break
tf_idf_array = tf_idf_map[temp]
for i in tf_idf_map:
    if i == temp:
        continue
    tf_idf_array = torch.cat([tf_idf_array, tf_idf_map[i]], 0)

# pruning 
class TFIDFMasker(L1FilterPrunerMasker):

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter
        global threshold
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)
        
        mask_weight = torch.gt(w_tf_idf_structured, threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}
    
    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        if wrapper.name.split('.')[-1] == 'module':
            name = wrapper.name[0:-7]
        #print(name)
        w_tf_idf_structured = tf_idf_map[name]
        return w_tf_idf_structured


class TFIDFPruner(L1FilterPruner):
    def __init__(self, model, config_list, pruning_algorithm='l1', optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = TFIDFMasker(model, self, **algo_kwargs)

# 二分查找最优阈值
ratio=0
upper=tf_idf_array.shape[0]
mid=int(tf_idf_array.shape[0]*(1-sparsity))
lower=0
count=0
while(np.abs(ratio-args.flops_rate)> 0.003):
    # 只要差距大于 0.5%
    # 如果按sparsity 得到的flops 被压缩率比目标值小 说明保留的filter多 则保留 小侧的区间 
    # 如果按sparsity 得到的flops 被压缩率比目标值大 说明保留的filter少 则保留 大侧的区间

    threshold = torch.topk(tf_idf_array, mid)[0].min()

    net = load_model(args)

    flops_r, param, detail_flops = count_flops_params(net, (1, 3, 32, 32))
    # test(net, testloader)

    config_list = [{'sparsity': sparsity,'op_types': ['Conv2d']}]
    pruner = TFIDFPruner(net, config_list)
    _ = pruner.compress()

    flops, param, detail_flops = count_flops_params(net, (1, 3, 32, 32))
    
    ratio=(flops_r-flops)/flops_r
    if(ratio < args.flops_rate):
        upper=mid
    else:
        lower=mid
    mid=(upper+lower)//2
    count+=1
    print("Cutter Flops is: ",flops)
    print("Rate is: ",ratio)
print("Finally Flops ratio is:",ratio)
print("Time is: ",count,threshold)
print("net: ", type(net))
save_info += str(flops)[0:4]
print(save_info)

acc_record=[0]

# Warmup scheduler
warmup_step = args.warmup*len(trainloader) if not args.warm_batches else args.warm_batches
print("Warmup config:",args.warmup,len(trainloader))

train(net, epochs=300, lr=0.1, train_loader=trainloader, test_loader=testloader, save_info=save_info, save_acc=save_acc,ablation=acc_record, warmup_step=warmup_step,warm_lr=args.warm_lr, label_smoothing=args.label_smoothing)

with open("ablation.csv","a+") as f:
    row=[args.coe, args.flops_rate, args.seed, acc_record[0],args.model]
    row+=[warmup_step,args.label_smoothing]
    csv_writer = csv.writer(f)
    csv_writer.writerow(row)

    
    

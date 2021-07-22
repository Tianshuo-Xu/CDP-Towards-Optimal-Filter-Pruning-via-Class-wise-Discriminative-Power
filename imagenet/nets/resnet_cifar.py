from utils import *
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from nets.base_models import *
import torch.utils.model_zoo as model_zoo


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)

        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(MyNetwork):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=True):
        super(ResNet_CIFAR, self).__init__()
        if cfg is None:
            cfg = [16, 16, 32, 64]
        num_blocks = []
        if depth==20:
            num_blocks = [3, 3, 3]
        elif depth==32:
            num_blocks = [5, 5, 5]
        elif depth==44:
            num_blocks = [7, 7, 7]
        elif depth==56:
            num_blocks = [9, 9, 9]
        elif depth==110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.layer1 = self._make_layer(block, cfg[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[3], num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d'%i, block(self.in_planes, planes, strides[i])))
            self.in_planes = planes
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def cfg2params(self, cfg):
        params = 0
        params += (3 * 3 * 3 * cfg[0] + cfg[0] * 2) # conv1+bn1
        in_c = cfg[0]
        cfg_idx = 1
        for i in range(3):
            num_blocks = self.num_blocks[i]
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                params += (in_c * 3 * 3 * c + 2 * c + c * 3 * 3 * c + 2 * c) # per block params
                if in_c != c:
                    params += in_c * c # shortcut
                in_c = c
            cfg_idx += 1
        params += (self.cfg[-1] + 1) * self.num_classes # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        size = 32
        flops = 0
        flops += (3 * 3 * 3 * cfg[0] * 32 * 32 + cfg[0] * 32 * 32 * 4) # conv1+bn1
        in_c = cfg[0]
        cfg_idx = 1
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                flops += (in_c * 3 * 3 * c * size * size + c * size * size * 4 + c * 3 * 3 * c * size * size + c * size * size * 4) # per block flops
                if in_c != c:
                    flops += in_c * c * size * size # shortcut
                in_c = c
            cfg_idx += 1
        flops += (2 * self.cfg[-1] + 1) * self.num_classes # fc layer
        return flops

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': [16, 16, 32, 64],
            'dataset': 'cifar10',
        }

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
from nni.algorithms.compression.pytorch.pruning \
    import L1FilterPruner, L1FilterPrunerMasker

from utils.counter import count_flops_params
from utils.train_util import load_model_pytorch

class TFIDFMasker(L1FilterPrunerMasker):
    def __init__(self, model, pruner, threshold, tf_idf_map, preserve_round=1, dependency_aware=False):
        super().__init__(model, pruner, preserve_round, dependency_aware)
        self.threshold=threshold
        self.tf_idf_map=tf_idf_map
        
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)
        
        mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}
    
    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        if wrapper.name.split('.')[-1] == 'module':
            name = wrapper.name[0:-7]
        #print(name)
        w_tf_idf_structured = self.tf_idf_map[name]
        return w_tf_idf_structured


class TFIDFPruner(L1FilterPruner):
    def __init__(self, model, config_list, cdp_config:dict, pruning_algorithm='l1', optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = TFIDFMasker(model, self, threshold=cdp_config["threshold"], tf_idf_map=cdp_config["map"], **algo_kwargs)


def acculumate_feature(model, loader, stop:int):
    model=model.cuda()
    features = {}
    
    def hook_func(m, x, y, name, feature_iit):
        #print(name, y.shape)
        f = F.relu(y)
        #f = y
        feature = F.avg_pool2d(f, f.size()[3])
        feature = feature.view(f.size()[0], -1)
        feature = feature.transpose(0, 1)
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)
            
    hook=functools.partial(hook_func, feature_iit=features)
    
    handler_list=[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
        #if not isinstance(m, nn.Linear):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx % (stop//10) == 0:
            print(batch_idx)
        if batch_idx >= stop:
            break
        model.eval()
        with torch.no_grad():
            model(inputs.cuda())
    
    [ k.remove() for k in handler_list]
    return features
    
def calc_tf_idf(feature:dict, name:str, coe:int, tf_idf_map:dict):   # feature = [c, n] ([64, 10000])    
    # calc tf
    balance_coe = np.log((feature.shape[0]/coe)*np.e) if coe else 1.0
    print(balance_coe)
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

def calculate_cdp(features:dict, coe:int):
    tf_idf_map = {}
    for i in features:
        calc_tf_idf(features[i],i, coe=coe, tf_idf_map=tf_idf_map)
    return tf_idf_map

def get_threshold_by_sparsity(mapper:dict, sparsity:float):
    assert 0<sparsity<1
    tf_idf_array=torch.cat([v for v in mapper.values()],0)
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-sparsity)))[0].min()
    return threshold

def get_threshold_by_flops(mapper:dict, reduced_ratio:float):
    pass

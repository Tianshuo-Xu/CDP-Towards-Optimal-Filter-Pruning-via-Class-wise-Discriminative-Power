import os
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.collect_env
from contextlib import contextmanager
from torch.nn.parallel import DistributedDataParallel
class DDP(DistributedDataParallel):
  # Distributed wrapper. Supports asynchronous evaluation and model saving
  def forward(self, *args, **kwargs):
    # DDP has a sync point on forward. No need to do this for eval. This allows us to have different batch sizes
    if self.training: return super().forward(*args, **kwargs)
    else:             return self.module(*args, **kwargs)

  def load_state_dict(self, *args, **kwargs):
    self.module.load_state_dict(*args, **kwargs)

  def state_dict(self, *args, **kwargs):
    return self.module.state_dict(*args, **kwargs)

def get_rank():
    """
    get distributed rank
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank=torch.distributed.get_rank()
    else:
        rank=0
    return rank

def get_world_size():
    """
    get total number of distributed workers
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size=torch.distributed.get_world_size()
    else:
        world_size=1
    return world_size

def init_distributed(cuda):
    """
    initialized distributed backend 
    cuda: bool True to initilialize nccl backend
    """
    world_size=int(os.environ.get('WORLD_SIZE',1))
    distributed=(world_size>1)
    if distributed:
        backend='nccl' if cuda else 'gloo'
        dist.init_process_group(backend=backend,init_method='env://')
        assert dist.is_initialized()
    return distributed
def set_device(cuda, local_rank):
    """
    Sets device based on local_rank and returns instance of torch.device.
    :param cuda: if True: use cuda
    :param local_rank: local rank of the worker
    """
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, warmup=0, keep=False):
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)

    def reduce(self, op):
        """
        Reduces average value over all workers.
        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        print('all reduce start',distributed)
        if distributed:
            backend = dist.get_backend()
            print('backend is {}'.format(backend))
            cuda = (backend == 'nccl')#== dist.Backend.NCCL)
            print(cuda)

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])
            print(_sum)
            print(avg)
            dist.all_reduce(avg)
            dist.all_reduce(_sum)
            print(avg.item(),_sum.item())
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()
            print('all_recude done')

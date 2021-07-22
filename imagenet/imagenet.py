import os
import sys
import csv
import random
import argparse

from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner

from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
from utils.counter_imagenet import count_flops_params
from utils.train_util import load_model_pytorch, LabelSmoothCELoss
from nets.resnet_imagenet import ResNet_ImageNet
from nets.mobilenet_imagenet import MobileNetV2

from run import train, test, save_checkpoint, find_bounds_clr

# Warmup
from utils.scheduler import linear_warmup_scheduler


parser = argparse.ArgumentParser(description=' Models training with PyTorch ImageNet')
parser.add_argument('--dataroot', required=True, metavar='PATH',
                    help='Path to ImageNet train and val folders')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

# Optimization options
parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=1e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[200, 300],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=1, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=20,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpoints
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--make-mask', action='store_true', help='Make mask')
# parser.add_argument('--save', '-s', type=str, default='./ckpt', help='Folder to save checkpoints.') ## GAI
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results') ## GAI
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of MobileNet, multiple of 32 (default 224).')
# Warmup
parser.add_argument('--warmup', type=int, default=0, help='Warm up epochs.')
parser.add_argument('--warm_lr', type=float, default=10e-5, help='Warm up learning rate.')

# Label Smoothing
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing rate.')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    # if args.save is '':
    #     args.save = time_stamp
    save_path = os.path.join(args.results_dir, time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    #model = MobileNetV2()
    if args.model == 'resnet18':
        model = models.resnet18()
    elif args.model == 'resnet50':
        model = ResNet_ImageNet(depth=50)
    elif args.model == 'mobilenetv2':
        model = MobileNetV2()
    else:
        print('No Model Implementation')
        return 
        
    model = model.cuda()
    _, _, _ = count_flops_params(model, (1, 3, 224, 224))
    
    if args.make_mask:
        # masking
        print('masking')
        config_list = [{
                'sparsity': 0.5,  
                'op_types': ['Conv2d']
        }]
        # Act like a placeholder. 
        # After pruned, NNI pruner layer will be inserted into model
        # Prepare for loading
        pruner = L1FilterPruner(model, config_list)
        _ = pruner.compress()
    
    if args.resume:
        load_model_pytorch(model, args.resume, args.model)
        _, _, _ = count_flops_params(model, (1, 3, 224, 224))

    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.input_size,
                                           args.workers)
    # define loss function (criterion) and optimizer
    criterion = LabelSmoothCELoss(args.label_smoothing)
    
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    
    if args.evaluate:
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
        return
    
    # Warmup scheduler
    if args.warmup == 0:
        warmup_scheduler = None
        print('No warm up epoch')
    else:
        warmup_step = args.warmup*len(train_loader)
        print("Warmup config:",args.warmup,len(train_loader))
        warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, args.warm_lr, args.learning_rate)

    if args.find_clr:
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=save_path)
        return

    if args.clr:
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        print('use cosine LR')
        #scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warmup, eta_min=0)
        
    best_test = 0

    # optionally resume from a checkpoint
    data = None        

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)
    
    
    claimed_acc1 = 100.00-31.89
    claimed_acc5 = 100.00-11.24
    train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, optimizer, criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                  best_test, warmup_scheduler)


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test, warmup_scheduler):
    for epoch in range(start_epoch, epochs + 1):

        train_loss, train_accuracy1, train_accuracy5, = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval, scheduler, warmup_scheduler)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype)
        
        if not isinstance(scheduler, CyclicLR) and not warmup_scheduler.if_in_warmup():
            scheduler.step()
            
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


if __name__ == '__main__':
    main()

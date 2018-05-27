import argparse
import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.distributed as dist


def parse_args(arguments=[]):
    """Example of argparse with different inputs.
    Args:
        arguments (list): Arguments passed as a list of strings. This argument
                          can be used when calling the function from a
                          notebook. Alternatively when using the command line,
                          we don't need this variable.
    """
    parser = argparse.ArgumentParser(description="Parser",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet',
                       help='the neural network to use')
    parser.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='max num of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')        
    parser.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    if arguments:  # when calling from notebook
        args = parser.parse_args(arguments)
    else:  # when calling from command line
        args = parser.parse_args()    
    return args


def init_model(m, lr, momentum):
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class
    opt = optim.SGD(m.parameters(), lr, momentum)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion


def get_cifar(batch_size, download_folder='data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=download_folder, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=download_folder, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def manage_multitraining(model, distributed=False, dist_backend, dist_url, world_size, rank):
    # 1. Auto-tune
    torch.backends.cudnn.benchmark=True
    if distributed:
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    
    





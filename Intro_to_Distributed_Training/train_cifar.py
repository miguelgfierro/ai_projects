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
from torchvision import datasets, transforms
import torchvision
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--epochs', type=int, default=5,
                        help='max num of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')        
    parser.add_argument('--momentum', type=float, default=0.9,
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


def init_model(model, lr, momentum):
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class
    optimizer = optim.SGD(model.parameters(), lr, momentum)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def get_cifar(batch_size, download_folder='data', distributed=False):
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
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              shuffle=(train_sampler is None), 
                                              num_workers=4,
                                              pin_memory=True, 
                                              sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root=download_folder, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def manage_multitraining(model, distributed=False, dist_backend=None, dist_url=None, world_size=2, rank=None):
    # 1. Auto-tune
    torch.backends.cudnn.benchmark=True
    if distributed:
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model    


def train(train_loader, model, criterion, optimizer, epoch):
    logger.info("Training epoch {}...".format(epoch))
    model.train()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
def validate(val_loader, model, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    logger.info("Epoch {} - validation accuracy: {:0.2f}%".format(epoch, (100 * correct / total)))

    
def main():
    args = parse_args()
    logger.info("Arguments: {}".format(vars(args)))   
    
    from models.resnet import ResNet50
    model = ResNet50()
    
    optimizer, criterion = init_model(model, args.lr, args.momentum)
    
    train_loader, test_loader = get_cifar(args.batch_size, download_folder='data', distributed=False)
    
    model = manage_multitraining(model)
    
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        validate(test_loader, model, criterion, epoch)


if __name__ == "__main__":
    main()
    
    




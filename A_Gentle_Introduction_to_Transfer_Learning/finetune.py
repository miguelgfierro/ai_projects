
# A Gentle Introduction to Transfer Learning for Image Classification
# Example of use:
# python finetune.py -d /datadrive/simpsons/simpsons -b 128 -m resnet18 -lr 0.001 -g 4 -e 15 -f False


import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD
from torch.autograd import Variable
import argparse
from torchvision import models
from utils import (get_gpu_name, get_number_processors, train_model, create_dataset, 
                   available_models, finetune, freeze_and_train)

SETS = ['train', 'val']
parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset_path", type=str)
parser.add_argument("-b","--batch_size", type=int, default=64)
parser.add_argument("-m","--model_name", type=str, default='resnet18')
parser.add_argument("-lr","--learning_rate", type=float, default=0.001)
parser.add_argument("-lrs","--learning_rate_step", type=float, default=0.1)
parser.add_argument("-lre","--learning_rate_epochs", type=int, default=10)
parser.add_argument("-mm","--momentum", type=float, default=0.9)
parser.add_argument("-g","--gpus", type=int, default=1)
parser.add_argument("-e","--epochs", type=int, default=25)
parser.add_argument("-f","--finetune", type=bool, default=False)
args = parser.parse_args()
print("Arguments: {}".format(args))

model_names = available_models()
if args.model_name not in model_names:
    raise ValueError("Wrong model name, please select one among {}".format(model_names))
        
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("Number of CPU processors: ", get_number_processors())
print("GPU: ", get_gpu_name())

torch.backends.cudnn.benchmark=True # enables cudnn's auto-tuner


# Datasets
dataset = create_dataset(args.dataset_path, batch_size=args.batch_size)


# Training
if args.finetune:
    model, metrics = finetune(dataset, args.model_name, SETS, args.epochs, args.gpus, 
                              args.learning_rate, args.momentum, args.learning_rate_step, 
                              args.learning_rate_epochs, verbose=True)
else:
    model, metrics = freeze_and_train(dataset, args.model_name, SETS, args.epochs, args.gpus, 
                              args.learning_rate, args.momentum, args.learning_rate_step, 
                              args.learning_rate_epochs, verbose=True)

print("Best validation accuracy {} in epoch {}".format(np.max(metrics['val_acc']),
                                                      np.argmax(metrics['val_acc']) + 1))

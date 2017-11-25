import subprocess
import os
import glob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torchvision import models
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from scipy.interpolate import interp1d


def get_number_processors():
    """Get the number of processors in a CPU.
    Returns:
        num (int): Number of processors.
    Examples:
        >>> get_number_processors()
        4
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing #force exception in case mutiprocessing is not installed
        num = multiprocessing.cpu_count()
    return num


def get_gpu_name():
    """Get the GPUs in the system
    Examples:
        >>> get_gpu_name()
        ['Tesla M60', 'Tesla M60', 'Tesla M60', 'Tesla M60']
        
    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_filenames_in_folder(folderpath):
    """ Return the files names in a folder.
    Parameters:
        folderpath (str): folder path
    Returns:
        number (list): list of files
    Examples:
        >>> get_filenames_in_folder('C:/run3x/codebase/python/minsc')
        ['paths.py', 'system_info.py', '__init__.py']

    """
    names = [os.path.basename(x) for x in glob.glob(os.path.join(folderpath, '*'))]
    return names


def create_dataset(data_dir, batch_size=32, sets=['train', 'val'], verbose=True):
    """Create a dataset object given the path. On data_dir there should be a train and validation folder
    and in each of them there should be the folders containing the data. One folder for each class
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=get_number_processors()) 
                   for x in sets}

    if verbose:
        dataset_sizes = {x: len(image_datasets[x]) for x in sets}
        class_names = dataloaders[sets[0]].dataset.class_to_idx
        print("There are {} clases in the dataset: {}".format(len(class_names), class_names))
        print("Sets sizes: ", dataset_sizes)
        for x in sets:   
            c = Counter(item[1] for item in image_datasets[x])
            c = dict(c)
            print("Number of items in set {}: {}".format(x, c))
    return dataloaders


def plot_pytorch_data_stream(dataobject, max_images=8, title=True):
    """Plot a batch of data"""
    inputs, classes = next(iter(dataobject))  
    if max_images > dataobject.batch_size:
        max_images = dataobject.batch_size
        print("Plotting only {} images, which is the batch size".format(max_images))
    inputs = inputs[:max_images,:,:,:]
    classes = classes[:max_images]
    out = torchvision.utils.make_grid(inputs)
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title:
        class_names = dataobject.dataset.classes
        names = [class_names[x] for x in classes]
        plt.title(names)


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25, verbose=True):
    since = time.time()
    sets = list(dataloaders.keys())
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in sets}
    best_model_wts = model.state_dict()
    best_acc = 0.0
    metrics = {'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
        if verbose:
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in sets:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            #metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if verbose: print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                metrics['train_acc'].append(epoch_acc)
                metrics['train_loss'].append(epoch_loss)
            else:
                metrics['val_acc'].append(epoch_acc)
                metrics['val_loss'].append(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()


    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def available_models():
    """Return available pytorch models, callable using `models.__dict__[name]`"""
    model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__") and 
                         callable(models.__dict__[name]))
    return model_names


def plot_metrics(metrics):
    """Plot metrics from training. metrics is a dict containing 'train_acc', 'val_acc', 'train_loss' and
    'val_loss', each of them contains the metrics values in a list"""
    max_epochs = len(metrics['train_acc']) + 1
    epochs = range(1, max_epochs)
    epochs_dx = np.linspace(epochs[0], epochs[-1], num=max_epochs*4, endpoint=True)
    s_train_acc = interp1d(epochs, metrics['train_acc'], kind='cubic')    
    s_val_acc = interp1d(epochs, metrics['val_acc'], kind='cubic')    
    s_train_loss = interp1d(epochs, metrics['train_loss'], kind='cubic')    
    s_val_loss = interp1d(epochs, metrics['val_loss'], kind='cubic')    

    plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(epochs, metrics['train_acc'], 'b.', label='train')
    ax1.plot(epochs_dx, s_train_acc(epochs_dx), 'b')
    ax1.plot(epochs, metrics['val_acc'], 'g.', label='val')
    ax1.plot(epochs_dx, s_val_acc(epochs_dx), 'g')
    ax1.legend( loc="lower right")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.subplots_adjust(right=2)

    ax2 = plt.subplot(122)
    ax2.plot(epochs, metrics['train_loss'], 'b.', label='train')
    ax2.plot(epochs_dx, s_train_loss(epochs_dx), 'b')
    ax2.plot(epochs, metrics['val_loss'], 'g.', label='val')
    ax2.plot(epochs_dx, s_val_loss(epochs_dx), 'g')
    ax2.legend(loc="upper right")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.show()



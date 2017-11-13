import subprocess
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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









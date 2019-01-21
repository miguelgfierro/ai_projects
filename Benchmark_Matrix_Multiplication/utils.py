import sys
import os
import subprocess
import socket
import glob
import psutil 
from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError


def get_number_processors():
    """Get the number of processors in a CPU.
    Returns:
        int: Number of processors.
    Examples:
        >>> num = get_number_processors()
        >>> num >= 2
        True
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing  # force exception in case mutiprocessing is not installed

        num = multiprocessing.cpu_count()
    return num


def get_ram_memory(units="Mb"):
    """Get the RAM memory of the current machine.
    Args:
        units (str): Units [bytes, Kb, Mb, Gb]
    Returns:
        size (float): Memory size.
    Examples:
        >>> num = get_ram_memory("Gb") 
        >>> num >= 4
        True
    """
    s_bytes = psutil.virtual_memory()[0]
    return _manage_memory_units(s_bytes, units)


def get_total_gpu_memory(units="Mb"):
    """Get the memory of the GPUs in the system
    Returns:
        result (list): List of strings with the GPU memory in Mb
    Examples:
        >>> get_total_gpu_memory()
        []
    """
    try:
        memory_list = []
        for gpu in cuda.gpus:
            with gpu:
                meminfo = cuda.current_context().get_memory_info()
                memory_list.append(_manage_memory_units(meminfo[1], units))
        return memory_list
    except CudaSupportError:
        return []


def get_gpu_name():
    """Get the GPU names in the system.
    Returns:
        list: List of strings with the GPU name.
    Examples:
        >>> get_gpu_name()
        []
        
    """
    try:
        return [gpu.name.decode("utf-8") for gpu in cuda.gpus]
    except CudaSupportError:
        return []
    
    
def get_cuda_version():
    """Get CUDA version
    Returns:
        str: Version of the library.
    """
    if sys.platform == "win32":
        raise NotImplementedError("Implement this!")
    elif sys.platform == "linux" or sys.platform == "darwin":
        path = "/usr/local/cuda/version.txt"
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = f.read().replace("\n", "")
            return data
        else:
            return "No CUDA in this machine"
    else:
        raise ValueError("Not in Windows, Linux or Mac")
        
        
def get_cudnn_version():
    """Get the CuDNN version
    Returns:
        str: Version of the library.
    """

    def find_cudnn_in_headers(candiates):
        for c in candidates:
            file = glob.glob(c)
            if file:
                break
        if file:
            with open(file[0], "r") as f:
                version = ""
                for line in f:
                    if "#define CUDNN_MAJOR" in line:
                        version = line.split()[-1]
                    if "#define CUDNN_MINOR" in line:
                        version += "." + line.split()[-1]
                    if "#define CUDNN_PATCHLEVEL" in line:
                        version += "." + line.split()[-1]
            if version:
                return version
            else:
                return "Cannot find CUDNN version"
        else:
            return "No CUDNN in this machine"

    if sys.platform == "win32":
        candidates = [r"C:\NVIDIA\cuda\include\cudnn.h"]
    elif sys.platform == "linux":
        candidates = [
            "/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h",
            "/usr/local/cuda/include/cudnn.h",
            "/usr/include/cudnn.h",
        ]
    elif sys.platform == "darwin":
        candidates = ["/usr/local/cuda/include/cudnn.h", "/usr/include/cudnn.h"]
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    return find_cudnn_in_headers(candidates)


def _manage_memory_units(data_in_bytes, units):
    if units == "bytes":
        return data_in_bytes
    elif units == "Kb":
        return data_in_bytes / 1024
    elif units == "Mb":
        return data_in_bytes / 1024 / 1024
    elif units == "Gb":
        return data_in_bytes / 1024 / 1024 / 1024
    else:
        raise AttributeError("Units not correct")
        

class AttributeDict(dict):
    """Dictionary-like class to access its attributes like a class
    source: https://stackoverflow.com/a/5021467
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
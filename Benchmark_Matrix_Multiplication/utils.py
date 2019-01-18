import sys
import os
import subprocess
import socket
import glob
from psutil import virtual_memory
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


def get_ram():
    """Get RAM memory in Gb.
    Returns:
        float: RAM memory in Gb.
    Examples:
        >>> num = get_ram()
        >>> num >= 2
        True
    """
    mem = virtual_memory()
    return mem.total/1024/1024/1024


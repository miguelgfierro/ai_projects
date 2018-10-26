import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import subprocess


def plot_tsne(t_sne_result):
    t_sne_result_plot = np.transpose(t_sne_result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(t_sne_result_plot[0], t_sne_result_plot[1])
    plt.show()
    
    
def get_gpu_name():
    """Get the GPUs in the system.
    Returns:
        result (list): List of strings with the GPU name.
    Examples (non executable):
        $ get_gpu_name()
        ['Tesla M60', 'Tesla M60', 'Tesla M60', 'Tesla M60']
        
    """
    try:
        out_str = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=subprocess.PIPE,
        ).stdout
        out_list = out_str.decode("utf-8").split("\n")
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)
        
def get_cuda_version():
    """Get CUDA version
    Returns:
        version (str): Version of the library.
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
    """Get the CUDNN version
    Returns:
        version (str): Version of the library.
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


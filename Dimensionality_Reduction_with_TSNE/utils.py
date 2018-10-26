import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(t_sne_result):
    t_sne_result_plot = np.transpose(t_sne_result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(t_sne_result_plot[0], t_sne_result_plot[1])
    plt.show()
    
    
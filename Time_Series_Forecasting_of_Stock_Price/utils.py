import subprocess
import os
import sys
import glob
import json
import numpy as np
import time
from ast import literal_eval
import matplotlib.pyplot as plt


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


def to_1dimension(df, step_size):
    X, y = [], []
    for i in range(len(df)-step_size-1):
        data = df[i:(i+step_size), 0]
        X.append(data)
        y.append(df[i + step_size, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y
     
    
def plot_series(values, xlabel=None, ylabel=None, color='b', legend=None):
    xx = np.arange(1, len(values) + 1, 1)
    plt.plot(xx, values, color, label=legend)
    plt.legend(loc = 'upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
def plot_series_prediction(true_values, train_predict, test_predict, time_ahead=1, title=None,
                           xlabel=None, ylabel=None, color=['g','r','b'], legend=[None,None,None]):
    pred_trainPlot = np.empty_like(true_values)
    pred_trainPlot[:, :] = np.nan
    pred_trainPlot[time_ahead:len(train_predict)+time_ahead, :] = train_predict

    pred_testPlot = np.empty_like(true_values)
    pred_testPlot[:, :] = np.nan
    pred_testPlot[len(train_predict)+(time_ahead*2)+1:len(true_values)-1, :] = test_predict

    plt.plot(true_values, color[0], label=legend[0])
    plt.plot(pred_trainPlot, color[1], label=legend[1])
    plt.plot(pred_testPlot, color[2], label=legend[2])
    plt.legend(loc = 'upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
    
    
    
    
    
    
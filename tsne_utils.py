from time import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None, list_class_to_plot=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    class_label = set(y)
    if list_class_to_plot != None:
        class_label = list_class_to_plot
    num_class = len(class_label)
    item_index = range(X.shape[0])
    color=['r', 'b']
    plt.figure()
    ax = plt.subplot(111)
    for o, label in enumerate(class_label):
        class_index = [i for i in item_index if y[i]==label]
        cmhot = plt.get_cmap("hot")

        plt.scatter(X[class_index, 0], 
            X[class_index, 1], 
            label=label, 
            c=color[int(label)])

    plt.xticks([]), plt.yticks([])
    # ax.legend(loc='upper right')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.05), shadow=True, ncol=1)

    if title is not None:
        plt.title(title)
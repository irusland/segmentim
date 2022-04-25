import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as tfs

from matplotlib.colors import ListedColormap

import os
import pandas as pd
from torchvision.io import read_image
from pathlib import Path
import imagesize
from dataclasses import dataclass
from PIL import Image
from torchvision.transforms import ToTensor
from functools import lru_cache
from datetime import datetime
import time
from itertools import islice
from os import listdir
from os.path import isfile, join
from matplotlib.pyplot import text
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-10):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()

        total = (inputs + targets).sum()

        union = total - intersection 

        IoU = (intersection + smooth)/(union + smooth)
    
        return 1 - IoU
    
    

for cmap in [
    plt.get_cmap(), plt.get_cmap('turbo'), *[plt.get_cmap(n) for n in ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']]
]:
    ncolors = 256
    color_array = cmap(range(ncolors))
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name=f'transparent_{cmap.name}',colors=color_array)
    plt.register_cmap(cmap=map_object)


cmap = plt.get_cmap()
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

def tensor_to_image(t):
    return t.permute(2,1,0).squeeze()

def plot_lineup(im, ms=None, gt=None, figsize=None, ax=None, title=''):
    if not figsize:
        figsize = (20,20)
    if not ax:
        fig, ax = plt.subplots(figsize = figsize)

    img = ax.imshow(im, interpolation='nearest', origin='lower')
    if gt is not None:
        img = ax.imshow(gt,
    #                     interpolation='nearest', 
                        origin='lower', 
                        alpha=0.5, 
                        cmap='transparent_viridis')
    if ms  is not None:
        img = ax.imshow(ms, 
    #                     interpolation='nearest', 
                        alpha=0.5, 
                        origin='lower',
                        cmap='transparent_BuPu',
                       )
    ax.set_title(title)

def dataset_samples(dataset):
    print("Dataset size =", len(dataset))
    figure = plt.figure(figsize=(20, 20))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        name = dataset.get_image_name_by_idx(sample_idx)
        figure.add_subplot(rows, cols, i)
        plt.imshow(tensor_to_image(img), interpolation='nearest', origin='lower')
        plt.imshow(tensor_to_image(label), interpolation='nearest', alpha=0.5, origin='lower', cmap=my_cmap)
        plt.title(f'[{sample_idx}]: {name}')
    plt.show()
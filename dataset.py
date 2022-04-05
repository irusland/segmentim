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

@dataclass(init=True, eq=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int
        
class InriaImageDataset(Dataset):
    def __init__(
        self, 
        base_path, 
        img_dir, 
        label_dir, 
        transform=None, 
        target_transform=None, 
        max_size=None, 
        size_floor=True,
        load=False,
        debug=False,
        crop_tensor=False,
        labeled=True,
    ):
        self.debug = debug
        self.base_path = Path(base_path)
        print('Preparing dataset in', self.base_path)
        self.img_dir = self.base_path / img_dir
        print('Images directory', self.img_dir)
        self.label_dir = self.base_path / label_dir
        print('Labels directory', self.label_dir)
        self.to_tensor =  ToTensor()
        
        transform = tfs.Compose([
            tfs.Normalize(0, 1),
        ])
        
        self.transform = transform
        self.target_transform = target_transform
        self.max_size = max_size
        image_names = os.listdir(self.img_dir)
        label_names = os.listdir(self.label_dir)
        
        self._labeled = labeled
        if self._labeled:
            assert image_names == label_names
        print('Total images =', len(image_names))
        
        self.image_names = image_names

        self.index_to_image_and_rect = []

        for img_name in self.image_names:
            width, height = imagesize.get(self.img_dir / img_name)
            if max_size:
                max_width, max_height = max_size
            else:
                max_width, max_height = width, height

            for wi in range(width // max_width):
                for hi in range(height // max_height):
                    rect = Rectangle(x=wi, y=hi, width=max_width, height=max_height)
                    self.index_to_image_and_rect.append((img_name, rect))
            # todo % rest
        print('Total crops =', len(self.index_to_image_and_rect))
        
        if load:
            set_len = len(self.image_names) + len(label_names)
            print('Loading in memory', set_len, "images")
            for i, (img_name, label_name) in enumerate(zip(self.image_names, label_names)):
                self._get_image(self.img_dir / img_name)
                self._get_image(self.label_dir / label_name)
                print('Loaded', i*2, "/", set_len)
        if crop_tensor:
            self.get_crop = self._get_crop_tensor
        else:
            self.get_crop = self._get_crop_image

    def __len__(self):
        return len(self.index_to_image_and_rect)

#     @lru_cache(maxsize=400)
    def _get_image(self, path):
        if self.debug:
            print('open', path)
        start = time.time()
        image = Image.open(path)
        if self.debug:
            print('Read time', time.time() - start)
        return image
    
#     @lru_cache(maxsize=400)
    def _crop_tensor(self, path, size):
        image = self._get_image(path)
        image = self.to_tensor(image)
        start = time.time()
        mw, mh = size
        c, w, h = image.shape
        cropw = [*(i*mw for i in range(1, w // mw)), w-(w % mw)]
        croph = [*(i*mh for i in range(1, h // mh)), h-(h % mh)]

        crop_map = {}
        vert_splits = torch.tensor_split(image, cropw, dim=2)
        x, y = 0, 0
        for vert_split in vert_splits:
            for split in torch.tensor_split(vert_split, croph, dim=1):
                _, w, h = split.shape
                if w < mw or h < mh:
                    continue
                crop_map[(x, y)] = split
                y += 1
            x += 1
            y = 0
        if self.debug:
            print('Crop time',  time.time() - start)
        return crop_map
    
    def _get_crop_tensor(self, path, crop_rect):
        start = time.time()
        size = (crop_rect.width, crop_rect.height)
        crop_map = self._crop_tensor(path, size)
        image = crop_map[(crop_rect.x, crop_rect.y)]
        if self.debug:
            print('Crop tensor first time',  time.time() - start)
        return image
    
    def _get_crop_image(self, path, crop_rect):
        start = time.time()
        image = self._get_image(path)
        x, y = crop_rect.x * crop_rect.width, crop_rect.y * crop_rect.height
        size = (x, y, x + crop_rect.width, y + crop_rect.height)
        image = image.crop(size)
        image = self.to_tensor(image)
        if self.debug:
            print('Crop image first time',  time.time() - start)
        return image

    def get_image_name_by_idx(self, idx):
        img_name, rect = self.index_to_image_and_rect[idx]
        return img_name

    def __getitem__(self, idx):
        if self.debug:
            print('idx', idx)
        img_name, rect = self.index_to_image_and_rect[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / img_name
        
        image = self.get_crop(img_path, rect)
        if self.transform:
            image = self.transform(image)
            
        if os.path.isfile(label_path):
            label = self.get_crop(label_path, rect)
            label = label[[0]]
            if self.target_transform:
                label = self.target_transform(label)
        else:
            _, w, h = image.shape
            label = torch.zeros((1, w, h))
        return image, label
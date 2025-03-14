from common.utils import HDF5Dataset, GraphCreator
import os
import h5py
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HDF5Dataset_FS_2D_Normalised(Dataset):
    """Load samples of an FS 2D PDE Dataset"""

    def __init__(self, path: str, mode: str, dtype=torch.float64, resolution: list=None, load_all: bool=False, mean=None, std=None):

        """Initialize the dataset object
        Args:
            path: path to dataset
            dtype: floating precision of data
            load_all: load all the data into memory
        """

        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.resolution = (100, 64, 64) if resolution is None else resolution
        self.dataset = f'pde_{self.resolution[0]}-{self.resolution[1]}-{self.resolution[2]}'

        self.nt = self.data[self.dataset].attrs['nt']
        self.dx = self.data[self.dataset].attrs['dx']
        self.dt = self.data[self.dataset].attrs['dt']
        self.tmin = self.data[self.dataset].attrs['tmin']
        self.tmax = self.data[self.dataset].attrs['tmax']

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data
        
        # mean and std
        if mean is None or std is None:
            if mode == 'train':
                self.mean, self.std = self.compute_mean_std()
        else:
            self.mean = mean
            self.std = std

    def compute_mean_std(self):
        sum_ = 0.0
        sum_sq = 0.0
        count = 0
        for i in range(len(self)):
            u = self.data[self.dataset][i]
            sum_ += np.sum(u)
            sum_sq += np.sum(u ** 2)
            count += u.size

        mean = sum_ / count
        std = np.sqrt((sum_sq / count) - (mean ** 2))
        return mean, std

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        u = self.data[self.dataset][idx]
        u_normalised = (u - self.mean) / (self.std + 1e-8)
        return u_normalised

class HDF5Dataset_FS_2D(Dataset):
    """Load samples of an FS 2D PDE Dataset"""

    def __init__(self, path: str, mode: str, dtype=torch.float64, resolution: list=None, load_all: bool=False):

        """Initialize the dataset object
        Args:
            path: path to dataset
            dtype: floating precision of data
            load_all: load all the data into memory
        """

        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.resolution = (100, 128, 128) if resolution is None else resolution
        self.dataset = f'pde_{self.resolution[0]}-{self.resolution[1]}-{self.resolution[2]}'

        self.nt = self.data[self.dataset].attrs['nt']
        self.dx = self.data[self.dataset].attrs['dx']
        self.dt = self.data[self.dataset].attrs['dt']
        self.tmin = self.data[self.dataset].attrs['tmin']
        self.tmax = self.data[self.dataset].attrs['tmax']

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        u = self.data[self.dataset][idx]
        # n_time, Y, X = u.shape
        # u = u.reshape(n_time, Y//4, 4, X//4, 4)
        # u = u.mean(axis=(2, 4))
        return u


train_string = "data/dif_fs_2d_pde_128_train_dataset.h5"
# train_string = 'data/test_boyancy_fs_2d_pde_64_train_dataset.h5'
# train_dataset = HDF5Dataset_FS_2D_Normalised(train_string, mode='train')
train_dataset = HDF5Dataset_FS_2D(train_string, mode='train')

train_loader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0)


n = 0
for i, batch in enumerate(train_loader):
    print(f"Batch {i}:")
    print(f"Shape: {batch.shape}")
    print(f"dtype: {batch.dtype}") 
    # print(batch[0])

    # sample = batch[1].numpy()
    if n == 1:
        break
    n+=1

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

print('Largest Value: ', batch.max().item())
caxs = []
for i, ax in enumerate(axes):
    cax = ax.imshow(batch[i][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=batch.max())
    caxs.append(cax)
    ax.set_title(f"Sample {i+1}")
    fig.colorbar(cax, ax=ax, label='Smoke Intensity')

plt.tight_layout(rect=[0, 0, 1, 0.95])

suptitle = fig.suptitle(f"Time Step: 1", fontsize=16)

def update(frame):
    for i, cax in enumerate(caxs):
        suptitle.set_text(f"Time Step: {frame+1}")
        cax.set_array(batch[i][frame].numpy()) 
    return caxs

anim = FuncAnimation(fig, update, frames=batch.shape[1], interval=100, blit=True)

# Save the animation
# anim.save("not_downsampled_diffused_smoke_simulation_4_samples_128.mp4", fps=10)

plt.show()


# fig, ax = plt.subplots()
# cax = ax.imshow(sample[0], origin='lower', cmap='viridis', animated=True, vmin=0, vmax=sample.max())
# fig.colorbar(cax, ax=ax, label='Smoke Intensity')

# def update(frame):
#     cax.set_array(sample[frame])
#     ax.set_title(f"Time Step: {frame}")
#     return cax,

# anim = FuncAnimation(fig, update, frames=sample.shape[0], interval=100, blit=True)

# anim.save("smoke_simulation.mp4", fps=10)

# plt.show()
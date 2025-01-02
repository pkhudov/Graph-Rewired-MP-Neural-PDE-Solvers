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


class HDF5Dataset_FS_2D(Dataset):
    """Load samples of an FS 2D PDE Dataset"""

    def __init__(self, path: str, mode: str, dtype=torch.float64, resolution: list=None, load_all: bool=False, downsample: bool=False, normalise: bool=False):  

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
        self.resolution = (100, 32, 32) if resolution is None else resolution
        self.dataset = f'pde_{self.resolution[0]}-{self.resolution[1]}-{self.resolution[2]}'
        self.downsample = downsample
        self.normalise = normalise

        self.nt = self.data[self.dataset].attrs['nt']
        self.dx = self.data[self.dataset].attrs['dx']
        self.dt = self.data[self.dataset].attrs['dt']
        self.tmin = self.data[self.dataset].attrs['tmin']
        self.tmax = self.data[self.dataset].attrs['tmax']

        if normalise:
            self.mean, self.std = self.compute_mean_std()

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

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
        if self.downsample:
            n_time, Y, X = u.shape
            u = u.reshape(n_time, Y//4, 4, X//4, 4)
            u = u.mean(axis=(2, 4))
        if self.normalise:
            u = (u - self.mean) / (self.std + 1e-8)
        return u


res32_string = 'data/fs_2d_pde_32_train_dataset.h5'
res128_string = 'data/fs_2d_pde_128_train_dataset.h5'

res32_dataset = HDF5Dataset_FS_2D(res32_string, mode='train', resolution=(100, 32, 32))
res128_dataset = HDF5Dataset_FS_2D(res128_string, mode='train', resolution=(100, 128, 128))
res128_dataset_downsampled = HDF5Dataset_FS_2D(res128_string, mode='train', resolution=(100, 128, 128), downsample=True)


res32_loader = DataLoader(res32_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
res128_dataset_loader = DataLoader(res128_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
res128_dataset_downsampled_loader = DataLoader(res128_dataset_downsampled,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)



res32_sample = next(iter(res32_loader))
res128_sample = next(iter(res128_dataset_loader))
res128_downsampled_sample = next(iter(res128_dataset_downsampled_loader))
print(f"Res32 Sample Shape: {res32_sample.shape}; dtype: {res32_sample.dtype}")
print(f"Res128 Sample Shape: {res128_sample.shape}; dtype: {res128_sample.dtype}")
print(f"Res128 Downsampled Sample Shape: {res128_downsampled_sample.shape}; dtype: {res128_downsampled_sample.dtype}")

print('\nRes32 Sample Max: ', res32_sample.max().item())
print('Res128 Sample Max: ', res128_sample.max().item())
print('Res128 Downsampled Sample Max: ', res128_downsampled_sample.max().item())

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
axes = axes.flatten()

caxs = []
caxs.append(axes[0].imshow(res32_sample[0][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=res32_sample.max()))
axes[0].set_title('32x32')
caxs.append(axes[1].imshow(res128_downsampled_sample[0][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=res128_downsampled_sample.max()))
axes[1].set_title('128x128 Downsampled')
caxs.append(axes[2].imshow(res128_sample[0][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=res128_sample.max()))
axes[2].set_title('128x128')
plt.tight_layout(rect=[0, 0, 1, 0.95])
# fig.colorbar(caxs[0], ax=axes[0], label='Smoke Intensity')
# fig.colorbar(caxs[1], ax=axes[1], label='Smoke Intensity')
# fig.colorbar(caxs[2], ax=axes[2], label='Smoke Intensity')
suptitle = fig.suptitle(f"Time Step: 1", fontsize=16)

def update(frame):
    suptitle.set_text(f"Time Step: {frame+1}")
    caxs[0].set_array(res32_sample[0][frame].numpy())
    caxs[1].set_array(res128_downsampled_sample[0][frame].numpy())
    caxs[2].set_array(res128_sample[0][frame].numpy())
    return caxs

anim = FuncAnimation(fig, update, frames=res32_sample.shape[1], interval=100, blit=True)

# Save the animation
anim.save("compare_32vs128(ds)vs128.mp4", fps=10)

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
import numpy as np
from common.data_creator import HDF5Dataset_FS_2D

LOSS_BASELINE = 70
LOSS_CAYLEY = 60

dataset_path = "data/fs_2d_pde_128_train_dataset.h5"
dataset = HDF5Dataset_FS_2D(dataset_path, mode='train', super_resolution=(100, 128, 128), downsample=True)
total_l2_norm = 0.0

num_samples = len(dataset)
for i in range(num_samples):
    u = dataset[i]
    l2_norm = np.sqrt(np.sum(u**2))
    total_l2_norm += l2_norm
average_l2_norm = total_l2_norm / num_samples

print("Average L2 norm of ground truth):", average_l2_norm)
print("RE baseline:", LOSS_BASELINE / average_l2_norm * 100, "%")
print("RE Cayley:", LOSS_CAYLEY / average_l2_norm * 100, "%")

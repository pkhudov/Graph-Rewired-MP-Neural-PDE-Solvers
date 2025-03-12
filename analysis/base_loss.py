import numpy as np
import torch
from torch.utils.data import DataLoader
from common.data_creator import HDF5Dataset_FS_2D, GraphCreator_FS_2D
from types import SimpleNamespace

res32_string = 'data/fs_2d_pde_32_train_dataset.h5'
res128_string = 'data/fs_2d_pde_128_train_dataset.h5'

res32_dataset = HDF5Dataset_FS_2D(res32_string, mode='train', super_resolution=(100, 32, 32))
res128_dataset = HDF5Dataset_FS_2D(res128_string, mode='train', super_resolution=(100, 128, 128))
res128_dataset_downsampled = HDF5Dataset_FS_2D(res128_string, mode='train', super_resolution=(100, 128, 128), downsample=True)


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
tw = 25
nr_gt_steps = 2
batch_size = 4
pde = SimpleNamespace(Lx=32, Ly=32, dt=1.0, grid_size=(100,32,32), tmin=0.0, tmax=100.0)
eq_variables={}
graph_creator = GraphCreator_FS_2D(pde=pde,
                                 neighbors=2,
                                 time_window=tw,
                                 x_resolution=32,
                                 y_resolution=32,
                                 edge_prob=0,
                                 edge_mode='RadiusOnly',
                                 rand_edges_per_node=0)
criterion = torch.nn.MSELoss(reduction="sum")
losses_base = []
for (base_sample, super_sample) in zip(res32_loader, res128_dataset_downsampled_loader):
    losses_base_tmp = []
    same_steps = [tw * nr_gt_steps] * batch_size

    # Losses for numerical baseline
    for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
                        graph_creator.tw):
        same_steps = [step] * batch_size
        _, labels_super = graph_creator.create_data(super_sample, same_steps)
        _, labels_base = graph_creator.create_data(base_sample, same_steps)
        loss_base = criterion(labels_super, labels_base) / 32
        losses_base_tmp.append(loss_base / batch_size)

    losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

losses_base = torch.stack(losses_base)
print(f'Losses Base: {losses_base.item()}')
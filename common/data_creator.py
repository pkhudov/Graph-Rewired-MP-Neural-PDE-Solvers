import os
import h5py
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils import coalesce, to_undirected
import random
import networkx

# from einops import rearrange

class HDF5Dataset_FS_2D(Dataset):
    """Load samples of an FS 2D PDE Dataset"""

    def __init__(self, path: str, mode: str, dtype=torch.float64, super_resolution: list=None, load_all: bool=False, downsample: bool=False, normalise: bool=False):  

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
        self.downsample=downsample
        self.normalise=normalise
        self.data = f[self.mode]
        self.resolution = (100, 128, 128) if super_resolution is None else super_resolution
        self.dataset = f'pde_{self.resolution[0]}-{self.resolution[1]}-{self.resolution[2]}'

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


class GraphCreator_FS_2D(nn.Module):
    """
    Helper class to construct graph datasets
    params:
        neighbors: now many neighbors the graph has in each direction
        time_window: how many time steps are used for PDE prediction
        time_ratio: time ratio between base and super resolution
        space_ratio: space ratio between base and super resolution
    """

    def __init__(self,
                 pde,
                 neighbors: int=2,
                 time_window: int=10,
                 t_resolution: int=100,
                 x_resolution: int=32,
                 y_resolution: int=32,
                 edge_prob: float = 0.0,
                 edge_path: str = None,
                 edge_mode: str = 'Radiusonly',
                 rand_edges_per_node: int = 2,
                 ):

        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution
        self.y_res = y_resolution
        self.edge_prob = edge_prob
        self.custom_edge_index = None
        self.edge_path = edge_path
        self.edge_mode = edge_mode.lower()
        self.rand_edges_per_node = rand_edges_per_node

        if self.edge_mode == 'cayley-cgp':
            self.cayley_graph = networkx.read_edgelist('full_cayley_edges', nodetype=int)
            self.n_cayley_nodes = self.cayley_graph.number_of_nodes()

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)
    
    def save_edge_index(self, path):
        torch.save(self.custom_edge_index, path)

    def create_data(self, datapoints, steps):
        """
        getting data out of PDEs
        """

        data = torch.Tensor()
        labels = torch.Tensor()

        for (dp, step) in zip(datapoints, steps):
            # d = dp[step - self.tw*2:step]
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]

            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self, data, labels, steps):
        """
        getting graph structure out of data sample
        previous timesteps are combined in one node
        """

        # h = 2
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        ny = self.pde.grid_size[2]

        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        dt = t[1] - t[0]
        x = torch.linspace(0, self.pde.Lx, nx)
        dx = x[1]-x[0]
        y = torch.linspace(0, self.pde.Ly, ny)
        dy = y[1]-y[0]

        radius = self.n * torch.sqrt(dx**2 + dy**2) + 0.0001
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid = grid.view(-1, 2)

        n_nodes = self.x_res * self.y_res

        u_new = torch.Tensor()
        x_new = torch.Tensor()
        t_new = torch.Tensor()
        y_new = torch.Tensor()
        batch = torch.Tensor()

        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):

            u_tmp = torch.transpose(torch.cat([d.reshape(-1, nx*ny) for d in data_batch]), 0, 1)
            y_tmp = torch.transpose(torch.cat([l.reshape(-1, nx*ny) for l in labels_batch]), 0, 1)

            u_new = torch.cat((u_new, u_tmp), )
            x_new = torch.cat((x_new, grid), )
            y_new = torch.cat((y_new, y_tmp), )
            b_new = torch.ones(nx*ny)*b
            t_tmp = torch.ones(nx*ny)*t[step]
            t_new = torch.cat((t_new, t_tmp), )
            batch = torch.cat((batch, b_new), )

        # calculating the edge_index
        local_edge_index = radius_graph(x_new, r=radius, batch=batch.long(), loop=False)

        if self.edge_mode == 'cayley-cgp':
            extra_nodes = self.n_cayley_nodes - n_nodes

            batch_size = int(batch.max().item()) + 1
            new_u_list = []
            new_x_list = []
            new_t_list = []
            new_batch_list = []
            
            self_edges = []
            for b in range(batch_size):
                start = b * n_nodes
                end = (b + 1) * n_nodes
                sample_u = u_new[start:end]
                sample_x = x_new[start:end]
                sample_t = t_new[start:end]

                virtual_u = torch.zeros(extra_nodes, sample_u.shape[1], device=sample_u.device)
                # virtual_x = torch.zeros(extra_nodes, grid.shape[1], device=grid.device)
                virtual_x = sample_x.mean(dim=0, keepdim=True).repeat(extra_nodes, 1)
                # virtual_t = torch.zeros(extra_nodes, device=sample_t.device)
                virtual_t = sample_t[0:1].repeat(extra_nodes)
                new_u_list.append(torch.cat([sample_u, virtual_u], dim=0))
                new_x_list.append(torch.cat([sample_x, virtual_x], dim=0))
                new_t_list.append(torch.cat([sample_t, virtual_t], dim=0))
                new_batch_list.append(torch.full((self.n_cayley_nodes,), b, device=sample_u.device))

                base = b*self.n_cayley_nodes
                for i in range(base + n_nodes, base+self.n_cayley_nodes):
                    self_edges.append([i,i])
                
            u_new = torch.cat(new_u_list, dim=0)
            x_new = torch.cat(new_x_list, dim=0)
            t_new = torch.cat(new_t_list, dim=0)
            batch = torch.cat(new_batch_list, dim=0)

            self_edges = torch.tensor(self_edges, dtype=torch.long, device=x_new.device).t().contiguous()
            local_edge_index = torch.concat([local_edge_index, self_edges], dim=1)

        if self.edge_path is not None:
            self.custom_edge_index = torch.load(self.edge_path)

        if self.custom_edge_index is None and self.edge_mode not in ['radiusonly', 'complete']: # to ensure that the random edges are only generated once
            batch_size = int(batch.max()) + 1
            if self.edge_mode == 'cayley-cgp':
                n_nodes = self.n_cayley_nodes
            all_custom_edges = []
            for sample in range(batch_size):
                offset = sample * n_nodes

                if self.edge_mode == 'erdosrenyi':
                    custom_edges = erdos_renyi_graph(n_nodes, self.edge_prob)
                elif self.edge_mode == 'augmentnode':
                    custom_edges = []
                    for node_i in range(self.x_res * self.y_res):
                        possible_neighbors = list(range(n_nodes))
                        possible_neighbors.remove(node_i)
                        neighbors = random.sample(possible_neighbors, self.rand_edges_per_node)
                        for nb in neighbors:
                            custom_edges.append([node_i, nb])
                    custom_edges = torch.tensor(custom_edges).t()
                elif self.edge_mode == 'randomregular':
                    rnd_reg_graph = networkx.random_regular_graph(self.rand_edges_per_node, n_nodes)
                    custom_edges = torch.tensor(list(rnd_reg_graph.edges)).t()
                elif self.edge_mode == 'cayley':
                    cayley_graph = networkx.read_edgelist('cayley_edges', nodetype=int)
                    custom_edges = torch.tensor(list(cayley_graph.edges)).t()
                elif self.edge_mode == 'cayley-cgp':
                    custom_edges = torch.tensor(list(self.cayley_graph.edges)).t()
                else:
                    raise ValueError(f'Unknown edge mode: {self.edge_mode}')
                
                custom_edges = to_undirected(custom_edges, num_nodes=n_nodes) + offset
                all_custom_edges.append(custom_edges)
            self.custom_edge_index = coalesce(torch.cat(all_custom_edges, dim=1))
            print(f'Generated {self.edge_mode} edges')

            # Join local and custom edges
            # self.custom_edge_index = coalesce(torch.cat((local_edge_index, self.custom_edge_index), 1))

        if self.edge_mode == 'complete':
            self.custom_edge_index = radius_graph(x_new, r=100* torch.sqrt(dx**2 + dy**2) + 0.0001, batch=batch.long(), loop=False)

        graph = Data(x=u_new, edge_index_local=local_edge_index, edge_index_custom=self.custom_edge_index)
        graph.y = y_new

        graph.pos = torch.cat((t_new[:, None], x_new), 1)
        graph.batch = batch.long()

        return graph
    
    
    def create_next_graph(self, graph, pred, virtual, labels, steps):

        """
        getting new graph for the next timestep
        """
        full_pred = torch.cat((pred, virtual), 0)  # shape: (n_full, feat_dim)
        # Update the full state by concatenating the current state with the new prediction and then shifting the temporal window.
        graph.x = torch.cat((graph.x, full_pred), 1)[:, self.tw:]

        #graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]

        #graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]

        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        ny = self.pde.grid_size[2]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        t_new = torch.Tensor()
        y_new = torch.Tensor()

        for (labels_batch, step) in zip(labels, steps):
            y_tmp = torch.transpose(torch.cat([l.reshape(-1, nx*ny) for l in labels_batch]), 0, 1)
            y_new = torch.cat((y_new, y_tmp), )
            if self.edge_mode == 'cayley-cgp':
                t_tmp = torch.ones(self.n_cayley_nodes) * t[step]
            else:
                t_tmp = torch.ones(nx * ny) * t[step]
            t_new = torch.cat((t_new, t_tmp), )

        graph.y = y_new
        graph.pos[:, 0] = t_new

        return graph


    # def create_next_graph(self, graph, pred, labels, steps):
    #     """
    #     Update the graph for the next timestep.
    #     In 'cayley-cgp' mode the graph.x has been extended to include virtual nodes.
    #     Here, we update only the real nodes (the first n_real nodes per sample) with the new prediction,
    #     and leave the virtual nodes unchanged.
    #     """
    #     nt = self.pde.grid_size[0]
    #     nx = self.pde.grid_size[1]
    #     ny = self.pde.grid_size[2]
    #     t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)

    #     if self.edge_mode == 'cayley-cgp':
    #         n_real = self.x_res * self.y_res  # original number of (real) nodes per sample
    #         new_x_list = []
    #         new_batch_list = []
    #         # Process each sample in the batch separately.
    #         for b in torch.unique(graph.batch):
    #             idx = (graph.batch == b).nonzero(as_tuple=True)[0]
    #             # Assume that the first n_real nodes in each sample are the real nodes,
    #             # and any nodes after that are virtual.
    #             real_idx = idx[:n_real]
    #             virt_idx = idx[n_real:]
    #             # Update only the real nodes:
    #             # The update is analogous to: torch.cat((graph.x, pred), 1)[:, self.tw:]
    #             # but applied only to the real nodes.
    #             pred_sample = pred[graph.batch == b]  # shape should be (n_real, feat_dim)
    #             updated_real = torch.cat((graph.x[real_idx], pred_sample), 1)[:, self.tw:]
    #             # For virtual nodes, simply carry their state forward.
    #             updated_sample = torch.cat((updated_real, graph.x[virt_idx]), 0)
    #             new_x_list.append(updated_sample)
    #             new_batch_list.append(torch.full((updated_sample.size(0),), b, device=graph.x.device))
    #         graph.x = torch.cat(new_x_list, 0)
    #         graph.batch = torch.cat(new_batch_list, 0)
    #     else:
    #         # For other modes, update the entire state as before.
    #         graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]
        
    #     # Update labels and positions as before.
    #     t_new = torch.Tensor()
    #     y_new = torch.Tensor()
    #     for (labels_batch, step) in zip(labels, steps):
    #         y_tmp = torch.transpose(torch.cat([l.reshape(-1, nx * ny) for l in labels_batch]), 0, 1)
    #         y_new = torch.cat((y_new, y_tmp), )
    #         t_tmp = torch.ones(nx * ny) * t[step]
    #         t_new = torch.cat((t_new, t_tmp), )
        
    #     graph.y = y_new
    #     graph.pos[:, 0] = t_new

    #     return graph

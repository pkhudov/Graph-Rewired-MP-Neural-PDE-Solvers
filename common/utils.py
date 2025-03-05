import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from equations.PDEs import *
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils import coalesce, to_undirected
import random
import networkx

class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 5,
                 t_resolution: int = 250,
                 x_resolution: int = 100,
                 edge_prob: float = 0.0,
                 edge_path: str = None,
                 edge_mode: str = 'Radiusonly',
                 rand_edges_per_node: int = 2) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution
        self.edge_prob = edge_prob
        self.edge_path = edge_path
        self.edge_mode = edge_mode.lower()
        self.rand_edges_per_node = rand_edges_per_node
        self.custom_edge_index = None

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def save_edge_index(self, path):
        """Save the custom edge index to a file."""
        torch.save(self.custom_edge_index, path)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)

        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), )
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            x_pos = torch.cat((x_pos, x[0]), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
            batch = torch.cat((batch, torch.ones(nx) * b), )

        # Calculate the edge_index
        if f'{self.pde}' == 'CE':
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            local_edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        elif f'{self.pde}' == 'WE':
            local_edge_index = knn_graph(x_pos, k=self.n, batch=batch.long(), loop=False)

        # Load custom edge index if provided
        if self.edge_path is not None:
            self.custom_edge_index = torch.load(self.edge_path)

        # Generate custom edges if not preloaded and mode is not 'default'
        if self.custom_edge_index is None and self.edge_mode != 'radiusonly':
            batch_size = int(batch.max()) + 1
            all_custom_edges = []
            n_nodes = self.x_res

            for sample in range(batch_size):
                offset = sample * n_nodes

                if self.edge_mode == 'erdosrenyi':
                    custom_edges = erdos_renyi_graph(n_nodes, self.edge_prob)
                elif self.edge_mode == 'augmentnode':
                    custom_edges = []
                    for node_i in range(n_nodes):
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
                    cayley_graph = networkx.read_edgelist('cayley_edges_100', nodetype=int)
                    custom_edges = torch.tensor(list(cayley_graph.edges)).t()
                else:
                    raise ValueError(f'Unknown edge mode: {self.edge_mode}')
                    
                custom_edges = to_undirected(custom_edges, num_nodes=n_nodes) + offset
                all_custom_edges.append(custom_edges)

            self.custom_edge_index = coalesce(torch.cat(all_custom_edges, dim=1))
            print('Generated custom edges')

        graph = Data(x=u, edge_index_local=local_edge_index, edge_index_custom=self.custom_edge_index)
        graph.y = y
        graph.pos = torch.cat((t_pos[:, None], x_pos[:, None]), 1)
        graph.batch = batch.long()

        # Equation specific parameters
        if f'{self.pde}' == 'CE':
            alpha, beta, gamma = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                alpha = torch.cat((alpha, torch.tensor([variables['alpha'][i]])[:, None]), )
                beta = torch.cat((beta, torch.tensor([variables['beta'][i]*(-1.)])[:, None]), )
                gamma = torch.cat((gamma, torch.tensor([variables['gamma'][i]])[:, None]), )

            graph.alpha = alpha
            graph.beta = beta
            graph.gamma = gamma

        elif f'{self.pde}' == 'WE':
            bc_left, bc_right, c = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                bc_left = torch.cat((bc_left, torch.tensor([variables['bc_left'][i]])[:, None]), )
                bc_right = torch.cat((bc_right, torch.tensor([variables['bc_right'][i]])[:, None]), )
                c = torch.cat((c, torch.tensor([variables['c'][i]])[:, None]), )

            graph.bc_left = bc_left
            graph.bc_right = bc_right
            graph.c = c

        return graph


    def create_next_graph(self,
                             graph: Data,
                             pred: torch.Tensor,
                             labels: torch.Tensor,
                             steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph


import torch
import sys
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
# from einops import rearrange

def gaussian_function(x, sigma, a=1.0):
    return a*torch.exp(-(x**2./sigma))

class FourierFeatures(nn.Module):
    def __init__(self, in_features, number_features, trainable=False, sigma=1.0):
        super(FourierFeatures, self).__init__()
        self.in_features = in_features
        self.number_features = number_features
        self.sigma = sigma
        self.out_features = number_features * 2

        B = torch.randn(in_features, number_features) * sigma

        self.B = nn.Parameter(B, requires_grad=trainable)

    def forward(self, x):
        x_proj = x @ self.B # [N, number_features]
        return torch.cat([torch.cos(2*torch.pi*x_proj), torch.sin(2*torch.pi*x_proj)], dim=-1) # [N, number_features * 2]
        # return torch.cat([x_proj, x_proj], dim=-1) # [N, number_features * 2]


class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer_FS_2D(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.

    out_features : int
        Dimensionality of output features.
    hidden_features : int
        Dimensionality of hidden features.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 time_window,
                 n_variables,
                 rff_message=None,
                 gaussian_sigma=0.0):

        super(GNN_Layer_FS_2D, self).__init__(node_dim=-2, aggr='mean')
        self.rff_message = rff_message
        self.rff_dim = 0 if rff_message is None else rff_message.out_features

        if gaussian_sigma != 0.0:
            self.gaussian_sigma = nn.Parameter(torch.tensor(gaussian_sigma), requires_grad=True)
            self.gaussian_coeff = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.gaussian_sigma = None
            self.gaussian_coeff = None

        mn1_input_dim = 2 * in_features + time_window + 2 + self.rff_dim + n_variables
        self.message_net_1 = nn.Sequential(nn.Linear(mn1_input_dim, hidden_features),
                                           nn.ReLU()
                                           )

        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                           nn.ReLU()
                                           )

        self.update_net_1 = nn.Sequential(nn.Linear(in_features + out_features + n_variables, hidden_features),
                                          nn.ReLU()
                                          )

        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          nn.ReLU()
                                          )

        self.norm = BatchNorm(hidden_features)

    def forward(self, x, u, pos_x, pos_y, variables, edge_index, batch):
        """ Propagate messages along edges """

        x = self.propagate(edge_index, x=x, u=u, pos_x=pos_x,
                           pos_y=pos_y, variables=variables)
        x = self.norm(x)

        return x

    def message(self, x_i, x_j, u_i, u_j, pos_x_i, pos_x_j, pos_y_i, pos_y_j, variables_i):

        """ Message update """
        dx = pos_x_i - pos_x_j
        dy = pos_y_i - pos_y_j
        rel_pos = torch.cat([dx, dy], dim=-1)

        if self.rff_message is not None:
            rel_pos_rff = self.rff_message(rel_pos)
            message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, dx, dy, rel_pos_rff, variables_i), dim=-1))
        else:
            message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, dx, dy, variables_i), dim=-1))

        message = self.message_net_2(message)

        if self.gaussian_sigma is not None:
            message = message * gaussian_function(torch.norm(rel_pos, p=2., dim=-1), sigma=self.gaussian_sigma, a=self.gaussian_coeff).unsqueeze(-1)

        return message

    def update(self, message, x, variables):
        """ Node update """

        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)

        return x + update


class NPDE_GNN_FS_2D(torch.nn.Module):

    def __init__(
            self,
            pde,
            time_window=10,
            hidden_features=128,
            hidden_layer=6,
            eq_variables={},
            random_ff=False,
            rff_number_features=8,
            rff_sigma=1.0,
            gaussian_sigma=0.0
    ):
        super(NPDE_GNN_FS_2D, self).__init__()

        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables
        if random_ff:
            self.rff_number_features = rff_number_features
            self.rff_node = FourierFeatures(3, rff_number_features, sigma=rff_sigma, trainable=True)
            self.rff_message = FourierFeatures(2, rff_number_features, sigma=rff_sigma, trainable=True)
        else:
            self.rff_number_features = 0
            self.rff_node = None
            self.rff_message = None


        # in_features have to be of the same size as out_features for the time being

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer_FS_2D(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            # variables = eq_variables + time
            n_variables=len(self.eq_variables) + 1,
            rff_message=self.rff_message,
            gaussian_sigma=gaussian_sigma
        ) for _ in range(self.hidden_layer)))

        embedding_input_dim = self.time_window + 3 + (2 * self.rff_number_features) + len(self.eq_variables)
        self.embedding_mlp = nn.Sequential(
            nn.Linear(embedding_input_dim, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features)
            # Swish()
        )

        self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=5),
                                        # nn.BatchNorm1d(8),
                                        nn.ReLU(),
                                        nn.Conv1d(8, 1, 14, stride=2)

                                        )
    def __repr__(self):
        return f'GNN'

    def forward(self, data):

        u = data.x
        pos = data.pos
        pos_x = pos[:, 1][:, None]/self.pde.Lx
        pos_y = pos[:, 2][:, None]/self.pde.Ly
        pos_t = pos[:, 0][:, None]/self.pde.tmax
        edge_index = data.edge_index

        batch = data.batch
        variables = pos_t    # we put the time as equation variable

        if self.rff_node:
            coord_rff = self.rff_node(torch.cat([pos_x, pos_y, pos_t], dim=-1))
            node_input = torch.cat((u, pos_x, pos_y, coord_rff, variables), -1)
        else:
            node_input = torch.cat((u, pos_x, pos_y, variables), -1)

        h = self.embedding_mlp(node_input)

        for i in range(self.hidden_layer):
            if i % 2 == 0:
                current_edge_index = data.edge_index_local
            else:
                current_edge_index = data.edge_index_custom

            h = self.gnn_layers[i](
                h, u, pos_x, pos_y, variables, current_edge_index, batch)

        diff = self.output_mlp(h[:, None]).squeeze(1)
        dt = (torch.ones(1, self.time_window) * self.pde.dt * 0.1).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out

import torch
from common.data_creator import HDF5Dataset_FS_2D, GraphCreator_FS_2D
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from experiments.gnn_2d import NPDE_GNN_FS_2D



model_path = 'models/GNN_FS_FS_resolution32_n8_tw5_unrolling2_time1210175.pt'
data_path = 'data/not_downsampled/fs_2d_pde_32_train_dataset.h5'
neighbors = 4
batch_size = 1
nr_gt_steps = 2
nx_base_resolution = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator_FS_2D,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    trajectory = []
    for u in loader:
        u = u.float()
        losses_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u, same_steps)
            graph = graph_creator.create_graph(data, labels, same_steps).to(device)
            pred = model(graph)
            trajectory.append(pred)
            loss = criterion(pred, graph.y) / nx_base_resolution
            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u, same_steps)
                graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                pred = model(graph)
                trajectory.append(pred)
                loss = criterion(pred, graph.y) / nx_base_resolution
                losses_tmp.append(loss / batch_size)
        losses.append(torch.sum(torch.stack(losses_tmp)))
        break

    losses = torch.stack(losses)
    trajectory = torch.stack(trajectory)

    trajectory = trajectory.permute(0, 2, 1).reshape(90, 32, 32)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    return losses, trajectory


pde = SimpleNamespace(Lx=32, Ly=32, dt=1.0, grid_size=(100,32,32), tmin=0.0, tmax=100.0)
eq_variables={}
criterion = torch.nn.MSELoss(reduction="sum")
graph_creator = GraphCreator_FS_2D(pde=pde,
                                 neighbors=neighbors,
                                 time_window=5,
                                 x_resolution=32,
                                 y_resolution=32).to(device)
model = NPDE_GNN_FS_2D(pde=pde,
                        time_window=graph_creator.tw,
                        eq_variables=eq_variables).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

dataset = HDF5Dataset_FS_2D(data_path, mode='train', super_resolution=(100,32,32))
loader = DataLoader(dataset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=0)

_, traj = test_unrolled_losses(model=model,
                        steps=[0],
                        batch_size=batch_size,
                        nr_gt_steps=nr_gt_steps,
                        nx_base_resolution=nx_base_resolution,
                        loader=loader,
                        graph_creator=graph_creator,
                        criterion=criterion,
                        device=device)

fig, axes = plt.subplots(1, 1, figsize=(12, 9))
traj = traj.unsqueeze(0)
axes = [axes]
# axes = axes.flatten()

print('Largest Value: ', traj.max().item())
caxs = []
for i, ax in enumerate(axes):
    cax = ax.imshow(traj[i][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=traj.max())
    caxs.append(cax)
    ax.set_title(f"Sample {i+1}")
    fig.colorbar(cax, ax=ax, label='Smoke Intensity')

plt.tight_layout(rect=[0, 0, 1, 0.95])

suptitle = fig.suptitle(f"Time Step: 1", fontsize=16)

def update(frame):
    for i, cax in enumerate(caxs):
        suptitle.set_text(f"Time Step: {frame+1}")
        cax.set_array(traj[i][frame].numpy()) 
    return caxs

anim = FuncAnimation(fig, update, frames=traj.shape[1], interval=100, blit=True)


plt.show()

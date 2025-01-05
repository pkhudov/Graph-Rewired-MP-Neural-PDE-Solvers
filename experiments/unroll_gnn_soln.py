import torch
from common.data_creator import HDF5Dataset_FS_2D, GraphCreator_FS_2D
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from experiments.gnn_2d import NPDE_GNN_FS_2D

mode = 'test'
sample_no = 5


model_path = 'models/GNN_FS_resolution32_n4_tw5_unrolling2_time121734.pt'
data_path = f'data/fs_2d_pde_128_{mode}_dataset.h5'
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
    n=0
    for u in loader:
        trajectory = []
        true_traj = []
        u = u.float()
        losses_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u, same_steps)
            graph = graph_creator.create_graph(data, labels, same_steps).to(device)
            pred = model(graph)
            trajectory.append(pred)
            true_traj.append(graph.y)
            loss = criterion(pred, graph.y) / nx_base_resolution
            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u, same_steps)
                graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                pred = model(graph)
                trajectory.append(pred)
                true_traj.append(graph.y)
                loss = criterion(pred, graph.y) / nx_base_resolution
                losses_tmp.append(loss / batch_size)
        losses.append(torch.sum(torch.stack(losses_tmp)))

        if n == sample_no:
            break
        n+=1

    losses = torch.stack(losses)
    trajectory = torch.stack(trajectory)
    true_traj = torch.stack(true_traj)
    trajectory = trajectory.permute(0, 2, 1).reshape(90, 32, 32)
    true_traj = true_traj.permute(0, 2, 1).reshape(90, 32, 32)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    return losses, trajectory, true_traj


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

dataset = HDF5Dataset_FS_2D(data_path, mode=mode, super_resolution=(100,128,128), downsample=True)
loader = DataLoader(dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0)

_, traj, true_traj = test_unrolled_losses(model=model,
                        steps=[0],
                        batch_size=batch_size,
                        nr_gt_steps=nr_gt_steps,
                        nx_base_resolution=nx_base_resolution,
                        loader=loader,
                        graph_creator=graph_creator,
                        criterion=criterion,
                        device=device)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
traj = traj.unsqueeze(0)
true_traj = true_traj.unsqueeze(0)
gnn_ax = axes[0]
true_ax = axes[1]


print('Largest Value: ', traj.max().item())
caxs = []
cax_gnn = gnn_ax.imshow(traj[0][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=traj.max())
cax_true = true_ax.imshow(true_traj[0][0].numpy(), origin='lower', cmap='viridis', animated=True, vmin=0, vmax=traj.max())
caxs.append(cax_gnn)
caxs.append(cax_true)
gnn_ax.set_title('GNN')
true_ax.set_title('Solver')

plt.tight_layout(rect=[0, 0, 1, 0.95])

suptitle = fig.suptitle(f"Time Step: 1", fontsize=16)
def update(frame):
    suptitle.set_text(f"Time Step: {frame+1}")
    cax_gnn.set_array(traj[0][frame].numpy())
    cax_true.set_array(true_traj[0][frame].numpy())
    return caxs


anim = FuncAnimation(fig, update, frames=traj.shape[1], interval=100, blit=True)


plt.show()

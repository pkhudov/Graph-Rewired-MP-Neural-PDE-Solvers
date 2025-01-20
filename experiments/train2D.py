import argparse
import os
import copy
import sys
import time
from datetime import datetime, timedelta
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from types import SimpleNamespace
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.data_creator import HDF5Dataset_FS_2D, GraphCreator_FS_2D
from experiments.gnn_2d import NPDE_GNN_FS_2D
from experiments.models_cnn import BaseCNN
from experiments.train_helper2D import *
from equations.PDEs import *

def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'experiments/log'):
        os.mkdir(f'experiments/log')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')

def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator_FS_2D,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()

    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    for i in range(graph_creator.t_res):
        losses, batch_grads_mean = training_loop(model, unrolling, args.batch_size, optimizer, loader, graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}; Norm Grads: {batch_grads_mean}')

def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator_FS_2D,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    # next we test the unrolled losses
    losses = test_unrolled_losses(model=model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  nr_gt_steps=args.nr_gt_steps,
                                  nx_base_resolution=args.resolution[1],
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    return torch.mean(losses)


def main(args: argparse):

    device = args.device
    check_directory()

    # base_resolution = args.base_resolution
    # super_resolution = args.super_resolution

    # Load datasets
    train_string = f'data/fs_2d_pde_128_train_dataset.h5'
    valid_string = f'data/fs_2d_pde_128_valid_dataset.h5'
    test_string = f'data/fs_2d_pde_128_test_dataset.h5'
    try:
        train_dataset = HDF5Dataset_FS_2D(train_string, mode='train', downsample=True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1)

        valid_dataset = HDF5Dataset_FS_2D(valid_string, mode='valid', downsample=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=1)

        test_dataset = HDF5Dataset_FS_2D(test_string, mode='test', downsample=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=1)
    except:
        raise Exception("Datasets could not be loaded properly")


    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

    if(args.log):
        logfile = f'experiments/log/{args.model}_{args.experiment}_resolution{args.resolution[1]}_n{args.neighbors}_edgeprob{args.edge_prob}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}_rffs{args.fourier_features}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    save_path = f'models/GNN_{args.experiment}_resolution{args.resolution[1]}_n{args.neighbors}_edgeprob{args.edge_prob}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}_rffs{args.fourier_features}.pt'
    save_edges_path = f'models/edges/Edges_GNN_{args.experiment}_resolution{args.resolution[1]}_n{args.neighbors}_edgeprob{args.edge_prob}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}_rffs{args.fourier_features}.pt'
    print(f'Training on dataset {train_string}')
    print(device)
    print(save_path)
    

    pde = SimpleNamespace(Lx=args.resolution[1], Ly=args.resolution[2], dt=1.0, grid_size=tuple(args.resolution), tmin=0.0, tmax=100.0)

    eq_variables={}

    graph_creator = GraphCreator_FS_2D(pde=pde,
                                 neighbors=args.neighbors,
                                 time_window=args.time_window,
                                 x_resolution=args.resolution[1],
                                 y_resolution=args.resolution[2],
                                 edge_prob=args.edge_prob).to(device)

    if args.model == 'GNN':
        model = NPDE_GNN_FS_2D(pde=pde,
                              time_window=graph_creator.tw,
                              eq_variables=eq_variables,
                              random_ff=args.fourier_features).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_decay)

    # Training loop
    min_val_loss = 10e30
    test_loss = 10e30
    criterion = torch.nn.MSELoss(reduction="sum")
    total_train_time = timedelta()
    best_epoch = 0
    time_upto_best_epoch = timedelta()
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train_start_time = datetime.now()
        train(args, pde, epoch, model, optimizer, train_loader, graph_creator, criterion, device=device)
        train_end_time = datetime.now()
        total_train_time += train_end_time - train_start_time
        print("Evaluation on validation dataset:")
        val_loss = test(args, pde, model, valid_loader, graph_creator, criterion, device=device)
        if(val_loss < min_val_loss):
            print("Evaluation on test dataset:")
            test_loss = test(args, pde, model, test_loader, graph_creator, criterion, device=device)
            # Save model
            torch.save(model.state_dict(), save_path)
            if args.edge_prob > 0.0:
                graph_creator.save_edge_index(save_edges_path)
            print(f"Saved model at {save_path}\n")
            print("Training time: ", total_train_time)
            time_upto_best_epoch = total_train_time
            best_epoch = epoch
            min_val_loss = val_loss
        scheduler.step()

    print(f"Test loss: {test_loss}")
    print(f"Training time (until epoch {best_epoch}): ", {time_upto_best_epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--experiment', type=str, default='FS',
                        help='Experiment for PDE solver should be trained: FS')

    # Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')
    
    # Graph construction
    parser.add_argument('--neighbors', type=int,
                        default=8, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--edge_prob', type=float,
                        default=0.0, help="Probability with which an edge is added to the graph according to Erdos-Renyi model")

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=4,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=1,
            help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')
    parser.add_argument('--parameter_ablation', type=eval, default=False,
                        help='Flag for ablating MP-PDE solver without equation specific parameters')
    parser.add_argument('--resolution', type=int, nargs=3, default=(100, 32, 32), help='Downsampled resolution nt nx ny')

    parser.add_argument('--time_window', type=int,
                        default=5, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=2, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--nr_gt_steps', type=int,
                        default=2, help="Number of steps done by numerical solver")
    parser.add_argument('--fourier_features', type=eval, default=False, help='Whether to use fourier features')

    # Misc
    parser.add_argument('--print_interval', type=int, default=10,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')

    args = parser.parse_args()
    main(args)

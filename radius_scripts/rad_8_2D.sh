#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 48 hours:
#$ -l h_rt=47:59:59
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#$ -e /exports/eddie/scratch/s2242913/Gnn_Pavel/Graph-Rewired-MP-Neural-PDE-Solvers/eddie_outputs/
#$ -o /exports/eddie/scratch/s2242913/Gnn_Pavel/Graph-Rewired-MP-Neural-PDE-Solvers/eddie_outputs/
#
# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#
# Request 32 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=32G
# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda/12.1

# Activate the conda environment
module load anaconda
source activate graph-mp-pde

# Experiment E1: Burger's equation (0.5, 0, 0)
# 2096 trajectories
# Base Resoultion: 100
# Super Resolution: 200
# Temporal Resolution: 250
# Learning Rate: 1e-4
# Weight Decay: 1e-8
# Epochs: 20
# Batch Size: 16
# Hidden size: 164
# Maximum unrolling: 2
# Neighbours: 6

python experiments/train2D.py --device=cuda:0 --neighbors=8 --resolution=32 --batch_size=4 --lr=1e-4 --unrolling=2 --num_epochs=25 --log=True

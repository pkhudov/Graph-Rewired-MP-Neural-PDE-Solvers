#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 4 hours:
#$ -l h_rt=7:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
#
#
#
#
# Request 20 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=16G

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda/12.1

# Activate the conda environment
module load anaconda
source activate graph-mp-pde

python generate/generate_smoke.py

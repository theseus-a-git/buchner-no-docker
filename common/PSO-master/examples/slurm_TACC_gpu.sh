#!/bin/bash

# For fitness functions that are CUDA-compatible.
# MPI distributes independent PSO runs.
# Each MPI task receives one GPU.
# CUDA parallelizes the particle fitness
# evaluations within that PSO run.

# One MPI task performs one PSO run.
# Request one GPU per MPI task.
#
# gpu-a100 : 3 GPUs/node
# gpu-h100 : 2 GPUs/node
#
# Example:
# 6 runs
#   gpu-a100 -> 2 nodes
#   gpu-h100 -> 3 nodes
#
# NOTE: Lonestar6 does not support --gpus-per-task

#SBATCH --partition=gpu-a100
#SBATCH --nodes=2           
#SBATCH --ntasks=6

#SBATCH --job-name=PSO_example_job_gpu
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j_%t_out.txt
#SBATCH --error=%x_%j_%t_err.txt

# NOTE: default python on TACC is 3.9.7
ml python/3.12

# NOTE: mpirun is a python wrapper for mpi and requires mpi 
#   to be loaded first. On Lonestar6, MPI is provided by 
#   Intel MPI and is already loaded.

# NOTE: if using gpus, cuda must be loaded first.
ml cuda/12

# Launch the py file
ibrun python ./PSO_Launcher_cc.py

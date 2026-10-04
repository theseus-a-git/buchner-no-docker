#!/bin/bash
#SBATCH --job-name=PSO_example_job
#SBATCH --nodes=3           # On CRADLE use number of runs
#SBATCH --ntasks=3          # On CRADLE 1 task per node
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j_%t_out.txt
#SBATCH --error=%x_%j_%t_err.txt
#SBATCH --partition=gpuq

# If the fitness function is not CUDA-compatible, it must 
#   be run on cpu cores. If the fitness function is 
#   CUDA-compatible, request gpus.
#   ONLY CHOOSE ONE OF THESE OPTIONS BELOW:

#SBATCH --cpus-per-task=40  # choose number of particles
# #SBATCH --gpus-per-task=1   # choose number of runs

# NOTE: mpirun is a python wrapper for mpi and requires mpi 
#   to be loaded first.
ml mpi

# NOTE: if using gpus, cuda must be loaded first.
# ml cuda/12.3

# Launch the .py file
mpirun python3 ./PSO_Launcher_cc.py

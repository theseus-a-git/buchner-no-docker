#!/bin/bash

# For fitness functions that are not CUDA-compatible.
# MPI parallelizes independent PSO runs.
# CPU cores within each task are used for
# fitness_parallel_mode='cpu'.


# One MPI task performs one PSO run.
# The normal partition allows 2 tasks per node.
# Thus request ceil(ntasks/2) nodes 

#SBATCH --partition=normal
#SBATCH --nodes=3           
#SBATCH --ntasks=6          # choose number of runs
#SBATCH --cpus-per-task=40  # choose number of particles

#SBATCH --job-name=PSO_example_job_cpu
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j_%t_out.txt
#SBATCH --error=%x_%j_%t_err.txt

# NOTE: default python on TACC is 3.9.7
ml python/3.12

# NOTE: mpirun is a python wrapper for mpi and requires mpi 
#   to be loaded first. On Lonestar6, MPI is provided by 
#   Intel MPI and is already loaded.

# Launch the .py file
ibrun python ./PSO_Launcher_cc.py


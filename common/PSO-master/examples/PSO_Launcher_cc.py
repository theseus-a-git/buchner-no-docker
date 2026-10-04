# -*- coding: utf-8 -*-
"""
@file PSO_Launcher_cc.py

@brief This is an example for calling run_PSO on a computing cluser. By default
fitness_func is assumed to not be CUDA-compatible and is run on cpu cores. If 
the fitness function supports CuPy arrays and the computing cluser has 
CUDA-capable gpus, then you can use 
    fitness_function = Rastrigin_cupy
and set 
    fitness_parallel_mode='cuda-cupy'

@author
Ben Carlson

@date
2026-07-02
"""
from numpy import array

# Add path to PSO_main.py so run_PSO can be imported.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import run_PSO
from PSO_main import run_PSO

# Import sample benchmark functions
from benchmarks.benchmark_functions import Rastrigin
from benchmarks.benchmark_functions import Griewank
from benchmarks.benchmark_functions import Rastrigin_cupy
from benchmarks.benchmark_functions import Griewank_cupy


'''*************************PARAMETERS**************************************'''
'''*************************************************************************'''
# Number of PSO runs
N_runs = 6
# Number of iterations per run
N_iter = 5000
# coordinate boundary conditions
boundary_conditions = array([[-5.12, 5.12], # bounds on x1
                              [-5.12, 5.12], # bounds on x2
                            [-5.12, 5.12]]) # bounds on x3
fitness_function = Rastrigin
# fitness_function = Rastrigin_cupy

'''******************************RUNNING PSO********************************'''
'''*************************************************************************'''
def main():
    
    # Run PSO
    (fitness_vals, 
     best_locations, 
     _, _, _) = run_PSO(N_runs, 
                        N_iter, 
                        boundary_conditions,
                        fitness_function,
                        run_parallel_mode='mpi',
                        fitness_parallel_mode='cpu',
                        seed_vals=True)
  
    print("\nBest fitness vals: ")
    print(fitness_vals)
    print("\nBest Locations: ")
    print(best_locations)


if __name__ == '__main__':
    main()  

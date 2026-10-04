# -*- coding: utf-8 -*-
"""
@file PSO_main.py

@brief Contains run_PSO which calls the Particle Swarm Optimization algorithm
and returns the results.

@author
Ben Carlson

@date
2026-05-19
"""
from numpy import ndarray, array, float64, argmin
from numpy.typing import NDArray
import datetime

from utils.config import PSOConfig, print_config
from utils.input_validation import check_fitness_func
from core.coordinate_standardization import un_standardize
from parallel_wrappers.wrap_perform_pso_run import perform_run_wrapper

def run_PSO(N_runs: int, 
            N_iterations: int,
            boundary_conditions: ndarray,
            fitness_func: callable,
            **kwargs
            ) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    @brief Main function that calls the Particle Swarm Optimization algorithm
    and returns the results.
    
    @param N_runs
    The desired number of PSO runs to perform.
    
    @param N_iterations
    The number of iterations to perform in each PSO run.
    
    @param boundary_conditions
        NumPy array of shape (N, 2) containing coordinate bounds for the search
        space dimensions. Each row must contain:
        - boundary_conditions[i, 0] : lower bound
        - boundary_conditions[i, 1] : upper bound
        
    @param fitness_func
    User-supplied function to optimize over. IMPORTANT: for parallelization 
    modes, this function must be defined at module level and be pickleable.

    Series/CPU modes expect:

    @code
    fitness_func(x) -> float
    @endcode

    where

    @code
    x.shape == (N_x,)
    @endcode

    GPU modes expect a batched fitness function:

    @code
    fitness_func(X) -> fitness_vals
    @endcode

    where

    @code
    X.shape == (N_particles, N_x)
    fitness_vals.shape == (N_particles,)
    @endcode
    
    @return best_fitness_vals
    Array containing the best fitness value found in each PSO run.
    
    @return best_physical_locations
    Array containing the locations in parameter space corresponding to the 
    best fitness values. These are returned in physical coordinates.
    
    @return particle_history
    By default this is an empty array. If save_history=True, this array
    contains the position and velocity history of each particle as they 
    explored the parameter space. These are returned in standardized 
    coordinates.
    
    @return fitness_history
    By default this is an empty array. If save_history=True, this array
    contains the fitness history of each particle as they explored the 
    parameter space.
    
    @return pbest_history
    By default this is an empty array. If save_history=True, this array
    contains the pbest history of each particle as they explored the 
    parameter space. These are returned in standardized coordinates.
    """
    # Create configuration parameters and validate inputs.
    core_fields = {
        "N_runs": N_runs,
        "N_iterations": N_iterations,
        "boundary_conditions": boundary_conditions,
    }
    # prevent override attempts
    for k in core_fields:
        if k in kwargs:
            raise TypeError(f"{k} is a required argument of run_PSO and "
                            "cannot be set via kwargs")
    config = PSOConfig( **core_fields, **kwargs )
    check_fitness_func(fitness_func, config)
    
    # Print summary of settings. MPI check to avoid multiple printings
    is_root = True
    if config.run_parallel_mode == "mpi":
        from mpi4py import MPI
        is_root = (MPI.COMM_WORLD.Get_rank() == 0)
    if config.verbose and is_root:
        print(f'\n\t run_PSO began running on {datetime.datetime.now()}')
        print("Configuration parameters:\n")
        print_config(config)
        print()
        
    # Perform the PSO algorithm by wrapping the PSO runs in a parallel wrapper
    (best_fitness_vals, 
     best_std_locations,
     particle_history,
     fitness_history,
     pbest_history      ) = perform_run_wrapper(fitness_func, config)
                                            
    # un-standardize locations
    best_physical_locations = un_standardize(array(best_std_locations),
                                             config.boundary_conditions)
    best_fitness_vals = array(best_fitness_vals)
    
    # Print results                                           
    if config.verbose:
        final_fitness_val = min(best_fitness_vals)
        final_coords = best_physical_locations[argmin(best_fitness_vals)]
        print('\n\tFINAL RESULTS ----------')
        print(f'\nbest function value = {final_fitness_val}')
        print(f'best coordinates = {final_coords}')

    return (best_fitness_vals, 
            best_physical_locations, 
            particle_history, 
            fitness_history, 
            pbest_history)

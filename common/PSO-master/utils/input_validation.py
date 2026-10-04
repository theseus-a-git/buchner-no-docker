# -*- coding: utf-8 -*-
"""
@file input_validation.py

@brief Function for validating user user-supplied fitness function for PSO.

@note config parameter validation occurs inside the PSOConfig dataclass.

@author
Ben Carlson

@date
2026-05-08
"""
import inspect
import pickle

from utils.config import PSOConfig

def check_fitness_func(fitness_func: callable, config: PSOConfig):
    """
    @brief Function to check the user-supplied fitness function so the program 
    doesn't crash.
    
    @param fitness_func
    The fitness function that PSO will use to estimate parameters.
    
    @param config
    PSOConfig dataclass of PSO parameters.
    
    @return void
    
    @throws TypeError
    Raised fitness_func is invalid.
    
    @note
    parallel modes 'cpu' and 'mpi' require the fitness_func to be pickleable 
    while 'series' execution allows nested functions and lambdas.
    """
    
    # --------------------------------------------------------
    # CHECK fitness_func is callable
    # --------------------------------------------------------
    
    if not callable(fitness_func):
        raise TypeError("fitness_func must be a callable function.")
        
    # --------------------------------------------------------
    # CHECK fitness_func is pickleable
    # --------------------------------------------------------
    
    parallel_requires_pickling = (
        config.run_parallel_mode in {"cpu", "mpi"} or
        config.fitness_parallel_mode == "cpu"
    )
    
    if parallel_requires_pickling:
        try:
            pickle.dumps(fitness_func)
        except Exception as e:
            raise TypeError(
                "fitness_func must be pickleable for CPU or MPI parallel execution."
            ) from e
    
        if inspect.isfunction(fitness_func):
            if fitness_func.__qualname__ != fitness_func.__name__:
                raise TypeError(
                    "fitness_func must be defined at module scope "
                    "when using CPU or MPI parallel execution."
                )
    
    
    return

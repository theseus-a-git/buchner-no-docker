# -*- coding: utf-8 -*-
"""
@file wrap_fitness_func.py

@brief Program to call the fitness function for each particle to obtain the new 
fitness values for the new locations.

@warning
CPU and GPU fitness backends use different objective-function interfaces.

CPU modes expect scalar evaluations:

@code
fitness_func(x) -> float
@endcode

GPU modes expect batched evaluations:

@code
fitness_func(X) -> ndarray
@endcode

where multiple particles are evaluated simultaneously.

@author
Ben Carlson

@date
2026-05-18
"""
from numpy import array, ndarray, empty
# import cupy as cp # gpu import given in the branches for compatibility
# import torch

from utils.config import PSOConfig
from core.coordinate_standardization import un_standardize

def make_get_fitness_wrapper(config: PSOConfig, 
                             fitness_func: callable,
                             gpu_id=None,
                             executor=None
                             ) -> callable:
    """
    @brief Factory function that creates a fitness-evaluation wrapper.

    Returns a callable that evaluates particle fitness values according to
    the selected parallelization backend specified by

    @code
    config.fitness_parallel_mode
    @endcode

    Supported modes include:

        - "series"
        - "cpu"
        - "cuda-cupy"
        - "cuda-torch"

    The returned wrapper always presents a uniform interface to the PSO
    implementation regardless of the underlying execution backend.

    Expected wrapper signature:

    @code
    get_fitness_wrapper(particles) -> ndarray
    @endcode

    where

    @code
    particles.shape == (N_particles, 1, N_x)
    @endcode

    and the returned array satisfies

    @code
    fitness_vals.shape == (N_particles,)
    @endcode

    @param config
    PSO configuration object.

    @param fitness_func
    User-supplied objective function.

    CPU modes expect:

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

    @param gpu_id
    None or int. Determines which gpu to assign to the returned function.
    
    This argument is only used when:

    @code
    fitness_parallel_mode == "cuda-cupy"
    fitness_parallel_mode == "cuda-torch"
    @endcode

    @param executor
    Existing ProcessPoolExecutor used for CPU parallel fitness evaluations.

    This argument is only used when:

    @code
    fitness_parallel_mode == "cpu"
    @endcode

    Passing an existing executor avoids repeatedly destroying and recreating
    worker processes at every PSO iteration.

    @return
    Callable fitness-evaluation wrapper.

    @throws NotImplementedError
    Raised if the selected fitness_parallel_mode is not supported.
    """
    # unpack config
    fitness_mode = config.fitness_parallel_mode
    boundary_conditions = config.boundary_conditions
    
    if fitness_mode == "series":
        
        def get_fitness_wrapper_series(particles: ndarray) -> ndarray:
            """
            @brief Evaluate particle fitness values sequentially.
        
            Converts all particle coordinates from standardized space to 
            physical coordinates and evaluates the fitness function one 
            particle at a time.
        
            @param particles
            Particle array.
        
            Shape:
        
            @code
            (N_particles, 1, N_x)
            @endcode
        
            @return
            Fitness values.
        
            Shape:
        
            @code
            (N_particles,)
            @endcode
        
            @note
            Particle coordinates are first converted from standardized 
            coordinates to physical coordinates using
        
            @code
            un_standardize()
            @endcode
        
            before being passed to the user fitness function.
        
            @note
            The user fitness function receives one particle at a time.
        
            Expected signature:
        
            @code
            fitness_func(x) -> float
            @endcode
            """
            # unstandardize 
            all_x_vals = un_standardize(particles[:,0,:], boundary_conditions)
            # Perform fitness evaluations in series
            new_fitness_vals = empty((len(particles),))
            for j in range(len(particles)):
                new_fitness_vals[j] = fitness_func( all_x_vals[j] )
            return array(new_fitness_vals)
        
        return get_fitness_wrapper_series       
    
    
    elif fitness_mode == "cpu":
        
        def get_fitness_wrapper_cpu(particles: ndarray) -> ndarray:
            """
            @brief Evaluate particle fitness values in parallel using
            ProcessPoolExecutor.
        
            Particle coordinates are converted to physical coordinates and then
            distributed across worker processes for concurrent evaluation.
        
            @param particles
            Particle array.
        
            Shape:
        
            @code
            (N_particles, 1, N_x)
            @endcode
        
            @return
            Fitness values.
        
            Shape:
        
            @code
            (N_particles,)
            @endcode
        
            @note
            Result ordering is guaranteed to match particle ordering.
        
            @note
            Worker processes receive serialized copies of particle coordinates.
            Modifications performed by the user fitness function cannot affect
            the PSO state in the parent process.
            """
            # Set up parameters to pass to the parallel fitness calls
            all_x_vals = un_standardize(particles[:,0,:], boundary_conditions)
            # Perform fitness evaluations in parallel
            results = list( executor.map( fitness_func, all_x_vals ) )
            # Return results
            return array(results)
            
        return get_fitness_wrapper_cpu
    
    
    elif fitness_mode == "cuda-cupy":
        import cupy as cp
        if gpu_id is not None:
            cp.cuda.Device(gpu_id).use()
        else:
            cp.cuda.Device(0).use()
            

        def get_fitness_wrapper_cuda_cupy(particles: ndarray) -> ndarray:
            """
            @brief Evaluate particle fitness values on an NVIDIA GPU using CuPy.
        
            All particles are converted to physical coordinates and transferred
            to GPU memory in a single batch.
        
            @param particles
            Particle array.
        
            Shape:
        
            @code
            (N_particles, 1, N_x)
            @endcode
        
            @return
            Fitness values.
        
            Shape:
        
            @code
            (N_particles,)
            @endcode
        
            Returned as a NumPy array on the host.
        
            @note
            The fitness function must support batched CuPy input.
        
            Expected signature:
        
            @code
            fitness_func(X) -> fitness_vals
            @endcode
        
            where
        
            @code
            X.shape == (N_particles, N_x)
            fitness_vals.shape == (N_particles,)
            @endcode
        
            @note
            Double precision is used:
        
            @code
            cp.float64
            @endcode
        
            to maintain consistency with the CPU implementation.
            """
            # unstandardize 
            X_phys = un_standardize(particles[:, 0, :], boundary_conditions)
            X_phys = cp.asarray(X_phys, dtype=cp.float64)
    
            # ---- GPU CALL ----
            # fitness_func must accept (N_particles, N_x)
            results = fitness_func(X_phys)
    
            return cp.asnumpy(results)
        
        return get_fitness_wrapper_cuda_cupy
    
    
    elif fitness_mode == "cuda-torch":
        import torch
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        
        def get_fitness_wrapper_cuda_torch(particles: ndarray) -> ndarray:
            """
            @brief Evaluate particle fitness values on an NVIDIA GPU using 
            PyTorch.
        
            All particles are converted to physical coordinates and transferred
            to a CUDA tensor in a single batch.
        
            @param particles
            Particle array.
        
            Shape:
        
            @code
            (N_particles, 1, N_x)
            @endcode
        
            @return
            Fitness values.
        
            Shape:
        
            @code
            (N_particles,)
            @endcode
        
            Returned as a NumPy array on the host.
        
            @note
            The fitness function must support batched CUDA tensors.
        
            Expected signature:
        
            @code
            fitness_func(X) -> fitness_vals
            @endcode
        
            where
        
            @code
            X.shape == (N_particles, N_x)
            fitness_vals.shape == (N_particles,)
            @endcode
        
            @note
            Double precision is used:
        
            @code
            torch.float64
            @endcode
        
            to maintain consistency with the CPU implementation.
            """
            # unstandardize
            X_phys = un_standardize(particles[:, 0, :], boundary_conditions)
            X_phys = torch.as_tensor(X_phys, device="cuda", dtype=torch.float64)
    
            # ---- GPU CALL ----
            # fitness_func must accept (N_particles, N_x)
            results = fitness_func(X_phys)
    
            return results.detach().cpu().numpy()
        
        return get_fitness_wrapper_cuda_torch
        
    
    elif fitness_mode == "cuda-numba":
        raise NotImplementedError()
    
    elif fitness_mode == "gpu":
        raise NotImplementedError()
    
    else:
        raise NotImplementedError()

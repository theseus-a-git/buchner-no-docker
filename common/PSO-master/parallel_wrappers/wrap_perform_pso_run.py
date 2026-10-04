# -*- coding: utf-8 -*-
"""
@file wrap_perform_pso_run.py

@brief Execute multiple PSO runs either serially or in parallel.

This module provides a wrapper around `perform_run()` that manages
execution of multiple independent PSO runs and a second wrapper that
houses the executor that perform_run() uses for the fitness evaluations.

Depending on the configuration, runs may be executed:

    - sequentially in a single process
    - concurrently across multiple CPU processes

Parallel execution uses Python's
`concurrent.futures.ProcessPoolExecutor`
or
`mpi4py.MPI`

The ordering of returned results is guaranteed to match the ordering
of the requested runs regardless of execution order when running in
parallel.

Each run is independent and receives its own seed value and run index.

@author
Ben Carlson

@date
2026-05-18
"""
from numpy import array, float64, random
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor
#from mpi4py import MPI # import given in mpi branch for compatibility

from utils.config import PSOConfig
from parallel_wrappers.wrap_fitness_func import make_get_fitness_wrapper
from core.perform_pso_run import perform_run


        
def _perform_run_wrapper2(run_val: int,
                          fitness_func: callable,
                          config: PSOConfig
                          ):
    """
    @brief Second wrapper for `perform_run()`. This holds the executor for the 
    fitness evaluations so the pool is not destroyed and recreated each 
    iteration. It also checks gpu availability and generates gpu_id.
    
    @param run_val
    The index of this run. Lies within [0, ..., N_runs-1]
    
    @param fitness_func
    User-supplied objective function.
    
    @param config
    PSO configuration object.
    
    @return
    The return value produced by `perform_run()`
    """
    # Unpack config
    run_mode = config.run_parallel_mode
    fitness_mode = config.fitness_parallel_mode
    fitness_n_jobs = config.fitness_n_jobs

    # Get gpu ids if available
    gpu_id = None 
    if fitness_mode in ("cuda-torch", "cuda-cupy"):
        import cupy as cp
        if run_mode == "cpu":
            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus > 0:
                # Slurm-safe indexing
                gpu_id = run_val % n_gpus
            

    # Create fitness wrapper function
    if fitness_mode != "cpu":
        get_fitness_wrapper = make_get_fitness_wrapper(config, 
                                                       fitness_func,
                                                       gpu_id=gpu_id)
        return perform_run(run_val,
                           config,
                           get_fitness_wrapper)
    
    elif fitness_mode == "cpu":
        with ProcessPoolExecutor(max_workers=fitness_n_jobs) as fit_executor:
            get_fitness_wrapper = make_get_fitness_wrapper(
                                                config,
                                                fitness_func,
                                                executor=fit_executor
                                            )
            return perform_run(run_val,
                               config,
                               get_fitness_wrapper)
           
    

def _run_single(args):
    """
    @brief Helper function used by ProcessPoolExecutor.

    Unpacks a tuple of arguments and forwards them to
    `perform_run()`.

    This helper exists because `executor.map()` supplies a single
    argument to the mapped function, whereas `perform_run()`
    expects multiple positional arguments.

    @param args
    Tuple containing the positional arguments required by
    `perform_run()`.

    @return
    The return value produced by `perform_run()` wrapped by 
    `_perform_run_wrapper2`.
    """
    return _perform_run_wrapper2(*args)


def perform_run_wrapper(fitness_func: callable,
                        config: PSOConfig
                        ) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    @brief Execute multiple PSO runs and collect results.

    Calls `_perform_run_wrapper2` repeatedly according to the selected
    parallelization mode.

    When:

        - `config.run_parallel_mode == 'series'`

          Runs are executed sequentially in the current process.

        - `config.run_parallel_mode == 'cpu'`

          Runs are distributed across multiple worker processes
          using `ProcessPoolExecutor`.
          
        - `config.run_parallel_mode == 'mpi'`

          Runs are distributed across multiple worker processes
          using `mpi4py` which allows memory sharing across nodes.

    Returned result ordering always corresponds to run index order,
    even when runs complete out of order in parallel mode.

    @param fitness_func
    User-supplied objective function.

    Expected signature:

    @code
    fitness_func(x) -> float
    @endcode

    where:

    @code
    x.shape == (N_x,)
    @endcode

    @param config
    PSO configuration object.

    Controls:

        - swarm parameters
        - random-number generation
        - logging
        - history storage
        - run-level parallelization

    @return
    Tuple containing:

    @code
    (
        all_fs,
        all_gbests,
        hist_p,
        hist_f,
        hist_pb
    )
    @endcode

    where:

        - all_fs

          Best fitness value from each run.

          Shape:

          @code
          (N_runs,)
          @endcode

        - all_gbests

          Best solution location from each run.

          Shape:

          @code
          (N_runs, N_x)
          @endcode

        - hist_p

          Particle-history arrays from each run.

        - hist_f

          Fitness-history arrays from each run.

        - hist_pb

          Personal-best-history arrays from each run.

    @note
    Parallel execution with concurrent.futures is implemented using separate 
    processes rather than threads in order to avoid limitations imposed by
    Python's Global Interpreter Lock (GIL).
    
    @note
    When run_parallel_mode == "mpi", non-root MPI ranks terminate
    after their local work has been gathered. Only rank 0 returns
    results to the caller.

    @note
    Worker count is controlled by:

    @code
    config.run_n_jobs
    @endcode

    @note
    History arrays may be empty when:

    @code
    config.save_history == False
    @endcode

    @throws NotImplementedError
    Raised if the selected run parallelization mode is not supported.
    """
    # Unpack config
    N_runs = config.N_runs
    
    # lists to store results
    all_fs, all_gbests = [], []
    # history storage lists
    hist_p, hist_f, hist_pb = [], [], []
    
    # Performing runs in series
    if config.run_parallel_mode == 'series':
    
        # Perform all runs in a loop    
        for i in range(0, N_runs):
            # Perform a run
            (best_loc, 
             best_f, 
             p_history, 
             f_history, 
             pb_history) = _perform_run_wrapper2( i, fitness_func, config )
            
            # Store the results
            all_fs.append(best_f)
            all_gbests.append(best_loc)
            hist_p.append(p_history)
            hist_f.append(f_history)
            hist_pb.append(pb_history)
            
    
    # Performing runs in parallel over cpu cores
    elif config.run_parallel_mode == 'cpu':

        # Parameters that will be passed to each perform_run
        jobs = [( i, fitness_func, config ) for i in range(N_runs)]
    
        # Perform runs in parallel
        with ProcessPoolExecutor( max_workers=config.run_n_jobs ) as executor:
            results = list( executor.map( _run_single, jobs ) )

        # Unpack and store the results
        for (best_loc, best_f, p_history, f_history, pb_history) in results:
            all_fs.append(best_f)
            all_gbests.append(best_loc)
            hist_p.append(p_history)
            hist_f.append(f_history)
            hist_pb.append(pb_history)
        
            
    # Performing runs in parallel using MPI
    elif config.run_parallel_mode == 'mpi':
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Determine which runs belong to this rank
        local_results = []
        for i in range(rank, N_runs, size):
            (best_loc, 
             best_f, 
             p_history, 
             f_history, 
             pb_history) = _perform_run_wrapper2( i, fitness_func, config )

            local_results.append(( i,
                                   best_loc,
                                   best_f,
                                   p_history,
                                   f_history,
                                   pb_history ))

        # Gather results to rank 0
        gathered = comm.gather(local_results, root=0)
        if rank == 0:
            ordered = [None] * N_runs
            for proc_results in gathered:
                for result in proc_results:
                    run_idx = result[0]
                    ordered[run_idx] = result
            for result in ordered:
                (_, best_loc,
                 best_f,
                 p_history,
                 f_history,
                 pb_history) = result

                all_fs.append(best_f)
                all_gbests.append(best_loc)
                hist_p.append(p_history)
                hist_f.append(f_history)
                hist_pb.append(pb_history)
        # Force rank > 0 workers to exit
        else: 
            raise SystemExit(0)

    # Raise error if run_parallel_mode is not supported
    else:
        raise NotImplementedError(f"{config.run_parallel_mode} not "
                                  "implemented as a parallelization option.")
        
    # Return the results
    return (    array(all_fs),
                array(all_gbests),
                array(hist_p),
                array(hist_f),
                array(hist_pb) 
            )
    
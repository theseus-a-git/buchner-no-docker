# -*- coding: utf-8 -*-
"""
@file random_uniform.py

@brief Factory function that returns a callable for generating random values 
from a uniform distribution for Particle Swarm Optimization.

The generated-random implementation uses NumPy's modern Generator API
(`numpy.random.default_rng`) to ensure reproducibility and independent
random streams across parallel PSO runs.

@author
Ben Carlson

@date
2026-09-07
"""
from __future__ import annotations # must be first line of code
from typing import TYPE_CHECKING, Callable
from numpy import random, ndarray

if TYPE_CHECKING:
    from utils.config import PSOConfig
    
def make_random_uniform(config: PSOConfig, 
                        run_index: int,
                        generator_index: int
                        ) -> Callable[[int, int], ndarray]:
    """
    @brief Create a callable for retrieving random values.

    This factory function returns a function named `random_uniform` which 
    generates random values from a uniform distribution using a dedicated NumPy
    Generator object created via `numpy.random.default_rng(run_seed)`.

    @param config
    Configuration object containing PSO parameters.

    @param run_index
    The index of this run. Used to select the appropriate seed from
    `config.seed_vals`.
    
    @param generator_index
    The index of this generator. Used to select the appropriate row of seeds 
    from `config.seed_vals`.
    
    @return
    Callable function:

    `random_uniform(N: int, M: int) -> ndarray`

    which returns a matrix of random values of shape (N, M).
    
    @note
    The generated-random implementation uses independent NumPy
    Generator instances instead of NumPy's global random state.
    This avoids interference between concurrent PSO runs.
    
    @note
    Even when config.use_seed=False, seeds are still used. In that case, the 
    SeedSequenceObjects are generated from a random entropy inside config.
    
    @par Example
    Standard random-number generation:
    @code
    value_generator = make_random_uniform(config, 0)

    vals = value_generator(5, 2)
    @endcode

    @par Example
    Parallel independent RNG streams:
    @code
    rng_stream_1 = make_random_uniform(config, 0)
    rng_stream_2 = make_random_uniform(config, 1)

    vals1 = rng_stream_1(20, 5)
    vals2 = rng_stream_2(20, 5)
    @endcode
    """
    # Create independent NumPy Generator instance.
    rng = random.default_rng( config.seed_vals[generator_index][run_index] )
        
    def random_uniform(N: int, M: int) -> ndarray:
        """
        @brief Generate uniformly distributed random values.

        Generates `N*M` floating-point random values uniformly
        distributed on the interval [0, 1).

        @param N
        Number of rows.
        
        @param M
        Number of columns.

        @return
        ndarray of shape `(N,M)` containing generated random
        values.

        @note
        Random values are generated using a dedicated NumPy
        Generator instance local to this callable.
        """
        # Note: N=N_particles and M=N_x, so receiving is redundant but the
        #   structure is kept in case future options require it.
        return rng.random( (N, M) )
        
    return random_uniform

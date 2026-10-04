# -*- coding: utf-8 -*-
"""
@file premade.py

@brief Factory function that returns a callable for retrieving values 
from a user-supplied array for Particle Swarm Optimization.

@author
Ben Carlson

@date
2026-09-07
"""
from __future__ import annotations # must be first line of code
from typing import TYPE_CHECKING, Callable
from numpy import ascontiguousarray, float64, ndarray

if TYPE_CHECKING:
    from utils.config import PSOConfig

def make_premade(config: PSOConfig, 
                 run_index: int,
                 generator_index: int
                 ) -> Callable[[int, int], ndarray]:
    """
    @brief Create a callable for retrieving premade values.

    This factory function returns a function named `premade` which retrieves 
    user-provided values from `config.premade_vals`. Values are internally 
    converted to a NumPy array of type `float64`.

    @param config
    Configuration object containing PSO parameters.

    @param run_index
    The index of this run. Used to access the correct portion of premade_vals.
    
    @param generator_index
    The index of this generator. Used to select the appropriate array of 
    premade values from config.
    
    @return
    Callable function:

    `premade(N: int, M: int) -> ndarray`

    which returns a matrix of random values of shape (N, M).
    
    @note
    The precomputed-value implementation uses array slicing and
    an internal index pointer for efficient sequential access.

    @par Example
    Using precomputed values:
    @code
    premade = [[0.1, 0.5, 0.9, 0.3]] # inside config

    value_generator = make_premade(config, 0)

    vals = value_generator(2, 2)
    @endcode
    
    @par Example
    Parallel independent streams:
    @code
    premade = [[0.1, 0.5, 0.9, 0.3, ...], 
               [0.2, 0.4, 0.3, 0.7, ...]] # inside config
    
    val_stream_1 = make_premade(config, 0)
    val_stream_2 = make_premade(config, 1)

    vals1 = val_stream_1(20, 5)
    vals2 = val_stream_2(20, 5)
    @endcode
    
    @throws NotImplementedError
    Raised if the received generator_index is not supported.
    """
    if generator_index == 0:
        premade_vals = ascontiguousarray(
            config.position_init_premade_vals[run_index], 
            dtype=float64
            )
    elif generator_index == 1:
        premade_vals = ascontiguousarray(
            config.velocity_init_premade_vals[run_index], 
            dtype=float64
            )
    elif generator_index == 2:
        premade_vals = ascontiguousarray(
            config.update_premade_vals[run_index], 
            dtype=float64
            )
    else:
        raise NotImplementedError("Received generator_index not supported")
        
    premade_vals.flags.writeable = False
    idx = 0
    
    def premade(N: int, M: int) -> ndarray:
        """
        @brief Retrieve precomputed values.

        Sequentially retrieves `N*M` values from the internal
        precomputed random-value array.

        @param N
        Number of rows.
        
        @param M
        Number of columns.

        @return
        ndarray of shape `(N,M)` containing random values.

        @note
        Returned arrays are views into the underlying
        `premade_vals` array whenever possible for improved
        performance.

        @warning
        No bounds checking is currently performed in this function. Size
        validation occurs inside config.
        """
        # Note: N=N_particles and M=N_x, so receiving is redundant but the
        #   structure is kept in case future options require it.
        num_vals = N * M
        nonlocal idx
        vals = premade_vals[idx:idx+num_vals]
        idx += num_vals
        return vals.reshape( (N,M) )
       
    return premade

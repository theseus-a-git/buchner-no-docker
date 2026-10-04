# -*- coding: utf-8 -*-
"""
@file sobol.py

@brief Factory function that returns a callable for generating Sobol
quasirandom values for Particle Swarm Optimization.

The Sobol implementation uses SciPy's quasi-Monte Carlo Sobol sequence
generator. Scrambling is enabled and a run-specific seed is used to provide
reproducible, independent quasirandom streams across parallel PSO runs.

@author
Ben Carlson

@date
2026-09-08
"""
from __future__ import annotations  # must be first line of code
from typing import TYPE_CHECKING, Callable
from numpy import ndarray, random
from scipy.stats import qmc

if TYPE_CHECKING:
    from utils.config import PSOConfig

def make_sobol(config: PSOConfig, 
               run_index: int,
               generator_index: int
               ) -> Callable[[int, int], ndarray]:
    """
    @brief Create a callable for retrieving Sobol quasirandom values.
    
    This factory function returns a function named `sobol` which generates
    quasirandom values from a Sobol low-discrepancy sequence.
    
    A separate scrambled Sobol sequence is associated with each PSO run.
    The run-specific seed from `config.seed_vals[run_int]` is used to
    initialize the scrambling, providing reproducible results while
    maintaining independent quasirandom streams across parallel PSO runs.
    
    The returned callable maintains the state of each Sobol sequence between
    calls. Consequently, successive calls continue the sequence rather than
    restarting it.
    
    @param config
    Configuration object containing PSO parameters and run-specific seeds.
    
    @param run_index
    The index of this run. Used to select the appropriate seed from
    `config.seed_vals`.
    
    @param generator_index
    The index of this generator. Used to select the appropriate row of seeds 
    from `config.seed_vals`.
    
    @return
    Callable function:
    
    `sobol(N: int, M: int) -> ndarray`
    
    which returns a matrix of Sobol quasirandom values of shape `(N, M)`.
    
    @note
    The Sobol sequence is generated using `scipy.stats.qmc.Sobol`.
    
    @note
    Scrambling is enabled so that different PSO runs can use independent
    randomized Sobol sequences while retaining the low-discrepancy
    characteristics of the Sobol construction.
    
    @note
    Even when config.use_seed=False, seeds are still used. In that case,
    the SeedSequenceObjects are generated from random entropy inside config.
    
    @note
    Sobol sequences are most naturally generated in blocks containing
    powers of two points. This implementation uses the `random()` method
    rather than `random_base2()` because the PSO API permits arbitrary values
    of `N`.
    
    @par Example
    Standard Sobol quasirandom generation:
    
    @code
    value_generator = make_sobol(config, 0)
    
    vals = value_generator(5, 2)
    @endcode
    
    @par Example
    Parallel independent quasirandom streams:
    
    @code
    sobol_stream_1 = make_sobol(config, 0)
    sobol_stream_2 = make_sobol(config, 1)
    
    vals1 = sobol_stream_1(20, 5)
    vals2 = sobol_stream_2(20, 5)
    @endcode
    """
    # Create the sobol generator.
    sobol_engine = qmc.Sobol(
        d=config.N_x,
        scramble=True,
        seed=random.default_rng(config.seed_vals[generator_index][run_index])
    )
    
    def sobol(N: int, M: int) -> ndarray:
        """
        @brief Generate Sobol quasirandom values.
    
        Generates `N*M` floating-point values from a Sobol low-discrepancy
        sequence. Values lie in the half-open interval [0, 1).
    
        The Sobol sequence continues from its current position on subsequent
        calls.
    
        @param N
        Number of rows (N_particles).
    
        @param M
        Number of columns (N_x). Not used inside the function, but was used 
        when initializing the generator.
    
        @return
        ndarray of shape `(N,M)` containing Sobol quasirandom values.
    
        @note
        SciPy's `random()` method is used instead of `random_base2()` so that
        arbitrary values of `N` are supported by the PSO API.
        """
        # Note: N=N_particles and M=N_x, so receiving is redundant but the
        #   structure is kept in case future options require it.
        return sobol_engine.random(N)
    
    return sobol

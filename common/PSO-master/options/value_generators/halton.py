# -*- coding: utf-8 -*-
"""
@file halton.py

@brief Factory function that returns a callable for generating Halton
quasirandom values for Particle Swarm Optimization.

The Halton implementation uses SciPy's quasi-Monte Carlo Halton sequence
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

def make_halton(config: PSOConfig,
                run_index: int,
                generator_index: int
                ) -> Callable[[int, int], ndarray]:
    """
    @brief Create a callable for retrieving Halton quasirandom values.
    
    This factory function returns a function named `halton` which generates
    quasirandom values from a Halton low-discrepancy sequence.
    
    A separate scrambled Halton sequence is associated with each PSO run.
    The run-specific seed from `config.seed_vals[run_int]` is used to
    initialize the scrambling, providing reproducible results while
    maintaining independent quasirandom streams across parallel PSO runs.
    
    The returned callable maintains the state of the Halton sequence between
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
    
    `halton(N: int, M: int) -> ndarray`
    
    which returns a matrix of Halton quasirandom values of shape `(N,M)`.
    
    @note
    The Halton sequence is generated using `scipy.stats.qmc.Halton`.
    
    @note
    Scrambling is enabled so that different PSO runs can use independent
    randomized Halton sequences while retaining the low-discrepancy
    characteristics of the Halton construction.
    
    @note
    Even when config.use_seed=False, seeds are still used. In that case,
    the SeedSequenceObjects are generated from random entropy inside config.
    
    @note
    Halton sequences can generate arbitrary numbers of points. This
    implementation uses the `random()` method so that arbitrary values of
    `N` are supported by the PSO API.
    
    @par Example
    Standard Halton quasirandom generation:
    
    @code
    value_generator = make_halton(config, 0)
    
    vals = value_generator(5, 2)
    @endcode
    
    @par Example
    Parallel independent quasirandom streams:
    
    @code
    halton_stream_1 = make_halton(config, 0)
    halton_stream_2 = make_halton(config, 1)
    
    vals1 = halton_stream_1(20, 5)
    vals2 = halton_stream_2(20, 5)
    @endcode
    """
    # Create the Halton generator.
    halton_engine = qmc.Halton(
        d=config.N_x,
        scramble=True,
        seed=random.default_rng(config.seed_vals[generator_index][run_index])
    )
    
    def halton(N: int, M: int) -> ndarray:
        """
        @brief Generate Halton quasirandom values.
    
        Generates `N*M` floating-point values from a Halton low-discrepancy
        sequence. Values lie in the half-open interval [0, 1).
    
        The Halton sequence continues from its current position on subsequent
        calls.
    
        @param N
        Number of rows (N_particles).
    
        @param M
        Number of columns (N_x). Not used inside the function, but was used
        when initializing the generator.
    
        @return
        ndarray of shape `(N,M)` containing Halton quasirandom values.
    
        @note
        SciPy's `random()` method is used so that arbitrary values of `N`
        are supported by the PSO API.
        """
        # Note: N=N_particles and M=N_x, so receiving is redundant but the
        #   structure is kept in case future options require it.
        return halton_engine.random(N)
    
    return halton

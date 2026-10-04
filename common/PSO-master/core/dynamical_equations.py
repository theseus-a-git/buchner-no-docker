# -*- coding: utf-8 -*-
"""
@file dynamical_equations.py
@brief Particle Swarm Optimization (PSO) velocity update kernels.

This module provides factory-generated Numba-compiled kernels for updating
particle velocities using different PSO dynamical equations.

The factory function returns a specialized compiled kernel with constant
acceleration coefficients captured in the function closure. This avoids
repeatedly passing static parameters during optimization iterations.

Currently supported dynamical equations:
    - Inertia weight formulation
    - Constriction factor formulation

The returned kernels operate on the location and velocity arrays of a single 
particle.

@author
Ben Carlson

@date
2026-05-11
"""
from numpy import ndarray
from numba import njit
from math import sqrt

from utils.config import PSOConfig

def make_update_velocity(config: PSOConfig) -> callable:
    '''
    @brief Create a compiled PSO velocity update kernel.
    
    Factory function which generates and returns a specialized
    Numba-compiled velocity update kernel corresponding to the
    selected PSO dynamical equation.
    
    The returned function updates the velocity array of one particle.
    
    The acceleration coefficients are captured in the function
    closure and treated as constants for the lifetime of the
    compiled kernel.
    
    Supported equation types:
       - `"inertia"`
       - `"constriction"`
    
    @param config
    PSOConfig dataclass containing the PSO parameters:
    
        - C1: float Cognitive acceleration coefficient.
        - C2: float Social acceleration coefficient.
        - eq_type: String identifier specifying the dynamical equation type.
    
    @return
    Compiled velocity update kernel.
    
    Returned kernel signature:
    @code
    velocity_fn(
         w,
         prev_x,
         prev_v,
         R1,
         R2,
         pbest,
         lbest
    ) -> float
    @endcode
    
    @throws NotImplementedError
    Raised if the requested equation type is unsupported.
    
    @note
    The inertia factor `w` is passed dynamically because it may
    vary between optimization iterations.
    '''
    # unpack config
    C1 = config.C1
    C2 = config.C2
    eq_type = config.eq_type
    
    if eq_type == 'inertia':

        # @njit(signature, cache=True, inline='always')
        @njit(cache=True)
        def update_velocity_i(w      : float,
                              prev_x : ndarray,
                              prev_v : ndarray,
                              R1     : ndarray,
                              R2     : ndarray,
                              pbest  : ndarray,
                              lbest  : ndarray
                              ) -> ndarray:
            r"""
            @brief Compute new particle velocity array using inertia formulation.

            @param w
            Current inertia factor.

            @param prev_x
            Previous particle position array.

            @param prev_v
            Previous particle velocity array.

            @param R1
            Random cognitive scaling factor array.

            @param R2
            Random social scaling factor array.

            @param pbest
            Particle best-known position array.

            @param lbest
            Neighborhood/global best-known position array.

            @return
            Updated particle velocity array.
            
            Computes:
            @f[v_{t+1} = w v_t + C_1 R_1 (pbest - x_t) + C_2 R_2 (lbest - x_t)
            @f]
           
            where:
              - @f$w@f$ is the inertia factor
              - @f$C_1@f$ is the cognitive coefficient
              - @f$C_2@f$ is the social coefficient
            """
            return (w*prev_v) + C1*R1*(pbest - prev_x) + C2*R2*(lbest - prev_x)
        return update_velocity_i

    elif eq_type == "constriction":
        # Constriction factor
        phi = C1 + C2
        K = 2.0 / abs(2.0 - phi - sqrt(phi*phi - 4.0*phi))

        # @njit(signature, cache=True, inline='always')
        @njit(cache=True)
        def update_velocity_k(w      : float,
                              prev_x : ndarray,
                              prev_v : ndarray,
                              R1     : ndarray,
                              R2     : ndarray,
                              pbest  : ndarray,
                              lbest  : ndarray
                              ) -> ndarray:
            r"""
            @brief Compute new particle velocity array using constriction formulation.

            @param w
            Unused parameter retained for interface consistency with
            inertia formulation kernels.

            @param prev_x
            Previous particle position array.

            @param prev_v
            Previous particle velocity array.

            @param R1
            Random cognitive scaling factor array.

            @param R2
            Random social scaling factor array.

            @param pbest
            Particle best-known position array.

            @param lbest
            Neighborhood/global best-known position array.

            @return
            Updated particle velocity array.
            
            Computes:
            @f[v_{t+1} = K \left(v_t + C_1 R_1 (pbest - x_t) 
                                 + C_2 R_2 (lbest - x_t)\right)
            @f]
            
            where:
               - @f$K@f$ is the constriction factor
            """
            return K*(prev_v + C1*R1*(pbest - prev_x) + C2*R2*(lbest - prev_x))
        return update_velocity_k

    else:
        raise NotImplementedError(
            f"{eq_type} not implemented as a dynamical equation option"
        )
        
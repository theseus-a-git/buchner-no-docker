# -*- coding: utf-8 -*-
"""
@file update_particles.py
@brief Particle update kernel for Particle Swarm Optimization (PSO).

This module provides a factory function which generates a Numba-compiled
particle-update kernel for Particle Swarm Optimization (PSO).

The generated kernel updates the position and velocity vectors of all
particles for one optimization iteration using a supplied velocity-update
function.

Particle state is stored in a dense three-dimensional NumPy array with
shape:

@code
(N_particles, 2, N_x)
@endcode

where:

particles[:,0,:] stores particle position vectors
particles[:,1,:] stores particle velocity vectors

The generated kernel performs:

inertia-weight computation
velocity updates
optional velocity clamping
position updates

The actual PSO dynamical equation is delegated to a user-supplied
velocity-update kernel.

@author
Ben Carlson

@date
2026-05-08
"""
from numpy import ndarray, empty
from numba import njit

from utils.config import PSOConfig

def make_update_particles(config: PSOConfig,
                          upd_vel_func: callable,
                          ) -> callable:
    """
    @brief Create a compiled PSO particle-update kernel.
    
    Factory function which generates and returns a specialized
    Numba-compiled kernel for updating all particles in a PSO
    simulation.
    
    The generated kernel updates particle velocities using the
    supplied velocity-update function and then updates particle
    positions.
    
    Optional velocity clamping may also be applied.
    
    @param config
    PSO configuration dataclass containing:
        
        - N_particles: Number of particles
        - clamp_velocity: Enable velocity clamping
        - vmax: Maximum allowed velocity magnitude
    
    @param upd_vel_func
    Compiled velocity-update kernel.
    
    Expected signature:
    
    @code
    upd_vel_func(
        w,
        prev_x,
        prev_v,
        R1,
        R2,
        pbest,
        lbest
    ) -> ndarray
    @endcode
    
    where all vector quantities are one-dimensional NumPy arrays.
    
    @return
    Compiled particle-update kernel.
    
    Returned kernel signature:
    
    @code
    update_particles(
        current_particles,
        pbest_vals,
        lbest_vals,
        R1_matrix,
        R2_matrix,
        iteration
    ) -> ndarray
    @endcode
    
    @note
    Random matrices `R1_matrix` and `R2_matrix` are expected to have
    shape:
    
    @code
    (N_particles, N_x)
    @endcode
    
    where each row contains the random vector associated with one
    particle.
    
    @note
    This implementation is designed to preserve random-number
    consumption ordering compatible with equivalent MATLAB PSO
    implementations for cross-validation purposes.
    
    @warning
    The supplied velocity-update function must operate on vector
    arrays rather than scalar coordinate values.
    """
    # unpack config
    N_iterations = config.N_iterations
    N_particles = config.N_particles
    clamp_velocity = config.clamp_velocity
    vmax = config.vmax
    
    @njit(cache=True)
    def update_particles(current_particles: ndarray, 
                         pbest_vals: ndarray, 
                         lbest_vals: ndarray, 
                         R1_matrix: ndarray,
                         R2_matrix: ndarray,
                         iteration: int
                         ) -> ndarray:
        
        """ 
        @brief Update particle positions and velocities for one PSO iteration. 
        Updates all particle velocity and position vectors using the 
        supplied PSO dynamical equation. 
        
        The update procedure consists of: 
            1. Computing the inertia factor 
            2. Updating particle velocities 
            3. Applying optional velocity clamping 
            4. Updating particle positions 
            
        @param current_particles 
        Particle-state tensor with shape:
        
        @code 
        (N_particles, 2, N_x) 
        @endcode 
        
        where: 
            - `[:,0,:]` contains positions 
            - `[:,1,:]` contains velocities 
            
        @param pbest_vals 
        Matrix of particle personal-best position vectors. Shape: 
            
        @code 
        (N_particles, N_x) 
        @endcode 
        
        @param lbest_vals 
        Matrix of neighborhood/global-best position vectors. Shape: 
            
        @code 
        (N_particles, N_x) 
        @endcode 
        
        @param R1_matrix 
        Matrix of cognitive random vectors. Shape: 
            
        @code 
        (N_particles, N_x) 
        @endcode 
        
        @param R2_matrix 
        Matrix of social random vectors. Shape: 
            
        @code 
        (N_particles, N_x) 
        @endcode 
        
        @param iteration 
        Current optimization iteration number. 
        
        @return Updated particle-state tensor with shape: 
            
        @code 
        (N_particles, 2, N_x) 
        @endcode 
        
        @note 
        Random vectors are consumed particle-by-particle in order to preserve 
        compatibility with MATLAB PSO implementations. 
        """
        # Linearly decaying inertia factor.
        if N_iterations == 1:
            w = 0.4
        else:
            w = 0.9 - 0.5*(iteration-1) / (N_iterations-1)
        
        # Allocate output particle tensor.  
        new_particles = empty(current_particles.shape)
        
        # Update the velocity for each particle
        for j in range(0, N_particles):
            # Extract current particle state.
            x_vals = current_particles[j,0,:]
            v_vals = current_particles[j,1,:]
            
            # Extract associated best positions and random vectors.
            pbest = pbest_vals[j]
            lbest = lbest_vals[j]
            R1 = R1_matrix[j]
            R2 = R2_matrix[j]
            
            # Velocity update
            new_v = upd_vel_func(w, x_vals, v_vals, R1, R2, pbest, lbest)
            
            # Velocity clamping
            if clamp_velocity:
                new_v = new_v.clip(-vmax, vmax)  
                            
            # Position update
            new_x = x_vals + new_v
            
            # Store the new values
            new_particles[j,0] = new_x
            new_particles[j,1] = new_v
        
        return new_particles
    return update_particles

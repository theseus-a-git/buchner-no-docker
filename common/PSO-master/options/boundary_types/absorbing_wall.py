# -*- coding: utf-8 -*-
"""
@file absorbing_wall.py

@brief Contains the boundary handling routine for "absorbing_wall".

@author
Ben Carlson

@date
2026-08-28
"""
from numba import njit
from numpy import ndarray, full, inf, ones, bool_

@njit(cache=True)
def absorbing_wall(particles: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    @brief Identifies which particles have left the search space and 
    clamps each offending coordinate to the nearest boundary of the 
    [0,1] hypercube. This function also sets the velocity component in 
    the corresponding dimension to zero.
    
    @param particles
    Particle-state tensor with shape:
    
    @code 
    (N_particles, 2, N_x) 
    @endcode 
    
    where: 
        - `[:,0,:]` contains positions 
        - `[:,1,:]` contains velocities
        
    @return particles
    Particle-state tensor with shape:
    
    @code 
    (N_particles, 2, N_x) 
    @endcode 
    
    where: 
        - `[:,0,:]` contains positions 
        - `[:,1,:]` contains velocities
    
    @return indices_to_compute
    Bool array with shape:
        
    @code 
    (N_particles,) 
    @endcode 
    
    Entries of 1 (True) represent particles that are inside the 
    search space and will need their fitness computed. Entries of 0 
    (False) represent particles that are outside the search space. All
    values should be returned as 1 in this function.
    
    @return fitness_vals
    Fitness value array with shape:
        
    @code 
    (N_particles,) 
    @endcode
    
    All values should be set to inf inside this function and will be
    overwritten since indices_to_compute is all ones.
    
    @note
    Particle coordinates are assumed to be standardized [0,1].
    """
    N_p = particles.shape[0] # N_p = config.N_particles
    N_x = particles.shape[2] # N_x = config.N_x
    
    # Initialize fitness values to inf. Particles marked for 
    #   computation will have these values overwritten later.
    fitness_vals = full(N_p, inf)
    # Will need to compute fitness for each particle, set to 1 (True)
    indices_to_compute = ones(N_p, dtype=bool_)
    
    # loop over each particle
    for j in range(N_p):
        # loop over each coordinate
        for k in range(N_x):
            x = particles[j, 0, k]
            
            # check if outside search space and clamp
            if x < 0.0:
                particles[j, 0, k] = 0.0  
                particles[j, 1, k] = 0.0
            elif x > 1.0:
                particles[j, 0, k] = 1.0  
                particles[j, 1, k] = 0.0

    return particles, indices_to_compute, fitness_vals

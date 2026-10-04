# -*- coding: utf-8 -*-
"""
@file custom.py

@brief Contains the boundary handling routine for "custom".

@author
Ben Carlson

@date
2026-08-28
"""
from numba import njit
from numpy import ndarray, full, inf, ones, bool_

@njit(cache=True)
def custom(particles: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    @brief This function does not implement boundary conditions, but 
    passes the original particle array back with indices_to_compute all
    ones. This is to be used when the fitness function includes its own
    boundary handling. This function is only used to keep a consistent
    API.
    
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
    
    This is passed back without modification here. It is returned to 
    keep consistent API as other boundary condition functions may 
    change it.
    
    @return indices_to_compute
    Bool array with shape:
        
    @code 
    (N_particles,) 
    @endcode 
    
    Where all entries are 1 (True) to represent particles that need 
    their fitness computed.
    
    @return fitness_vals
    Fitness value array with shape:
        
    @code 
    (N_particles,) 
    @endcode
    
    All values should be set to inf inside this function, but should be 
    overwritten since indices_to_compute is all ones.
    
    @note
    Particle coordinates are assumed to be standardized [0,1].
    """
    n = particles.shape[0] # n = N_particles
    
    # Sets fitness to inf by default
    fitness_vals = full(n, inf)
    # Set all entries to 1 so that each particle will have its fitness
    #   computed. 
    indices_to_compute = ones(n, dtype=bool_)
    
    return particles, indices_to_compute, fitness_vals

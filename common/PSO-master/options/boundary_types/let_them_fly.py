# -*- coding: utf-8 -*-
"""
@file let_them_fly.py

@brief Contains the boundary handling routine for "let_them_fly" also known as
'invisible wall'.

@author
Ben Carlson

@date
2026-08-28
"""
from numba import njit
from numpy import ndarray, full, inf, ones, bool_


@njit(cache=True)
def let_them_fly(particles: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    @brief Identifies which particles have left the search space and 
    sets their fitness values to inf. Then creates a bool array so 
    perform_run knows which particles stayed in the search space and 
    computes fitness accordingly.
    
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
    
    Entries of 1 (True) represent particles that stayed inside the 
    search space and will need their fitness computed. Entries of 0 
    (False) represent particles that left the search space and should
    retain their fitness value of inf.
    
    @return fitness_vals
    Fitness value array with shape:
        
    @code 
    (N_particles,) 
    @endcode
    
    All values should be set to inf inside this function.
    
    @note
    Particle coordinates are assumed to be standardized [0,1].
    """
    n = particles.shape[0] # n = config.N_particles
    
    # Let them fly sets fitness to inf if the particle leaves the space
    fitness_vals = full(n, inf)
    # By default, will need to compute fitness for each particle, so 
    #   set to 1 (True) and change to False if handled here.
    indices_to_compute = ones(n, dtype=bool_)
    
# =============================================================================
#             for j in range(n):
#                 x_vals = particles[j,0,:]
#                 outside_boundary = ((x_vals < 0.0) | (x_vals > 1.0)).any()
#                 if outside_boundary:
#                     # Outside boundar, so set indice to False so it is not sent
#                     #   to fitness_f later
#                     indices_to_compute[j] = False
# =============================================================================
            
    for j in range(n):
        x_vals = particles[j, 0, :]
        for k in range(x_vals.shape[0]):
            x = x_vals[k]
            if x < 0.0 or x > 1.0:
                indices_to_compute[j] = False
                break

    return particles, indices_to_compute, fitness_vals

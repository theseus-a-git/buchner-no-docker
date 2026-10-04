# -*- coding: utf-8 -*-
"""
@file imrpd_1.py

@brief Contains the boundary handling routine for "IMRPD-1".

@author
Ben Carlson

@date
2026-08-28
"""
from numba import njit
from numpy import ndarray, full, inf, ones, bool_

@njit(cache=True)
def IMRPD_1(particles: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    @brief Identifies which particles have left the search space and 
    sets their fitness values to inf. Then creates a bool array so 
    perform_run knows which particles stayed in the search space and 
    computes fitness accordingly. This boundary condition differs from
    'let_them_fly' as it contains two extra conditions: 
        1) The 2D space for the index 2 and 3 dimensions is only 
        allowed to be a triangular half of the normal [0,1] box 
        separated by the x=y line. Any particles that enter the other 
        half have their position and velocity reflected across the x=y 
        line back to the allowed triangle. 
        2) The 2D space for the index 4 and 5 dimensions is restricted 
        to a 90 degree rotated square inside the search space with 
        vertices at (0, 0.5), (0.5, 1), (1, 0.5), and (0.5, 0). Any 
        particles that exit this region have their fitness set to
        infinity.
    
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
        
    This may be modified when the mass coordinates violate the
    canonical ordering. In that case, dimensions 2 and 3 and their
    corresponding velocity components are exchanged.
    
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
    # n = N_particles, but use shape to avoid capturing from outer scope
    n = particles.shape[0] 
    
    # Initialize fitness values to inf. Particles marked for 
    #   computation will have these values overwritten later.
    fitness_vals = full(n, inf)
    # By default, will need to compute fitness for each particle, so 
    #   set to 1 (True) and change to False if handled here.
    indices_to_compute = ones(n, dtype=bool_)
    
    # Check the position of each particle
    for j in range(n):
        
        # If the particle entered the not-allowed triangular half, 
        #   reflect it back. Set index 2 to be the larger mass and 
        #   index 3 to be the smaller mass.
        m1 = particles[j,0,2]
        m2 = particles[j,0,3]
        if m1 < m2:
            # Reflect position
            particles[j,0,2] = m2
            particles[j,0,3] = m1
            # Reflect velocity
            v = particles[j,1,2]
            particles[j,1,2] = particles[j,1,3]
            particles[j,1,3] = v
            
        # Check that the particle stayed in the [0,1] hypercube
        if ((particles[j,0,:] < 0.0) | (particles[j,0,:] > 1.0)).any():
            # Outside hypercube, set flag to False.
            indices_to_compute[j] = False
        # Check that the particle stayed in the 90 degree rotated 
        #   square for dimensions 4 and 5.
        elif abs(particles[j,0,4] - 0.5) + abs(particles[j,0,5] - 0.5) > 0.5:
            # Outside rotated square, set flag to False.
            indices_to_compute[j] = False
            
    return particles, indices_to_compute, fitness_vals

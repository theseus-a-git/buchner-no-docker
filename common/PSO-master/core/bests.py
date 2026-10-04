# -*- coding: utf-8 -*-
"""
@file bests.py
@brief Functions for computing particle and social best locations in PSO.

This module contains utilities used to compute:
- Particle best positions (`pbest`)
- Global best positions (`gbest`)
- Local best positions (`lbest`)

The module supports both:
- Global-best PSO topologies
- Local-best PSO topologies

The social-best functions are generated dynamically using a factory
function based on the selected PSO configuration.

Definitions:
- `pbest`:
    Best position previously visited by an individual particle.
- `gbest`:
    Best position found by the entire swarm.
- `lbest`:
    Best position found within a particle neighborhood.

@author
Ben Carlson

@date
2026-05-20
"""
from numpy import ndarray, argmin, tile, float64, empty
from numpy.typing import NDArray
from numba import njit

from utils.config import PSOConfig

@njit(cache=True)
def get_pbest(current_pbest: ndarray, 
              current_fitness_vals: ndarray,
              new_locations: ndarray,
              new_fitness_vals: ndarray
              ) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    @brief Update particle best locations.
    Compares the current particle best fitness values against newly
    evaluated fitness values.
    If a particle's new fitness value is better than its current best,
    the corresponding location is updated.
    
    @param current_pbest
    The current best locations found by each particle.
    
    Expected shape:
    @code
    (N_particles, N_dimensions)
    @endcode
    
    @param current_fitness_vals
    The fitness values corresponding to the current_pbest locations.
    
    Expected shape:
    @code
    (N_particles,)
    @endcode
    
    @param new_locations
    The new locations of each particle after the position update.
    
    Expected shape:
    @code
    (N_particles, N_dimensions)
    @endcode
    
    @param new_fitness_vals
    The fitness values corresponding to the new_locations locations.
    
    Expected shape:
    @code
    (N_particles,)
    @endcode
    
    @return new_pbest
    Numpy array of updated particle best locations.
    
    Shape:
    @code
    (N_particles, N_dimensions)
    @endcode
    
    @return best_fitness_vals
    Numpy array of particle best fitness values.
    
    Shape:
    @code
    (N_particles,)
    @endcode

    @note
    Lower fitness values are assumed to represent better solutions.
    """
    # Initialize new_pbest with the current pbest values    
    new_pbest = current_pbest.copy()
    best_fitness_vals = current_fitness_vals.copy()
    
    for p in range(0, current_pbest.shape[0]):
        if new_fitness_vals[p] < current_fitness_vals[p]:
            new_pbest[p] = new_locations[p]
            best_fitness_vals[p] = new_fitness_vals[p]
    
    return new_pbest, best_fitness_vals


def make_get_social_best(config: PSOConfig) -> callable:
    """
    @brief Factory function to create a social-best computation function.

    Factory function that returns either:
    - `get_gbest`
    - `get_lbest`

    depending on the configured PSO topology.

    Supported PSO types:
    - `gbest`
    - `lbest`

    
    @param config
    PSOConfig dataclass containing the PSO parameters.
    
    Required members:
    - `PSO_type`
    - `N_particles`
    - `N_local`
    
    @return
    Callable function that returns the social best locations array.
    
    Returned callable signature:
    @code
    social_best_func(
        pbest_vals: ndarray,
        fitness_vals: ndarray
    ) -> ndarray
    @endcode
    
    @throws NotImplementedError
    Raised if PSO_type is not supported. Should not happen because validation 
    occurs inside PSOConfig.
    """
    # unpack config
    PSO_type = config.PSO_type
    N_p = config.N_particles
    N_l = config.N_local
    
    if PSO_type == 'gbest':
        
        @njit(cache=True, inline='always')
        def get_gbest(pbest_vals: ndarray, fitness_vals: ndarray) -> ndarray:
            """
            @brief Finds the single best particle location in the swarm and
            returns an array where every particle receives that same
            global-best location.
            
            @param pbest_vals
            Numpy array of locations arrays. Each location array corresponds to
            a particle.
            
            Shape:
            @code
            (N_particles, N_dimensions)
            @endcode

            @param fitness_vals
            Numpy array of float values. The ith value is the fitness value 
            corresponding to the ith location array in pbest_vals.

            Shape:
            @code
            (N_particles,)
            @endcode

            @return
            Numpy array of where each element is the global best location.
            
            Shape:
            @code
            (N_particles, N_dimensions)
            @endcode
            """
            gbest_loc = pbest_vals[ argmin( fitness_vals ) ]
# =============================================================================
#             return tile(gbest_loc, (N_p, 1))
# =============================================================================
            N_x = pbest_vals.shape[1]
            gbest_array = empty((N_p, N_x))
            for i in range(N_p):
                gbest_array[i] = gbest_loc
            
            return gbest_array
        
        return get_gbest
    
    elif PSO_type == 'lbest':
        
        @njit(cache=True)
        def get_lbest(pbest_vals: ndarray, fitness_vals: ndarray) -> ndarray:
            """
            @brief Function to find and return the local best locations found 
            by each neighborhood. Neighborhoods are constructed using cyclic 
            ring indexing.
            
            @param pbest_vals
            Numpy array of locations arrays. Each location array corresponds to
            a particle.
            
            Shape:
            @code
            (N_particles, N_dimensions)
            @endcode

            @param fitness_vals
            Numpy array of float values. The ith value is the fitness value 
            corresponding to the ith location array in pbest_vals.
            
            Shape:
            @code
            (N_particles,)
            @endcode

            @return
            Numpy array of where each element is the local best location found 
            by the ith particle's neighborhood.   
            
            Shape:
            @code
            (N_particles, N_dimensions)
            @endcode
            
            @note
            Lower fitness values are assumed to represent better solutions.
            """
            N_x = pbest_vals.shape[1]
            lbest_array = empty((N_p, N_x))
            
            left_nbrs = (N_l - 1) // 2
            right_nbrs = N_l - 1 - left_nbrs
        
            for i in range(N_p):
        
                best_idx = i
                best_fit = fitness_vals[i]
        
                for offset in range(-left_nbrs, right_nbrs + 1):

                    idx = (i + offset) % N_p
        
                    if fitness_vals[idx] < best_fit:
                        best_fit = fitness_vals[idx]
                        best_idx = idx
                        
                lbest_array[i] = pbest_vals[best_idx]
        
            return lbest_array
        return get_lbest
    

    else:
        raise NotImplementedError(f"{PSO_type} is not implemented as an option.")

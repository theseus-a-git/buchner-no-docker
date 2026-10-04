# -*- coding: utf-8 -*-
"""
@file periodic.py

@brief Contains the boundary handling routine for "periodic".

@author
Ben Carlson

@date
2026-08-29
"""
from numba import njit
from numpy import ndarray, full, inf, ones, bool_

@njit(cache=True)
def periodic(particles: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    @brief Identifies which particles have left the search space and
    wraps them around periodically into the [0,1] hypercube. Velocity
    components are unchanged.

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

    Positions are wrapped periodically into the [0,1] interval.
    Velocity components are unchanged.

    @return indices_to_compute
    Bool array with shape:

    @code
    (N_particles,)
    @endcode

    Entries of 1 (True) represent particles that are inside the
    search space and will need their fitness computed. Entries of 0
    (False) represent particles that are outside the search space.
    All values are returned as 1 in this function.

    @return fitness_vals
    Fitness value array with shape:

    @code
    (N_particles,)
    @endcode

    All values are set to inf inside this function and will be
    overwritten since indices_to_compute is all ones.

    @note
    Particle coordinates are assumed to be standardized [0,1].

    @note
    Periodic boundary handling treats the search space as a torus.
    For example, a position of 1.2 wraps to 0.2, while a position
    of -0.3 wraps to 0.7.

    @note
    Velocity components are not modified when a position crosses
    a boundary.
    """
    N_p = particles.shape[0]  # N_p = config.N_particles
    N_x = particles.shape[2]  # N_x = config.N_x

    # Initialize fitness values to inf. Particles marked for
    # computation will have these values overwritten later.
    fitness_vals = full(N_p, inf)

    # Every particle will be inside the search space after periodic
    # wrapping, so all particles require fitness computation.
    indices_to_compute = ones(N_p, dtype=bool_)

    # Loop over each particle.
    for j in range(N_p):
        # Loop over each coordinate.
        for k in range(N_x):
            x = particles[j, 0, k]

            # Wrap the position periodically into [0, 1).
            #
            # The modulo operation handles both positive and
            # negative values:
            #
            #   1.2  -> 0.2
            #  -0.3  -> 0.7
            #
            # The upper boundary is represented periodically by
            # the lower boundary, so x = 1.0 remains 0.0.
            x = x % 1.0

            particles[j, 0, k] = x

    return particles, indices_to_compute, fitness_vals

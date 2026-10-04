# -*- coding: utf-8 -*-
"""
@file particle_initialization.py
@brief Particle initialization routine for Particle Swarm Optimization (PSO).

This module provides functionality to initialize particle positions and
velocities for a Particle Swarm Optimization (PSO) algorithm.

Each particle consists of:
- A position vector in the search space.
- A velocity vector used during swarm updates.

Positions are initialized using the supplied value_generator function.
Velocities are initialized relative to the particle positions and may 
optionally be clamped to a maximum magnitude. By default, the particles are 
uniformly distributed in a search space with box topology, but can be uniformly 
distributed on a spherical surface according to the configuration settings or
restrict the 2D subspace for various parameters to a triangular half of
the square grid. 

@author 
Ben Carlson

@date
2026-05-19
"""
from numpy import ndarray, clip, empty, float64, arccos, pi, maximum, minimum

from utils.config import PSOConfig
from utils.dispatch_tables import VALUE_GENERATORS

def initialize_particles(config: PSOConfig, run_val: int) -> ndarray:
    """
    @brief Initialize particle positions and velocities for PSO.

    Creates an array of particles where each particle contains:
    - A position vector.
    - A velocity vector.

    Position values are initialized using values generated from value 
    generators determined by user inputs inside config. Velocity values are 
    initialized relative to the particle's position and optionally clamped 
    according to the configuration settings. By default, the particles are 
    uniformly distributed in a search space with box topology, but can be 
    uniformly distributed on a spherical surface according to the configuration
    settings or restrict the 2D subspace for various parameters to a triangular
    half of the square grid. 

    @param config
    Configuration object containing PSO parameters.
    
    @param run_val
    The index of this run. Used to keep the results of each run in order and 
    to send the correct portion of premade_vals to each run.

    Required members:
    - `space_topology` : Topology type of the search space.
    - `boundary_type` : Denotes mothod for handling boundaries. Used for 
        triangle boundaries.
    - `N_x` : Number of search space dimensions.
    - `N_particles` : Number of particles in the swarm.
    - `clamp_initial_velocity` : Enables/disables velocity clamping.
    - `vmax` : Maximum allowed velocity magnitude.

    @return ndarray
    NumPy array containing all particles.

    Structure:
    @code
    particles[particle_index][0] -> Position vector
    particles[particle_index][1] -> Velocity vector
    @endcode

    Returned array shape:
    @code
    (N_particles, 2, N_x)
    @endcode

    where:
    - index 0 contains positions
    - index 1 contains velocities

    @note
    Initial positions are generated independently for each dimension and
    particle.

    @note
    Initial velocities are computed as:
    @code
    v = random_value - position
    @endcode

    @note
    If velocity clamping is enabled, velocity values exceeding `vmax`
    are limited to:
    @code
    sign(v) * vmax
    @endcode
    
    @note
    Using space_topology == "spherical_surface" assumes theta is the first 
    dimension.
    """
    # Unpack config
    space_topology = config.space_topology
    pos_init_generator = config.position_init_value_type
    vel_init_generator = config.velocity_init_value_type
    boundary_type = config.boundary_type
    N_x = config.N_x
    N_particles = config.N_particles
    clamp_init_velocity = config.clamp_initial_velocity
    vmax = config.vmax
    
    ###########################################################################
    ###########################################################################
    # Position initialization
    ###########################################################################
    
    # Retrieve value matrix for the positions
    pos_init_val_generator = VALUE_GENERATORS[pos_init_generator](config, run_val, 0)
    positions = pos_init_val_generator(N_particles, N_x)
    
    # Organize positions and into particles array. (velocities added later)
    # This is an array of array pairs, where 
    #   - the ith element particles[i] corresponds to the ith particle
    #   - particles[i][0] is the positions array of the ith particle
    #   - particles[i][0][j] is the position in the jth dimension of the ith particle
    #   - particles[i][1] is the velocities array fo the ith particle
    #   - particles[i][1][j] is the velocity in the jth dimension of the ith particle
    particles = empty((N_particles, 2, N_x), dtype=float64)
    particles[:, 0, :] = positions
    
    ###########################################################################
    ###########################################################################
    # These options require modifying the initialized positions before
    # computing the velocities:
    ###########################################################################
    
    # Spherical Surface topology:
    # By default, particles are uniformly distributed in a box, but the theta 
    # dimension must be modified for them to be uniformly distributed on a 
    # spherical surface.
    # NOTE: This assumes theta is the first dimension - documented in README.
    if space_topology == "spherical_surface":
        # Random numbers are on [0, 1]. Shift to [-1, 1] for arccosine.
        particles[:, 0, 0] = arccos(2.0*particles[:, 0, 0] - 1.0) / pi
        
    # 2,3 Triangle initialization:
    # Used when the 2D search subspace of parameters indexed at 2 and 3 is 
    # restricted to a triangle half the size of the original square.
    if boundary_type in ("ltf_23triangle", 
                         "ltf_23triangle_refl", 
                         "IMRPD-1"):
        # .copy() is required or the maximum() line can overwrite too soon.
        m1 = particles[:, 0, 2].copy()
        m2 = particles[:, 0, 3].copy()
        # Set index 2 to be the larger mass and index 3 to be the smaller mass.
        particles[:, 0, 2] = maximum(m1, m2)
        particles[:, 0, 3] = minimum(m1, m2)
                
    ###########################################################################
    ###########################################################################
    # Velocity initialization
    ###########################################################################
        
    # Retrieve values for the velocities
    vel_init_val_generator = VALUE_GENERATORS[vel_init_generator](config, run_val, 1)
    velocities = vel_init_val_generator(N_particles, N_x)
    # Standardize the coordinates based on the modified positions
    velocities = velocities - particles[:, 0, :]

    # Velocity clamping
    if clamp_init_velocity:
        velocities = clip(velocities, -vmax, vmax)

    # Add the velocities into particles array. 
    particles[:, 1, :] = velocities
    
    ###########################################################################
    ###########################################################################
    # These options require modifying the initialized positions after
    # computing the velocities:
    ###########################################################################
        
    # Spin combination initialization:
    # Spin conmbinations are only physical in a 90 degree rotated square inside
    # the [0,1]^2 plane. Need to map velocities too, since they are
    # initialy standardized on the positions and the positions are changing.
    # The position transformation is affine:
    #
    #     x' = A x + b
    #
    # and therefore includes the translation term. Velocities are
    # displacements, so the translation cancels:
    #
    #     v' = A v
    #
    # Consequently, the +1 term used for chi_s positions must not
    # be applied to the velocity transformation.
    if boundary_type in ("IMRPD-1"):
        # Transform spin positions.
        # .copy() is required to avoid overwrite too early.
        chi_a = particles[:, 0, 4].copy()
        chi_s = particles[:, 0, 5].copy()
        # Do affine transform to map all particles into the 90 degree square.
        # Both rotation and translation are needed here.
        particles[:, 0, 4] = 0.5 * (chi_a + chi_s)
        particles[:, 0, 5] = 0.5 * (chi_s - chi_a + 1.0)
    
        # Transform spin velocities.
        v_a = particles[:, 1, 4].copy()
        v_s = particles[:, 1, 5].copy()
        # Only rotation is needed here.
        particles[:, 1, 4] = 0.5 * (v_a + v_s)
        particles[:, 1, 5] = 0.5 * (v_s - v_a)
        
    return particles

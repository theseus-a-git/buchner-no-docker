# -*- coding: utf-8 -*-
"""
@file perform_pso_run.py

@brief Performs a single PSO run.

@author
Ben Carlson

@date
2026-05-18
"""
from numpy import ndarray, array, argmin, empty, float64

from utils.config import PSOConfig
from utils.dispatch_tables import VALUE_GENERATORS

from core.dynamical_equations import make_update_velocity
from core.update_particles import make_update_particles
from core.boundary_handling import make_boundary_handler
from core.bests import make_get_social_best
from core.bests import get_pbest
from core.particle_initialization import initialize_particles

def perform_run(run_val: int,
                config: PSOConfig,
                get_fitness_wrapper: callable
                ) -> tuple[ndarray, float, ndarray, ndarray, ndarray]:
    """
    @brief Function that performs 1 PSO run and returns the results.
    
    @param run_val
    The index of this run. Used to keep the results of each run in order and 
    to send the correct portion of premade_vals to each run.
    
    @param config
    PSO configuration object.
    
    @param get_fitness_wrapper
    Fitness function wrapper to call fitness_func in series/parallel according
    to the user-specified config.fitness_parallel_mode.
    
    @return gbest
    The best location in parameter space, corresponding to final_f.
    
    @return final_f
    The float64 value of the best fitness value obtained by this run.
    
    @return particle_history
    [] if save_history=False. If save_history=True it contains the locations 
    visited and velocities of every particle over every iteration this run. 
    Locations are in standardized coordinates.
    
    @return fitness_history
    [] if save_history=False. If save_history=True it contains the fitness 
    values corresponding to each location in particle_history.
    
    @return pbest_history
    [] if save_history=False. If save_history=True it contains the best 
    location found by each particle updated each iteration throughout the run.
    Locations are in standardized coordinates.
    """
    # Unpack config
    update_generator = config.update_value_type
    N_x = config.N_x
    N_iterations = config.N_iterations
    N_particles = config.N_particles
    save_history = config.save_history

    # Create functions
    upd_vel_func = make_update_velocity(config)
    upd_particles_func = make_update_particles(config, upd_vel_func)
    boundary_handler_func = make_boundary_handler(config)
    get_social_best = make_get_social_best(config)
    update_value_generator = VALUE_GENERATORS[update_generator](config, run_val, 2)
    
    # Initialize the particles
    particles = initialize_particles(config, run_val)
    # Handle particles that left the search space. All particles will always be
    # initialized inside the [0,1]^N box, but may not be inside the allowed 
    # search region if the space is restricted further (triangle spaces, etc.)
    particles, indices_to_compute, fitness_vals = boundary_handler_func(particles)
    fitness_vals[indices_to_compute] = get_fitness_wrapper(particles[indices_to_compute])
    pbest = particles[:,0] # initial pbest is initial locations
    gl_best = get_social_best(pbest, fitness_vals)
    
    # Arrays to store history
    if save_history:
        particle_history = empty((N_iterations+1, N_particles, 2, N_x), dtype=float64)
        fitness_history = empty((N_iterations+1, N_particles), dtype=float64)
        pbest_history = empty((N_iterations+1, N_particles, N_x), dtype=float64)
        # Store the starting values
        particle_history[0] = particles
        fitness_history[0] = fitness_vals
        pbest_history[0] = pbest
    else:
        particle_history = array([], dtype=float64)
        fitness_history = array([], dtype=float64)
        pbest_history = array([], dtype=float64)
        
    # Run N_iterations 
    for i in range(1, N_iterations+1):
        # Get the random number matrices
        R1_matrix = update_value_generator(N_particles, N_x)
        R2_matrix = update_value_generator(N_particles, N_x)
        
        # update the particle locations
        particles = upd_particles_func(particles, pbest, gl_best, 
                                       R1_matrix, R2_matrix, i)
        
        # Handle particles that left the search space
        (particles, 
         indices_to_compute, 
         new_fitness_vals) = boundary_handler_func(particles)
        
        # Compute fitness for all particles in the search space
        new_fitness_vals[indices_to_compute] = get_fitness_wrapper(
                                             particles[indices_to_compute])
        
        # Get new particle and social bests
        pbest, fitness_vals = get_pbest(pbest, 
                                        fitness_vals, 
                                        particles[:,0], 
                                        new_fitness_vals)
        gl_best = get_social_best(pbest, fitness_vals)
        
        # store history
        if save_history:
            particle_history[i] = particles
            fitness_history[i] = new_fitness_vals
            pbest_history[i] = pbest
            
    # Identify the best location and fitness to return
    best_id = argmin( fitness_vals )
    gbest = pbest[ best_id ]
    final_f = fitness_vals[ best_id ]
    
    return (
            gbest, 
            final_f, 
            particle_history, 
            fitness_history, 
            pbest_history
            )

# -*- coding: utf-8 -*-
"""
@file cross_validation_tool.py

@brief This calls run_PSO using the precomputed random values in data.txt which 
was created by create_random_vals.m and is designed for cross-validation with 
the MATLAB implementation of PSO in crcbpso.m

@author
Ben Carlson

@date
2026-07-03
"""
from numpy import array, loadtxt

# Add path to PSO_main.py so run_PSO can be imported.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import run_PSO
from PSO_main import run_PSO

# Import sample benchmark function
from benchmarks.benchmark_functions import Rastrigin


'''*************************PARAMETERS**************************************'''
'''*************************************************************************'''
# Number of particles
N_p = 5
# Number of iterations per run
N_iter = 2000
# coordinate boundary conditions
boundary_conditions = array([[-10.0, 10.0], 
                              [-10.0, 10.0], 
                              [-10.0, 10.0], 
                            [-10.0, 10.0]]) 

base_dir = Path(__file__).resolve().parent
premade_pos = array([loadtxt(base_dir / "data_pos_init.txt")])
premade_vel = array([loadtxt(base_dir / "data_vel_init.txt")])
premade_upd = array([loadtxt(base_dir / "data_upd.txt")])


'''******************************RUNNING PSO********************************'''
'''*************************************************************************'''
def main():
    
    print(f"{N_p} particles")
    print(f"{len(boundary_conditions)} dimensions")
    print(f"{N_iter} iterations")
    
    # Run PSO
    (fitness_vals, 
     best_locations, 
     particle_history,
     fitness_history,
     pbest_history)   = run_PSO(  1, 
                                  N_iter, 
                                  boundary_conditions,
                                  Rastrigin,
                                  N_particles=N_p,
                                  position_init_value_type="premade",
                                  velocity_init_value_type="premade",
                                  update_value_type="premade",
                                  position_init_premade_vals=premade_pos,
                                  velocity_init_premade_vals=premade_vel,
                                  update_premade_vals=premade_upd,
                                  save_history=True )
    
    # Print results for comparison
    print(f"\nbest fitness = {fitness_vals[0]:.25f}")
    print("\ngbest location (physical coordinates) = ")
    for x in best_locations[0]:
        print(f"{x:.25f}")
    
    last_particle_locations = particle_history[0,-1]
    
    print("\nLast locations of each particle (standardized coordinates): ")
    for i, particle in enumerate(last_particle_locations):
        print("\nparticle ", i+1)
        for x in particle[0, :]:
            print(f"{x:.25f}")
    

if __name__ == '__main__':
    main()  

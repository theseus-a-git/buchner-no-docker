# -*- coding: utf-8 -*-
"""
@file plot_benchmark_funcs.py

@brief Program to plot the benchmark functions in benchmark_functions.py

@author
Ben Carlson

@date
2026-05-08
"""
import matplotlib.pyplot as plt
from numpy import array, meshgrid, linspace, float64, zeros
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_functions import Rastrigin, Griewank
from core.coordinate_standardization import un_standardize


# Parameters
N = 400
boundary_conditions_Rast = array([   [-5.12, 5.12],  [-5.12, 5.12]  ])
boundary_conditions_Grie = array([   [-600, 600],  [-600, 600]  ])

x_vals = linspace(0.0, 1.0, N)
y_vals = linspace(0.0, 1.0, N)

# Calculate fitness with Rastrigin
z_vals_Rast = zeros((N, N), dtype=float64)
for i,x in enumerate(x_vals):
    for j,y in enumerate(y_vals):
        z_vals_Rast[i][j] = Rastrigin(
                                    un_standardize(
                                        array([x, y]),
                                        boundary_conditions_Rast)
                                    )

# Calculate fitness with Griewank
z_vals_Grie = zeros((N, N), dtype=float64)
for i,x in enumerate(x_vals):
    for j,y in enumerate(y_vals):
        z_vals_Grie[i][j] = Griewank(
                                    un_standardize(
                                        array([x, y]),
                                        boundary_conditions_Grie)
                                    )
        
x_vals, y_vals = meshgrid(x_vals, y_vals)

# Plot Rastrigin
fig1 = plt.figure(figsize=(10,7))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(x_vals, y_vals, z_vals_Rast, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Rastrigin(x,y)')
ax1.set_title('Surface Plot of Rastrigin')

# Plot Griewank
fig2 = plt.figure(figsize=(10,7))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x_vals, y_vals, z_vals_Grie, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('G(x,y)')
ax2.set_title('Surface Plot of Griewank')

plt.show()

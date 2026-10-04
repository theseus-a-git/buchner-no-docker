# -*- coding: utf-8 -*-
"""
@file coordinate_standardization.py
@brief Coordinate standardization utilities for particle swarm optimization.

This module contains functions that convert between physical parameter space 
and standardized parameter space.

The standardized coordinate system maps each parameter dimension onto the
interval [0, 1]:

@f[
s_i = \frac{x_i - x_{min,i}}{x_{max,i} - x_{min,i}}
@f]

The inverse transformation is:

@f[
x_i = s_i (x_{max,i} - x_{min,i}) + x_{min,i}
@f]

The transformation functions are compiled with Numba for efficient
use inside optimization loops.

@author
Ben Carlson

@date
2026-05-12
"""
from numba import njit

@njit(cache=True, inline='always')
def standardize(x_vals, bcs):
    """
    @brief Function that receives an array of physical coordinates and 
    uses the given boundary conditions to convert to standardized coordinates.
    
    @param x_vals
        One-dimensional NumPy array of physical coordinates.
    
    @param bcs
        NumPy array of shape (N, 2) containing coordinate bounds.
        Each row must contain:
        - boundary_conditions[i, 0] : lower bound
        - boundary_conditions[i, 1] : upper bound
        
    @return
        One-dimensional NumPy array of standardized parameter coordinates.
    """
    return (x_vals - bcs[:,0]) / (bcs[:,1] - bcs[:,0])


# @njit(float64[:](float64[:], float64[:,:]), cache=True, inline='always')
@njit(cache=True, inline='always')
def un_standardize(s_vals, bcs):
    """
    @brief Function that receives an array of standardized coordinates and 
    uses the given boundary conditions to convert to physical coordinates.
    
    @param s_vals
        One-dimensional NumPy array of standardized coordinates.
    
    @param bcs
        NumPy array of shape (N, 2) containing coordinate bounds.
        Each row must contain:
        - boundary_conditions[i, 0] : lower bound
        - boundary_conditions[i, 1] : upper bound
        
    @return
        One-dimensional NumPy array of physical parameter coordinates.
    """
    return s_vals*(bcs[:,1] - bcs[:,0]) + bcs[:,0]


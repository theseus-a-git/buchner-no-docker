# -*- coding: utf-8 -*-
"""
@file benchmark_functions.py

@brief This file contains benchmark functions used for testing PSO.

@author
Ben Carlson

@date
2026-04-18
"""
from numpy import ndarray, cos, pi, sqrt, exp, absolute
from numba import njit


@njit
def Rastrigin(s_vals: ndarray) -> float:
    """
    @brief Calculates the value of the fitness function for the array s_vals.
    using the Rastrigin Function.
    
    @param s_vals
    Numpy array of physical parameters with shape (N_x,).
    
    @return f_val
    The fitness value at the parameter location.
    """
    N = len(s_vals)
    f_val = 0.0
    for i in range(0, N):
        x = s_vals[i]
        # Compute fitness
        f_val += x*x - 10.0*cos(2.0*pi*x)
    f_val += 10.0*N
    return f_val

@njit
def Griewank(s_vals: ndarray) -> float:
    """
    @brief Calculates the value of the fitness function for the array s_vals.
    using the Griewank Function.
    
    @param s_vals
    Numpy array of physical parameters with shape (N_x,).
    
    @return f_val
    The fitness value at the parameter location.
    """
    f_val = 1.0
    summ = 0.0
    prod = 1.0
    for i in range( len(s_vals) ):
        x = s_vals[i]
        # Compute fitness
        summ += x*x
        prod *= cos(x/sqrt(i+1))
    f_val += summ/4000.0
    f_val -= prod
    return f_val

@njit
def Rosenbrock(s_vals: ndarray) -> float:
    """
    @brief Calculates the value of the fitness function for the array s_vals.
    using the Rosenbrock Function.
    
    @param s_vals
    Numpy array of physical parameters with shape (N_x,).
    
    @return f_val
    The fitness value at the parameter location.
    """
    summ = 0.0
    for i in range(len(s_vals) - 1):
        x_i = s_vals[i]
        x_ip1 = s_vals[i+1]
        
        diff1 = x_ip1 - x_i*x_i
        diff2 = x_i - 1.0
        
        summ += 100.0 * diff1 * diff1
        summ += diff2 * diff2
    return summ

@njit
def Ackley(s_vals: ndarray) -> float:
    """
    @brief Calculates the value of the fitness function for the array s_vals.
    using the AckleyFunction.
    
    @param s_vals
    Numpy array of physical parameters with shape (N_x,).
    
    @return f_val
    The fitness value at the parameter location.
    """
    N = len(s_vals)
    f_val = 20.0 + exp(1)
    summ1 = 0.0
    summ2 = 0.0
    for i in range(N):
        x = s_vals[i]
        summ1 += x*x
        summ2 += cos(2.0*pi*x)
    summ1 /= N
    summ2 /= N    
    f_val -= 20.0 * exp(-0.2 * sqrt(summ1))
    f_val -= exp(summ2)
    return f_val

@njit
def Schwefel_221(s_vals: ndarray) -> float:
    """
    @brief Calculates the value of the fitness function for the array s_vals.
    using the Schwefel 2.21 Function.
    
    @param s_vals
    Numpy array of physical parameters with shape (N_x,).
    
    @return f_val
    The fitness value at the parameter location.
    """
    abs_vals = absolute(s_vals)
    return max(abs_vals)


def Rastrigin_cupy(X: ndarray) -> ndarray:
    """
    @brief Calculates the value of the fitness function for the matrix of 
    particle locations, X using the batched Rastrigin Function for cupy.
    
    @param X
    Cupy array of physical parameters with shape (N_particles, N_x).
    
    @return
    Cupy array of fitness values at the parameter locations with shape 
    (N_particles,).
    """
    import cupy as cp
    N = X.shape[1]
    return cp.sum(X * X - 10.0 * cp.cos(2.0 * cp.pi * X), axis=1) + 10.0 * N


def Griewank_cupy(X: ndarray) -> ndarray:
    """
    @brief Calculates the value of the fitness function for the matrix of 
    particle locations, X using the batched Griewank Function for cupy.
    
    @param X
    Cupy array of physical parameters with shape (N_particles, N_x).
    
    @return
    Cupy array of fitness values at the parameter locations with shape 
    (N_particles,).
    """
    import cupy as cp
    dims = cp.arange(1, X.shape[1] + 1, dtype=cp.float64)
    summ = cp.sum(X * X, axis=1)
    prod = cp.prod( cp.cos(X / cp.sqrt(dims)), axis=1)
    return 1.0 + summ / 4000.0 - prod


def Rastrigin_torch(X):
    """
    @brief Calculates the value of the fitness function for the matrix of 
    particle locations, X using the batched Rastrigin Function for torch.
    
    @param X
    torch Tensor of physical parameters with shape (N_particles, N_x).
    
    @return
    torch Tensor of fitness values at the parameter locations with shape 
    (N_particles,).
    """
    import torch
    N = X.shape[1]
    return torch.sum( X * X - 10.0 * torch.cos(2.0 * torch.pi * X), 
                     dim=1) + 10.0 * N


def Griewank_torch(X):
    """
    @brief Calculates the value of the fitness function for the matrix of 
    particle locations, X using the batched Griewank Function for torch.
    
    @param X
    torch Tensor of physical parameters with shape (N_particles, N_x).
    
    @return
    torch Tensor of fitness values at the parameter locations with shape 
    (N_particles,).
    """
    import torch
    dims = torch.arange(
        1,
        X.shape[1] + 1,
        device=X.device,
        dtype=X.dtype
    )

    summ = torch.sum(X * X, dim=1)

    prod = torch.prod(
        torch.cos(X / torch.sqrt(dims)),
        dim=1
    )

    return 1.0 + summ / 4000.0 - prod


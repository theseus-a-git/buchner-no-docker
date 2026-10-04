# -*- coding: utf-8 -*-
"""
@file boundary_handling.py

@brief Returns the boundary handling routine corresponding to the
config.boundary_type parameter.

Supported options given in utils/dispatch_tables.py 

@author
Ben Carlson

@date
2026-08-28
"""
from utils.config import PSOConfig
from utils.dispatch_tables import BOUNDARY_HANDLERS

def make_boundary_handler(config: PSOConfig) -> callable:
    """
    @brief Factory function to create and return a callable which handles
    particles that leave the PSO search space.
    
    @param config
    PSOConfig datatype containing the PSO algorithm parameters.
    
    @return
    Callable function.
    
    @throws ValueError
    If the requested boundary handler is not in BOUNDARY_HANDLERS.
    """
    handler = BOUNDARY_HANDLERS.get(config.boundary_type)
    if handler:
        return handler
    raise ValueError("boundary_type not supported. Please check README for "
                     "supported options.")
    
# -*- coding: utf-8 -*-
"""
@file test_boundary_handling.py

@brief Unit tests for the boundary handling factory found in

@code
PSO.core.boundary_handling
@endcode

The tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-08-28
"""
from dataclasses import dataclass
import pytest

from core.boundary_handling import make_boundary_handler
from utils.dispatch_tables import BOUNDARY_HANDLERS

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight PSO configuration used for testing.

    Members:
    - boundary_type
    """
    boundary_type: str

def test_make_boundary_handler_returns_registered_handler():
    """
    @brief Verify the factory returns the boundary handling routine
    registered for each supported boundary type.
    
    Confirms that the factory correctly dispatches every boundary type
    defined in BOUNDARY_HANDLERS.
    """
    # Test every boundary handler registered in the dispatch table.
    for boundary_type, expected_handler in BOUNDARY_HANDLERS.items():
    
        # Construct a minimal config containing the boundary type.
        config = MockPSOConfig(boundary_type=boundary_type)
    
        returned_handler = make_boundary_handler(config)
    
        assert returned_handler is expected_handler

def test_make_boundary_handler_returns_callable():
    """
    @brief Verify the factory returns a callable for every supported
    boundary type.
    
    Confirms that all entries in the dispatch table satisfy the
    factory's callable interface.
    """
    for boundary_type in BOUNDARY_HANDLERS:
    
        config = MockPSOConfig(boundary_type=boundary_type)
    
        returned_handler = make_boundary_handler(config)
    
        assert callable(returned_handler)

def test_make_boundary_handler_unsupported_type():
    """
    @brief Verify the factory raises ValueError when given an unsupported
    boundary type.
    
    This test therefore verifies the behavior of the dispatch table when a key 
    is not present.
    """
    config = MockPSOConfig(boundary_type="unsupported_boundary_type")
    
    with pytest.raises(ValueError):
        make_boundary_handler(config)

@pytest.mark.parametrize(
    "boundary_type",
        [
        "let_them_fly",
        "absorbing_wall",
        "reflecting_wall",
        "custom",
        "ltf_23triangle",
        "ltf_23triangle_refl",
        "IMRPD-1",
        ],
    )
def test_make_boundary_handler_supported_types(boundary_type):
    """
    @brief Verify each supported boundary type can be dispatched.
    
    Confirms that each boundary type documented by the PSO package is
    present in the dispatch table and returns the corresponding
    boundary handling routine.
    """
    config = MockPSOConfig(boundary_type=boundary_type)
    
    returned_handler = make_boundary_handler(config)
    
    assert returned_handler is BOUNDARY_HANDLERS[boundary_type]

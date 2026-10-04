# -*- coding: utf-8 -*-
"""
@file test_dispatch_tables.py

@brief Unit tests for the dispatch tables found in

@code
PSO.utils.dispatch_tables
@endcode

The tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-08-28
"""
import pytest

from options.boundary_types.let_them_fly import let_them_fly
from options.boundary_types.absorbing_wall import absorbing_wall
from options.boundary_types.reflecting_wall import reflecting_wall
from options.boundary_types.periodic import periodic
from options.boundary_types.custom import custom

from options.boundary_types.ltf_23triangle import LTF_23Triangle
from options.boundary_types.ltf_23triangle_refl import LTF_23Triangle_Refl
from options.boundary_types.imrpd_1 import IMRPD_1

from options.value_generators.halton import make_halton
from options.value_generators.random_uniform import make_random_uniform
from options.value_generators.sobol import make_sobol
from options.value_generators.premade import make_premade

from utils.dispatch_tables import (
                                BOUNDARY_HANDLERS,
                                SUPPORTED_BOUNDARY_TYPES,
                                VALUE_GENERATORS,
                                SUPPORTED_VALUE_GENERATORS,
                            )

def test_boundary_handlers_contains_all_supported_types():
    """
    @brief Verify all supported boundary types are registered in the
    boundary handler dispatch table.
    
    Confirms that each boundary condition implemented by the PSO package
    has a corresponding entry in BOUNDARY_HANDLERS.
    """
    expected_handlers = {
        "let_them_fly": let_them_fly,
        "absorbing_wall": absorbing_wall,
        "reflecting_wall": reflecting_wall,
        "periodic": periodic,
        "custom": custom,
        "ltf_23triangle": LTF_23Triangle,
        "ltf_23triangle_refl": LTF_23Triangle_Refl,
        "IMRPD-1": IMRPD_1,
    }
    
    assert BOUNDARY_HANDLERS == expected_handlers

def test_boundary_handlers_are_callable():
    """
    @brief Verify every boundary handler in the dispatch table is
    callable.
    
    Confirms that BOUNDARY_HANDLERS contains executable boundary
    handling routines rather than invalid or non-callable objects.
    """
    for boundary_type, handler in BOUNDARY_HANDLERS.items():
    
        assert callable(handler), (
            f"Boundary handler for {boundary_type!r} is not callable."
        )

def test_supported_boundary_types_matches_boundary_handlers():
    """
    @brief Verify SUPPORTED_BOUNDARY_TYPES contains exactly the keys
    registered in BOUNDARY_HANDLERS.
    
    Confirms that the set used for defensive validation remains
    synchronized with the boundary handler dispatch table.
    """
    assert SUPPORTED_BOUNDARY_TYPES == frozenset(BOUNDARY_HANDLERS)

def test_supported_boundary_types_is_frozen():
    """
    @brief Verify SUPPORTED_BOUNDARY_TYPES is immutable.
    
    Confirms that the collection of supported boundary types cannot be
    accidentally modified after the dispatch table is initialized.
    """
    assert isinstance(SUPPORTED_BOUNDARY_TYPES, frozenset)

@pytest.mark.parametrize(
    "boundary_type, expected_handler",
        [
        ("let_them_fly", let_them_fly),
        ("absorbing_wall", absorbing_wall),
        ("reflecting_wall", reflecting_wall),
        ("periodic", periodic),
        ("custom", custom),
        ("ltf_23triangle", LTF_23Triangle),
        ("ltf_23triangle_refl", LTF_23Triangle_Refl),
        ("IMRPD-1", IMRPD_1),
        ],
    )
def test_boundary_handler_dispatch(boundary_type, expected_handler):
    """
    @brief Verify each boundary type maps to its expected handler.
    
    Confirms that individual entries in BOUNDARY_HANDLERS reference the
    correct boundary handling implementation.
    """
    assert BOUNDARY_HANDLERS[boundary_type] is expected_handler


def test_value_generators_contains_all_supported_types():
    """
    @brief Verify all supported value generator types are registered in
    the value generator dispatch table.

    Confirms that each value generation method implemented by the PSO
    package has a corresponding entry in VALUE_GENERATORS.
    """
    expected_generators = {
        "halton": make_halton,
        "random-uniform": make_random_uniform,
        "sobol": make_sobol,
        "premade": make_premade,
    }

    assert VALUE_GENERATORS == expected_generators


def test_value_generators_are_callable():
    """
    @brief Verify every value generator in the dispatch table is
    callable.

    Confirms that VALUE_GENERATORS contains executable value generation
    routines rather than invalid or non-callable objects.
    """
    for generator_type, generator in VALUE_GENERATORS.items():

        assert callable(generator), (
            f"Value generator for {generator_type!r} is not callable."
        )


def test_supported_value_generators_matches_value_generators():
    """
    @brief Verify SUPPORTED_VALUE_GENERATORS contains exactly the keys
    registered in VALUE_GENERATORS.

    Confirms that the set used for defensive validation remains
    synchronized with the value generator dispatch table.
    """
    assert SUPPORTED_VALUE_GENERATORS == frozenset(VALUE_GENERATORS)


def test_supported_value_generators_is_frozen():
    """
    @brief Verify SUPPORTED_VALUE_GENERATORS is immutable.

    Confirms that the collection of supported value generator types
    cannot be accidentally modified after the dispatch table is
    initialized.
    """
    assert isinstance(SUPPORTED_VALUE_GENERATORS, frozenset)


@pytest.mark.parametrize(
    "generator_type, expected_generator",
    [
         ("halton", make_halton),
         ("random-uniform", make_random_uniform),
         ("sobol", make_sobol),
         ("premade", make_premade),
    ],
)
def test_value_generator_dispatch(generator_type, expected_generator):
    """
    @brief Verify each value generator type maps to its expected
    generator.

    Confirms that individual entries in VALUE_GENERATORS reference the
    correct value generation implementation.
    """
    assert VALUE_GENERATORS[generator_type] is expected_generator

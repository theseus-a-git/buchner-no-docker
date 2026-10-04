# -*- coding: utf-8 -*-
"""
@file test_coordinate_standardization.py
@brief Unit tests for coordinate standardization utilities.

This module contains unit tests for the coordinate transformation functions
defined in:

@code
PSO.core.coordinate_standardization
@endcode

The tests verify:
- Correct forward standardization
- Correct inverse transformation
- Round-trip consistency
- Boundary behavior
- Out-of-bounds behavior
- Endpoint mapping correctness
- Output dtype consistency

The transformations under test are:

@f[
s_i = \frac{x_i - x_{min,i}}{x_{max,i} - x_{min,i}}
@f]

and:

@f[
x_i = s_i (x_{max,i} - x_{min,i}) + x_{min,i}
@f]

These tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-06-28
"""

import numpy as np

from PSO.core.coordinate_standardization import (
    standardize,
    un_standardize,
)


def test_standardize_basic():
    """
    @brief Verify standardization against known reference values.

    Tests a simple two-dimensional transformation with analytically known
    results.

    Boundary conditions:
    - Dimension 0: [0, 10]
    - Dimension 1: [-5, 5]

    Expected mappings:
    - x = 5  -> s = 0.5
    - x = 0  -> s = 0.5
    """

    bcs = np.array([
        [0.0, 10.0],
        [-5.0, 5.0]
    ])

    x = np.array([5.0, 0.0])

    expected = np.array([0.5, 0.5])

    result = standardize(x, bcs)

    assert np.allclose(result, expected)


def test_un_standardize_basic():
    """
    @brief Verify inverse transformation against known reference values.

    Tests reconstruction of physical coordinates from standardized
    coordinates using analytically known values.
    """

    bcs = np.array([
        [0.0, 10.0],
        [-5.0, 5.0]
    ])

    s = np.array([0.5, 0.5])

    expected = np.array([5.0, 0.0])

    result = un_standardize(s, bcs)

    assert np.allclose(result, expected)


def test_round_trip_transformation():
    """
    @brief Verify forward and inverse transformations are consistent.

    Tests that:

    @f[
    x \\rightarrow s \\rightarrow x
    @f]

    reproduces the original coordinates within floating-point tolerance.
    """

    bcs = np.array([
        [0.0, 10.0],
        [-5.0, 5.0],
        [100.0, 200.0]
    ])

    x_original = np.array([3.2, -1.7, 145.0])

    s = standardize(x_original, bcs)

    x_recovered = un_standardize(s, bcs)

    assert np.allclose(x_original, x_recovered)


def test_standardize_out_of_bounds():
    r"""
    @brief Verify out-of-bounds coordinates are not clipped.

    Confirms that the standardization function performs only an affine
    transformation and does not enforce the interval [0, 1].

    Example:

    @f[
    x = 15,\quad [x_{min}, x_{max}] = [0,10]
    @f]

    should produce:

    @f[
    s = 1.5
    @f]
    """

    bcs = np.array([
        [0.0, 10.0]
    ])

    x = np.array([15.0])

    expected = np.array([1.5])

    result = standardize(x, bcs)

    assert np.allclose(result, expected)


def test_un_standardize_out_of_bounds():
    r"""
    @brief Verify inverse transformation does not clip coordinates.

    Confirms that standardized coordinates outside [0, 1] generate physical
    coordinates outside the parameter bounds.

    Example:

    @f[
    s = -0.5,\quad [x_{min}, x_{max}] = [0,10]
    @f]

    should produce:

    @f[
    x = -5
    @f]
    """

    bcs = np.array([
        [0.0, 10.0]
    ])

    s = np.array([-0.5])

    expected = np.array([-5.0])

    result = un_standardize(s, bcs)

    assert np.allclose(result, expected)


def test_endpoint_mapping():
    """
    @brief Verify exact lower and upper endpoint mappings.

    Confirms:
    - Lower bounds map to 0
    - Upper bounds map to 1

    for each parameter dimension.
    """

    bcs = np.array([
        [-2.0, 8.0]
    ])

    lower = np.array([-2.0])
    upper = np.array([8.0])

    assert np.allclose(standardize(lower, bcs), [0.0])
    assert np.allclose(standardize(upper, bcs), [1.0])


def test_dtype_preservation():
    """
    @brief Verify output arrays preserve float64 dtype.

    Ensures the compiled transformation functions return NumPy arrays
    compatible with float64 inputs.
    """

    bcs = np.array([
        [0.0, 1.0]
    ], dtype=np.float64)

    x = np.array([0.5], dtype=np.float64)

    result = standardize(x, bcs)

    assert result.dtype == np.float64


def test_zero_span_boundary_conditions():
    """
    @brief Verify zero-width boundary conditions produce non-finite values.

    The current implementation performs no validation and therefore relies
    on NumPy's floating-point behavior when a parameter span is zero.
    """

    bcs = np.array([
        [1.0, 1.0]
    ])

    x = np.array([1.0])

    result = standardize(x, bcs)

    assert not np.isfinite(result[0])


def test_negative_span_boundary_conditions():
    """
    @brief Verify inverted boundary conditions are handled algebraically.

    The transformation is purely affine and therefore still produces a
    mathematically valid result even when the parameter span is negative.
    """

    bcs = np.array([
        [10.0, 0.0]
    ])

    x = np.array([5.0])

    expected = np.array([0.5])

    result = standardize(x, bcs)

    assert np.allclose(result, expected)
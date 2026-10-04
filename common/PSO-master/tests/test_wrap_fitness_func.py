# -*- coding: utf-8 -*-
"""
@file test_wrap_fitness_func.py

@brief Unit tests for fitness-function wrapper generation.

These tests validate:

* serial fitness evaluation
* CPU ProcessPoolExecutor evaluation
* CuPy GPU evaluation (when available)
* coordinate un-standardization
* preservation of particle ordering
* unsupported backend handling

The CUDA-Torch backend is intentionally not tested because it is
currently disabled by input validation.

@author
Ben Carlson

@date
2026-07-02
"""
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose
from unittest.mock import patch

from parallel_wrappers.wrap_fitness_func import make_get_fitness_wrapper


# -------------------------------------------------------------------------
# Optional CuPy support
# -------------------------------------------------------------------------

try:
    import cupy as cp

    try:
        CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        CUPY_AVAILABLE = False

except ImportError:
    cp = None
    CUPY_AVAILABLE = False


# -------------------------------------------------------------------------
# Mock configuration
# -------------------------------------------------------------------------

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration used for testing.
    """
    fitness_parallel_mode: str
    boundary_conditions: object = None


# -------------------------------------------------------------------------
# Test fitness functions
# -------------------------------------------------------------------------

def scalar_fitness(x):
    """
    @brief Simple scalar fitness function.
    """
    return np.sum(x ** 2)


def batched_cupy_fitness(X):
    """
    @brief Batched CuPy fitness function.
    """
    return cp.sum(X ** 2, axis=1)


# -------------------------------------------------------------------------
# Shared particle data
# -------------------------------------------------------------------------

PARTICLES = np.array(
    [
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
    ]
)

EXPECTED = np.array([
    5.0,
    25.0,
    61.0,
])


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

@patch(
    "parallel_wrappers.wrap_fitness_func.un_standardize",
    side_effect=lambda x, bc: x,
)
def test_series_wrapper(mock_unstandardize):
    """
    @brief Verify serial wrapper evaluates every particle.
    """
    config = MockPSOConfig("series")

    wrapper = make_get_fitness_wrapper(
        config,
        scalar_fitness,
    )

    result = wrapper(PARTICLES)

    assert_allclose(result, EXPECTED)
    mock_unstandardize.assert_called_once()


@patch(
    "parallel_wrappers.wrap_fitness_func.un_standardize",
    side_effect=lambda x, bc: x,
)
def test_cpu_wrapper(mock_unstandardize):
    """
    @brief Verify CPU wrapper evaluates every particle using
    ProcessPoolExecutor.
    """
    config = MockPSOConfig("cpu")

    with ProcessPoolExecutor(max_workers=2) as executor:

        wrapper = make_get_fitness_wrapper(
            config,
            scalar_fitness,
            executor=executor,
        )

        result = wrapper(PARTICLES)

    assert_allclose(result, EXPECTED)
    mock_unstandardize.assert_called_once()


@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
@patch(
    "parallel_wrappers.wrap_fitness_func.un_standardize",
    side_effect=lambda x, bc: x,
)
def test_cuda_cupy_wrapper(mock_unstandardize):
    """
    @brief Verify CuPy wrapper performs batched evaluation.
    """
    config = MockPSOConfig("cuda-cupy")

    wrapper = make_get_fitness_wrapper(
        config,
        batched_cupy_fitness,
    )

    result = wrapper(PARTICLES)

    assert_allclose(result, EXPECTED)
    mock_unstandardize.assert_called_once()


@patch(
    "parallel_wrappers.wrap_fitness_func.un_standardize",
    side_effect=lambda x, bc: x,
)
def test_result_order_preserved(mock_unstandardize):
    """
    @brief Verify wrapper preserves particle ordering.
    """
    config = MockPSOConfig("series")

    wrapper = make_get_fitness_wrapper(
        config,
        lambda x: x[0],
    )

    particles = np.array(
        [
            [[5.0]],
            [[1.0]],
            [[9.0]],
        ]
    )

    result = wrapper(particles)

    assert_allclose(result, np.array([5.0, 1.0, 9.0]))


@patch(
    "parallel_wrappers.wrap_fitness_func.un_standardize",
    side_effect=lambda x, bc: x,
)
def test_empty_particle_array(mock_unstandardize):
    """
    @brief Verify wrapper handles an empty particle array.
    """
    config = MockPSOConfig("series")

    wrapper = make_get_fitness_wrapper(
        config,
        scalar_fitness,
    )

    particles = np.empty((0, 1, 2))

    result = wrapper(particles)

    assert result.shape == (0,)


def test_invalid_backend():
    """
    @brief Verify unsupported fitness backend raises an exception.
    """
    config = MockPSOConfig("invalid")

    with pytest.raises(NotImplementedError):
        make_get_fitness_wrapper(
            config,
            scalar_fitness,
        )
        
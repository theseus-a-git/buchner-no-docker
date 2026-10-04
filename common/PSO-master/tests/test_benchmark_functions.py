# -*- coding: utf-8 -*-
"""
@file test_benchmark_functions.py

@brief Unit tests for benchmark optimization functions.

This module validates the benchmark objective functions implemented in
`benchmark_functions.py`.

The tests verify:

- Known global minima.
- Agreement with hand-computed values.
- Consistency between scalar and batched implementations.
- Agreement between NumPy, CuPy, and PyTorch implementations.
- Output shapes.
- Deterministic behavior.

GPU-specific tests are skipped automatically when the required backend
is unavailable.

@author
Ben Carlson

@date
2026-09-19
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from benchmarks.benchmark_functions import (
    Rastrigin,
    Griewank,
    Rosenbrock,
    Ackley,
    Schwefel_221,
    Rastrigin_cupy,
    Griewank_cupy,
    Rastrigin_torch,
    Griewank_torch,
)

# Check gpu availability before testing gpu functions
try:
    import cupy as cp
    try:
        CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        CUPY_AVAILABLE = False
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def torch_cuda_safe():
    """
    @brief Determine whether PyTorch CUDA execution is available.

    @return
    True if a CUDA tensor can be created and evaluated successfully.
    """
    if not TORCH_AVAILABLE:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        # Lightweight execution check. This catches broken wheels and
        # incompatible GPU architectures.
        x = torch.zeros(1, device="cuda")
        y = x + 1
        y.item()
        return True
    except Exception:
        return False

TORCH_CUDA_AVAILABLE = torch_cuda_safe()


# --------------------------------------------------------------------------
# Scalar benchmark functions
# --------------------------------------------------------------------------

def test_rastrigin_global_minimum():
    """
    @brief Verify Rastrigin evaluates to zero at the global minimum.
    """

    x = np.zeros(5)

    assert Rastrigin(x) == pytest.approx(0.0)


def test_griewank_global_minimum():
    """
    @brief Verify Griewank evaluates to zero at the global minimum.
    """

    x = np.zeros(5)

    assert Griewank(x) == pytest.approx(0.0)

def test_rosenbrock_global_minimum():
    """
    @brief Verify Rosenbrock evaluates to zero at the global minimum.
    """

    x = np.ones(5)

    assert Rosenbrock(x) == pytest.approx(0.0)


def test_ackley_global_minimum():
    """
    @brief Verify Ackley evaluates to zero at the global minimum.
    """

    x = np.zeros(5)

    assert Ackley(x) == pytest.approx(0.0)


def test_schwefel_221_global_minimum():
    """
    @brief Verify Schwefel 2.21 evaluates to zero at the global minimum.
    """

    x = np.zeros(5)

    assert Schwefel_221(x) == pytest.approx(0.0)

def test_rastrigin_known_value():
    """
    @brief Verify Rastrigin matches an analytical value.
    """

    x = np.array([1.0])

    expected = 1.0

    assert Rastrigin(x) == pytest.approx(expected)


def test_griewank_known_value():
    """
    @brief Verify Griewank matches an analytical value.
    """

    x = np.array([1.0])

    expected = 1.0 + 1.0 / 4000.0 - np.cos(1.0)

    assert Griewank(x) == pytest.approx(expected)
    
def test_ackley_known_value():
    """
    @brief Verify Ackley matches an analytically computed value.
    """

    x = np.array([1.0])

    expected = (
        20.0
        + np.e
        - 20.0 * np.exp(-0.2)
        - np.exp(np.cos(2.0 * np.pi))
    )

    assert Ackley(x) == pytest.approx(expected)


def test_schwefel_221_known_value():
    """
    @brief Verify Schwefel 2.21 returns the maximum absolute value.
    """

    x = np.array([-1.0, 4.5, -2.0, 3.0])
    expected = 4.5

    assert Schwefel_221(x) == pytest.approx(expected)


def test_schwefel_221_negative_values():
    """
    @brief Verify Schwefel 2.21 correctly handles negative values.
    """

    x = np.array([-7.0, -3.0, -5.0])
    expected = 7.0

    assert Schwefel_221(x) == pytest.approx(expected)

def test_rosenbrock_two_element_vector():
    """
    @brief Verify Rosenbrock correctly evaluates a two-element vector.
    """

    x = np.array([1.0, 2.0])
    expected = 100.0

    assert Rosenbrock(x) == pytest.approx(expected)
    
@pytest.mark.parametrize(
    "function",
    [
        Rastrigin,
        Griewank,
        Rosenbrock,
        Ackley,
        Schwefel_221,
    ],
)
def test_scalar_functions_are_deterministic(function):
    """
    @brief Verify scalar benchmark functions return deterministic results.
    """

    x = np.array([0.5, -1.25, 2.0, -3.5])

    result_1 = function(x)
    result_2 = function(x)

    assert result_1 == pytest.approx(result_2)


def test_rastrigin_empty_array():
    """
    @brief Verify Rastrigin handles an empty input array.
    """

    x = np.array([])

    assert Rastrigin(x) == pytest.approx(0.0)


def test_griewank_empty_array():
    """
    @brief Verify Griewank handles an empty input array.
    """

    x = np.array([])

    assert Griewank(x) == pytest.approx(0.0)


def test_rosenbrock_single_element_array():
    """
    @brief Verify Rosenbrock returns zero for a single-element array.

    The implementation contains no adjacent pairs for a one-element input.
    """

    x = np.array([3.0])

    assert Rosenbrock(x) == pytest.approx(0.0)

# --------------------------------------------------------------------------
# CuPy implementations
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
def test_rastrigin_cupy_matches_scalar():
    """
    @brief Verify CuPy batched Rastrigin matches scalar implementation.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.5, 0.5],
        ]
    )

    expected = np.array([Rastrigin(row) for row in X])
    result = cp.asnumpy(Rastrigin_cupy(cp.asarray(X)))
    assert_allclose(result, expected)


@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
def test_griewank_cupy_matches_scalar():
    """
    @brief Verify CuPy batched Griewank matches scalar implementation.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.5, 0.5],
        ]
    )

    expected = np.array([Griewank(row) for row in X])
    result = cp.asnumpy(Griewank_cupy(cp.asarray(X)))
    assert_allclose(result, expected)


@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
def test_cupy_output_shape():
    """
    @brief Verify CuPy benchmark functions return one value per particle.
    """
    X = cp.zeros((8, 5))

    assert Rastrigin_cupy(X).shape == (8,)
    assert Griewank_cupy(X).shape == (8,)

@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
def test_cupy_deterministic_results():
    """
    @brief Verify CuPy benchmark functions return deterministic results.
    """
    X = cp.asarray(
        [
            [0.25, -1.0, 2.5],
            [-2.0, 1.5, 0.75],
        ]
    )

    rastrigin_1 = cp.asnumpy(Rastrigin_cupy(X))
    rastrigin_2 = cp.asnumpy(Rastrigin_cupy(X))

    griewank_1 = cp.asnumpy(Griewank_cupy(X))
    griewank_2 = cp.asnumpy(Griewank_cupy(X))

    assert_allclose(rastrigin_1, rastrigin_2)
    assert_allclose(griewank_1, griewank_2)

# --------------------------------------------------------------------------
# Torch implementations
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not TORCH_CUDA_AVAILABLE,
    reason="CUDA-capable PyTorch not available or incompatible GPU arch",
)
def test_rastrigin_torch_matches_scalar():
    """
    @brief Verify Torch batched Rastrigin matches scalar implementation.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.5, 0.5],
        ]
    )

    expected = np.array([Rastrigin(row) for row in X])

    result = (
        Rastrigin_torch(
            torch.tensor(X, device="cuda", dtype=torch.float64)
        )
        .cpu()
        .numpy()
    )

    assert_allclose(result, expected)


@pytest.mark.skipif(
    not TORCH_CUDA_AVAILABLE,
    reason="CUDA-capable PyTorch not available or incompatible GPU arch",
)
def test_griewank_torch_matches_scalar():
    """
    @brief Verify Torch batched Griewank matches scalar implementation.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-1.5, 0.5],
        ]
    )

    expected = np.array([Griewank(row) for row in X])

    result = (
        Griewank_torch(
            torch.tensor(X, device="cuda", dtype=torch.float64)
        )
        .cpu()
        .numpy()
    )

    assert_allclose(result, expected)


@pytest.mark.skipif(
    not TORCH_CUDA_AVAILABLE,
    reason="CUDA-capable PyTorch not available or incompatible GPU arch",
)
def test_torch_output_shape():
    """
    @brief Verify Torch benchmark functions return one value per particle.
    """

    X = torch.zeros((8, 5), device="cuda", dtype=torch.float64)

    assert tuple(Rastrigin_torch(X).shape) == (8,)
    assert tuple(Griewank_torch(X).shape) == (8,)


@pytest.mark.skipif(
    not TORCH_CUDA_AVAILABLE,
    reason=(
        "CUDA-capable PyTorch not available or incompatible GPU arch"
    ),
)
def test_torch_deterministic_results():
    """
    @brief Verify Torch benchmark functions return deterministic results.
    """

    X = torch.tensor(
        [
            [0.25, -1.0, 2.5],
            [-2.0, 1.5, 0.75],
        ],
        device="cuda",
        dtype=torch.float64,
    )

    rastrigin_1 = Rastrigin_torch(X).cpu().numpy()
    rastrigin_2 = Rastrigin_torch(X).cpu().numpy()

    griewank_1 = Griewank_torch(X).cpu().numpy()
    griewank_2 = Griewank_torch(X).cpu().numpy()

    assert_allclose(rastrigin_1, rastrigin_2)
    assert_allclose(griewank_1, griewank_2)

# --------------------------------------------------------------------------
# Cross-backend consistency
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not (CUPY_AVAILABLE and TORCH_CUDA_AVAILABLE),
    reason="CUDA-capable CuPy device not available or CUDA-capable PyTorch not available or incompatible GPU arch",
)
def test_all_backends_agree():
    """
    @brief Verify NumPy, CuPy, and Torch implementations agree.

    Confirms that all implementations compute identical fitness values
    for the same batch of particle locations.
    """

    rng = np.random.default_rng(12345)

    X = rng.uniform(-5.0, 5.0, (10, 6))

    expected_r = np.array([Rastrigin(row) for row in X])
    expected_g = np.array([Griewank(row) for row in X])

    cupy_r = cp.asnumpy(Rastrigin_cupy(cp.asarray(X)))
    cupy_g = cp.asnumpy(Griewank_cupy(cp.asarray(X)))

    torch_r = (
        Rastrigin_torch(
            torch.tensor(X, device="cuda", dtype=torch.float64)
        )
        .cpu()
        .numpy()
    )

    torch_g = (
        Griewank_torch(
            torch.tensor(X, device="cuda", dtype=torch.float64)
        )
        .cpu()
        .numpy()
    )

    assert_allclose(cupy_r, expected_r)
    assert_allclose(cupy_g, expected_g)

    assert_allclose(torch_r, expected_r)
    assert_allclose(torch_g, expected_g)
    
    
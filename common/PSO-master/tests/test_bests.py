# -*- coding: utf-8 -*-
"""
@file test_bests.py
@brief Unit tests for PSO best-location computation functions.

This module contains unit tests for:

- `get_pbest`
- `make_get_social_best`
- `get_gbest`
- `get_lbest`

The tests verify:

- Correct updating of particle-best locations
- Correct updating of particle-best fitness values
- Preservation of inputs by `get_pbest`
- Correct computation of global-best locations
- Correct computation of local-best neighborhood locations
- Correct handling of odd and even neighborhood sizes
- Proper exception handling for unsupported PSO types
- Regression against Matlab reference data

Deterministic test data is used to ensure repeatability and
predictable behavior.

These tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-06-29
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.bests import get_pbest, make_get_social_best


@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object used for testing.

    Members:
    - `PSO_type`
    - `N_particles`
    - `N_local`
    """

    PSO_type: str
    N_particles: int
    N_local: int


def test_get_pbest_updates_improved_particles():
    """
    @brief Verify improved particles update both locations and fitness values.
    """

    current_pbest = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])

    current_fitness = np.array([5.0, 2.0, 8.0])

    new_locations = np.array([
        [1.5, 1.5],
        [2.5, 2.5],
        [0.5, 0.5]
    ])

    new_fitness = np.array([4.0, 3.0, 1.0])

    expected_locations = np.array([
        [1.5, 1.5],
        [2.0, 2.0],
        [0.5, 0.5]
    ])

    expected_fitness = np.array([
        4.0,
        2.0,
        1.0
    ])

    pbest, fitness = get_pbest(
        current_pbest,
        current_fitness,
        new_locations,
        new_fitness
    )

    assert np.allclose(pbest, expected_locations)
    assert np.allclose(fitness, expected_fitness)


def test_get_pbest_no_updates():
    """
    @brief Verify pbest remains unchanged when no improvements occur.
    """

    current_pbest = np.array([
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    current_fitness = np.array([1.0, 2.0])

    new_locations = np.array([
        [5.0, 5.0],
        [6.0, 6.0]
    ])

    new_fitness = np.array([
        10.0,
        20.0
    ])

    pbest, fitness = get_pbest(
        current_pbest,
        current_fitness,
        new_locations,
        new_fitness
    )

    assert np.allclose(pbest, current_pbest)
    assert np.allclose(fitness, current_fitness)


def test_get_pbest_does_not_modify_inputs():
    """
    @brief Verify get_pbest does not modify its input arrays.

    The implementation should return copies rather than modifying
    the supplied arrays in-place.
    """

    current_pbest = np.array([
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    current_fitness = np.array([5.0, 6.0])

    pbest_copy = current_pbest.copy()
    fitness_copy = current_fitness.copy()

    new_locations = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])

    new_fitness = np.array([1.0, 1.0])

    get_pbest(
        current_pbest,
        current_fitness,
        new_locations,
        new_fitness
    )

    assert np.array_equal(current_pbest, pbest_copy)
    assert np.array_equal(current_fitness, fitness_copy)


def test_get_gbest_returns_global_best_for_all_particles():
    """
    @brief Verify every particle receives the global-best location.
    """

    config = MockPSOConfig(
        PSO_type="gbest",
        N_particles=4,
        N_local=3
    )

    get_social_best = make_get_social_best(config)

    pbest = np.array([
        [5.0, 5.0],
        [1.0, 1.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ])

    fitness = np.array([
        10.0,
        1.0,
        5.0,
        7.0
    ])

    expected = np.tile([1.0, 1.0], (4, 1))

    result = get_social_best(pbest, fitness)

    assert np.allclose(result, expected)


def test_get_lbest_ring_topology():
    """
    @brief Verify odd-sized ring neighborhoods.

    Uses N_local=3, corresponding to one neighbor on each side.
    """

    config = MockPSOConfig(
        PSO_type="lbest",
        N_particles=5,
        N_local=3
    )

    get_social_best = make_get_social_best(config)

    pbest = np.array([
        [10.0, 10.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [2.0, 2.0],
        [7.0, 7.0]
    ])

    fitness = np.array([
        10.0,
        1.0,
        5.0,
        2.0,
        7.0
    ])

    expected = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0]
    ])

    result = get_social_best(pbest, fitness)

    assert np.allclose(result, expected)


def test_get_lbest_even_neighborhood():
    """
    @brief Verify even-sized neighborhood behavior.

    For N_local=4 the implementation uses:

    left = 1
    right = 2

    which should be tested explicitly.
    """

    config = MockPSOConfig(
        PSO_type="lbest",
        N_particles=6,
        N_local=4
    )

    get_social_best = make_get_social_best(config)

    pbest = np.arange(12, dtype=float).reshape((6, 2))

    fitness = np.array([
        5.,
        1.,
        6.,
        2.,
        3.,
        4.
    ])

    result = get_social_best(pbest, fitness)

    expected = np.array([
        pbest[1],
        pbest[1],
        pbest[1],
        pbest[3],
        pbest[3],
        pbest[1],
    ])

    assert np.allclose(result, expected)


def test_make_get_social_best_invalid_type():
    """
    @brief Verify unsupported PSO types raise NotImplementedError.
    """

    config = MockPSOConfig(
        PSO_type="invalid",
        N_particles=4,
        N_local=3
    )

    with pytest.raises(NotImplementedError):
        make_get_social_best(config)


def test_lbest_matches_matlab_reference():
    """
    @brief Verify lbest computation matches Matlab reference output.

    This regression test compares the Python implementation against
    reference data generated by the original Matlab implementation.
    """

    # parameters
    base_dir = Path(__file__).resolve().parent / "data"
    N_particles = 5
    N_x = 4

    # Read in the reference data
    matlab_pbest = list(np.loadtxt(base_dir / "pbest_positions_matlab_5p_4x.txt"))
    pbest = np.empty((N_particles, N_x))
    for x in range(N_x):
        for i in range(N_particles):
            pbest[i, x] = matlab_pbest.pop(0)

    fitness = np.loadtxt(base_dir / "pbest_fitness_matlab_5p_4x.txt")

    matlab_lbest = list(np.loadtxt(base_dir / "lbest_positions_matlab_5p_4x.txt"))
    expected = np.empty((N_particles, N_x))
    for x in range(N_x):
        for i in range(N_particles):
            expected[i, x] = matlab_lbest.pop(0)

    # Compute using python function
    config = MockPSOConfig(
        PSO_type="lbest",
        N_particles=N_particles,
        N_local=3
    )
    get_lbest = make_get_social_best(config)
    calculated = get_lbest(pbest, fitness)

    # compare
    for particle in range(N_particles):
        for x in range(N_x):
            assert expected[particle, x] == calculated[particle, x]


# -*- coding: utf-8 -*-
"""
@file test_let_them_fly.py
@brief Unit tests for the "let_them_fly" boundary handling routine found in

@code
PSO.options.boundary_types.let_them_fly
@endcode

The tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-08-28
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from options.boundary_types.let_them_fly import let_them_fly

def test_let_them_fly_all_particles_inside():
    """
    @brief Verify all particles remain eligible for fitness evaluation
    when inside the search space.
    """
    particles = np.array([
        [
            [0.2, 0.4],
            [0.0, 0.0],
        ],
        [
            [0.9, 0.1],
            [0.0, 0.0],
        ],
    ])

    returned_particles, mask, fitness = let_them_fly(particles)

    assert_array_equal(mask, np.array([True, True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, particles)


def test_let_them_fly_particle_outside_upper_bound():
    """
    @brief Verify particles with coordinates greater than one are
    excluded from fitness evaluation.
    """
    particles = np.array([
        [
            [0.5, 1.2],
            [0.0, 0.0],
        ]
    ])

    _, mask, fitness = let_them_fly(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])


def test_let_them_fly_particle_outside_lower_bound():
    """
    @brief Verify particles with coordinates less than zero are
    excluded from fitness evaluation.
    """
    particles = np.array([
        [
            [-0.1, 0.3],
            [0.0, 0.0],
        ]
    ])

    _, mask, _ = let_them_fly(particles)

    assert_array_equal(mask, np.array([False]))


def test_let_them_fly_mixed_population():
    """
    @brief Verify inside and outside particles are correctly identified
    within the same swarm.
    """
    particles = np.array([
        [[0.2, 0.4], [0.0, 0.0]],
        [[1.1, 0.5], [0.0, 0.0]],
        [[0.5, -0.2], [0.0, 0.0]],
        [[0.8, 0.7], [0.0, 0.0]],
    ])

    _, mask, _ = let_them_fly(particles)

    expected = np.array([True, False, False, True])

    assert_array_equal(mask, expected)


def test_let_them_fly_boundary_values_are_inside():
    """
    @brief Verify coordinates exactly on the boundaries remain inside
    the search space.

    Confirms that both 0 and 1 are treated as valid standardized
    coordinates.
    """
    particles = np.array([
        [[0.0, 1.0], [0.0, 0.0]],
    ])

    _, mask, _ = let_them_fly(particles)

    assert_array_equal(mask, np.array([True]))


def test_let_them_fly_does_not_modify_particles():
    """
    @brief Verify the particle array is returned unchanged.

    Confirms that the current boundary-handling strategy performs no
    modification of particle positions or velocities.
    """
    particles = np.random.default_rng(123).random((3, 2, 4))

    original = particles.copy()

    returned_particles, _, _ = let_them_fly(particles)

    assert_allclose(returned_particles, original)
    assert_allclose(particles, original)
    
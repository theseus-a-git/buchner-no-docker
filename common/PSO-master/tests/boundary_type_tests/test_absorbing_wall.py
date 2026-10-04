# -*- coding: utf-8 -*-
"""
@file test_absorbing_wall.py
@brief Unit tests for the "absorbing_wall" boundary handling routine found in

@code
PSO.options.boundary_types.absorbing_wall
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

from options.boundary_types.absorbing_wall import absorbing_wall

def test_absorbing_wall_particle_inside():
    """
    @brief Verify particles already inside the search space remain
    unchanged.

    Confirms that neither particle positions nor velocities are modified
    when every coordinate lies inside the standardized search space.
    """
    particles = np.array([
        [
            [0.2, 0.8],
            [1.5, -0.4],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = absorbing_wall(particles)

    assert_allclose(returned_particles, original)
    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    
def test_absorbing_wall_lower_boundary():
    """
    @brief Verify coordinates below zero are clamped to the lower
    boundary.

    Coordinates less than zero should be set to zero and the associated
    velocity components should be set to zero.
    """
    particles = np.array([
        [
            [-0.25],
            [2.0],
        ]
    ])

    returned_particles, _, _ = absorbing_wall(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.0)
    assert_allclose(returned_particles[0, 1, 0], 0.0)

def test_absorbing_wall_upper_boundary():
    """
    @brief Verify coordinates above one are clamped to the upper
    boundary.

    Coordinates greater than one should be set to one and the associated
    velocity components should be set to zero.
    """
    particles = np.array([
        [
            [1.75],
            [-3.0],
        ]
    ])

    returned_particles, _, _ = absorbing_wall(particles)

    assert_allclose(returned_particles[0, 0, 0], 1.0)
    assert_allclose(returned_particles[0, 1, 0], 0.0)

def test_absorbing_wall_only_clamps_offending_dimensions():
    """
    @brief Verify only coordinates outside the search space are modified.

    Coordinates already inside the standardized search space should
    remain unchanged, while offending coordinates are clamped and their
    associated velocity components are set to zero.
    """
    particles = np.array([
        [
            [-0.2, 0.5, 1.3],
            [2.0, 3.0, 4.0],
        ]
    ])

    returned_particles, _, _ = absorbing_wall(particles)

    expected_positions = np.array([0.0, 0.5, 1.0])
    expected_velocities = np.array([0.0, 3.0, 0.0])

    assert_allclose(returned_particles[0, 0], expected_positions)
    assert_allclose(returned_particles[0, 1], expected_velocities)

def test_absorbing_wall_mixed_population():
    """
    @brief Verify particles are handled independently.

    Particles inside the search space should remain unchanged while
    particles outside should be clamped to the nearest boundary.
    """
    particles = np.array([
        [[0.3, 0.6], [1.0, 2.0]],
        [[-0.4, 0.5], [3.0, 4.0]],
        [[0.4, 1.7], [5.0, 6.0]],
    ])

    returned_particles, mask, fitness = absorbing_wall(particles)

    expected_positions = np.array([
        [0.3, 0.6],
        [0.0, 0.5],
        [0.4, 1.0],
    ])

    expected_velocities = np.array([
        [1.0, 2.0],
        [0.0, 4.0],
        [5.0, 0.0],
    ])

    assert_allclose(returned_particles[:, 0], expected_positions)
    assert_allclose(returned_particles[:, 1], expected_velocities)
    assert_array_equal(mask, np.array([True, True, True]))
    assert np.all(np.isinf(fitness))

def test_absorbing_wall_boundary_values_are_unchanged():
    """
    @brief Verify coordinates exactly on the boundaries remain
    unchanged.

    Confirms that both zero and one are treated as valid coordinates and
    do not trigger velocity modification.
    """
    particles = np.array([
        [
            [0.0, 1.0],
            [2.0, -3.0],
        ]
    ])

    original = particles.copy()

    returned_particles, _, _ = absorbing_wall(particles)

    assert_allclose(returned_particles, original)

def test_absorbing_wall_fitness_initialized_to_inf():
    """
    @brief Verify fitness values are initialized to infinity.

    Every particle should be marked for fitness evaluation while the
    returned fitness array is initialized to infinity.
    """
    particles = np.random.default_rng(123).random((6, 2, 4))

    _, mask, fitness = absorbing_wall(particles)

    assert_array_equal(mask, np.ones(6, dtype=bool))
    assert fitness.shape == (6,)
    assert np.all(np.isinf(fitness))    

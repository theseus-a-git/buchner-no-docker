# -*- coding: utf-8 -*-
"""
@file test_reflecting_wall.py
@brief Unit tests for the "reflecting_wall" boundary handling routine found in

@code
PSO.options.boundary_types.reflecting_wall
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

from options.boundary_types.reflecting_wall import reflecting_wall

def test_reflecting_wall_particle_already_inside():
    """
    @brief Verify particles already inside the search space remain
    unchanged.

    Confirms that both particle positions and velocities are preserved
    when no reflection is required.
    """
    particles = np.array([
        [
            [0.2, 0.8],
            [0.3, -0.4],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = reflecting_wall(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, original)


def test_reflecting_wall_lower_boundary():
    """
    @brief Verify particles crossing the lower boundary are reflected
    back into the search space.

    Confirms that both the position and velocity are reflected.
    """
    particles = np.array([
        [
            [-0.25],
            [0.7],
        ]
    ])

    returned_particles, mask, _ = reflecting_wall(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles[0, 0, 0], 0.25)
    assert_allclose(returned_particles[0, 1, 0], -0.7)
    
    
def test_reflecting_wall_upper_boundary():
    """
    @brief Verify particles crossing the upper boundary are reflected
    back into the search space.

    Confirms that both the position and velocity are reflected.
    """
    particles = np.array([
        [
            [1.30],
            [-0.5],
        ]
    ])

    returned_particles, _, _ = reflecting_wall(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.70)
    assert_allclose(returned_particles[0, 1, 0], 0.5)
    
    
def test_reflecting_wall_even_number_of_reflections():
    """
    @brief Verify particles requiring multiple reflections are correctly
    mapped back into the search space.

    A coordinate lying farther than one domain width outside the search
    space should continue reflecting until it lies inside the unit
    interval.
    """
    particles = np.array([
        [
            [2.6],
            [1.0],
        ]
    ])

    returned_particles, _, _ = reflecting_wall(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.6)
    assert_allclose(returned_particles[0, 1, 0], 1.0)
    
    
def test_reflecting_wall_odd_number_of_reflections():
    """
    @brief Verify particles requiring multiple reflections are correctly
    mapped back into the search space.

    A coordinate lying farther than one domain width outside the search
    space should continue reflecting until it lies inside the unit
    interval.
    """
    particles = np.array([
        [
            [3.4],
            [1.0],
        ]
    ])

    returned_particles, _, _ = reflecting_wall(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.6)
    assert_allclose(returned_particles[0, 1, 0], -1.0)
    
    
def test_reflecting_wall_mixed_population():
    """
    @brief Verify multiple particles are reflected independently.

    Confirms that each particle coordinate is processed separately and
    particles already inside the search space remain unchanged.
    """
    particles = np.array([
        [[0.2],  [0.1]],
        [[-0.3], [0.4]],
        [[1.2],  [-0.5]],
    ])

    returned_particles, mask, _ = reflecting_wall(particles)

    expected_positions = np.array([0.2, 0.3, 0.8])
    expected_velocities = np.array([0.1, -0.4, 0.5])

    assert_array_equal(mask, np.array([True, True, True]))
    assert_allclose(returned_particles[:, 0, 0], expected_positions)
    assert_allclose(returned_particles[:, 1, 0], expected_velocities)
    
    
def test_reflecting_wall_boundary_values_unchanged():
    """
    @brief Verify particles exactly on the search-space boundary are not
    modified.

    Coordinates equal to 0 or 1 should not be reflected.
    """
    particles = np.array([
        [
            [0.0, 1.0],
            [0.2, -0.3],
        ]
    ])

    original = particles.copy()

    returned_particles, _, _ = reflecting_wall(particles)

    assert_allclose(returned_particles, original)
    
    
def test_reflecting_wall_fitness_initialized_to_inf():
    """
    @brief Verify fitness values are initialized to infinity.

    The reflecting-wall boundary condition always requests fitness
    evaluation for every particle, so these values are expected to be
    overwritten later.
    """
    particles = np.random.default_rng(42).random((5, 2, 3))

    _, mask, fitness = reflecting_wall(particles)

    assert_array_equal(mask, np.ones(5, dtype=bool))
    assert fitness.shape == (5,)
    assert np.all(np.isinf(fitness))
    
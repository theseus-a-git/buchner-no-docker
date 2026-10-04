# -*- coding: utf-8 -*-
"""
@file test_custom.py
@brief Unit tests for the "custom" boundary handling routine found in

@code
PSO.options.boundary_types.custom
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

from options.boundary_types.custom import custom

def test_custom_all_particles_marked_for_fitness():
    """
    @brief Verify all particles are marked for fitness evaluation.

    The custom boundary handler intentionally performs no boundary
    checking. All particles should therefore be marked for fitness
    computation regardless of whether they lie inside or outside the
    standardized search space.
    """
    particles = np.array([
        [[0.2, 0.8], [0.0, 0.0]],
        [[1.3, 0.4], [0.0, 0.0]],
        [[-0.5, 0.7], [0.0, 0.0]],
    ])

    _, mask, fitness = custom(particles)

    assert_array_equal(mask, np.array([True, True, True]))
    assert np.all(np.isinf(fitness))


def test_custom_returns_particles_unmodified():
    """
    @brief Verify the custom handler does not modify particle data.

    Confirms that particle positions and velocities are returned exactly
    as provided.
    """
    particles = np.random.default_rng(321).standard_normal((5, 2, 3))

    original = particles.copy()

    returned_particles, _, _ = custom(particles)

    assert_allclose(returned_particles, original)
    assert_allclose(particles, original)


def test_custom_boundary_values_are_ignored():
    """
    @brief Verify boundary values are ignored by the custom handler.

    The custom boundary handler performs no search-space validation.
    Particles exactly on the boundary or outside the search space should
    all remain eligible for fitness evaluation.
    """
    particles = np.array([
        [[0.0, 1.0], [0.0, 0.0]],
        [[1.2, -0.3], [0.0, 0.0]],
    ])

    _, mask, fitness = custom(particles)

    assert_array_equal(mask, np.array([True, True]))
    assert np.all(np.isinf(fitness))


def test_custom_fitness_initialized_to_inf():
    """
    @brief Verify fitness values are initialized to infinity.

    The custom handler initializes all fitness values to infinity. These
    values are expected to be overwritten later because every particle is
    marked for fitness evaluation.
    """
    particles = np.zeros((4, 2, 2))

    _, _, fitness = custom(particles)

    assert fitness.shape == (4,)
    assert np.all(np.isinf(fitness))
    
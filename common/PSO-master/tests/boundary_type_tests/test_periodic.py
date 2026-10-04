# -*- coding: utf-8 -*-
"""
@file test_periodic.py

@brief Unit tests for the "periodic" boundary handling routine found in

@code
PSO.options.boundary_types.periodic
@endcode

The tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-08-29
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from options.boundary_types.periodic import periodic

def test_periodic_particle_already_inside():
    """
    @brief Verify particles already inside the search space remain
    unchanged.

    Confirms that both particle positions and velocities are preserved
    when no periodic wrapping is required.
    """
    particles = np.array([
        [
            [0.2, 0.8],
            [0.3, -0.4],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = periodic(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, original)


def test_periodic_lower_boundary():
    """
    @brief Verify particles crossing the lower boundary wrap around
    to the upper portion of the search space.

    Confirms that a negative position is wrapped into the [0,1)
    interval and that its velocity is unchanged.
    """
    particles = np.array([
        [
            [-0.25],
            [0.7],
        ]
    ])

    returned_particles, mask, _ = periodic(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles[0, 0, 0], 0.75)
    assert_allclose(returned_particles[0, 1, 0], 0.7)


def test_periodic_upper_boundary():
    """
    @brief Verify particles crossing the upper boundary wrap around
    to the lower portion of the search space.

    Confirms that a position greater than 1 is wrapped into the [0,1)
    interval and that its velocity is unchanged.
    """
    particles = np.array([
        [
            [1.30],
            [-0.5],
        ]
    ])

    returned_particles, _, _ = periodic(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.30)
    assert_allclose(returned_particles[0, 1, 0], -0.5)


def test_periodic_multiple_positive_wraps():
    """
    @brief Verify particles requiring multiple positive wraps are
    correctly mapped back into the search space.

    A coordinate several domain widths above the upper boundary should
    be wrapped periodically until it lies in the [0,1) interval.
    """
    particles = np.array([
        [
            [3.6],
            [1.0],
        ]
    ])

    returned_particles, _, _ = periodic(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.6)
    assert_allclose(returned_particles[0, 1, 0], 1.0)


def test_periodic_multiple_negative_wraps():
    """
    @brief Verify particles requiring multiple negative wraps are
    correctly mapped back into the search space.

    A coordinate several domain widths below the lower boundary should
    be wrapped periodically until it lies in the [0,1) interval.
    """
    particles = np.array([
        [
            [-2.4],
            [-1.0],
        ]
    ])

    returned_particles, _, _ = periodic(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.6)
    assert_allclose(returned_particles[0, 1, 0], -1.0)


def test_periodic_mixed_population():
    """
    @brief Verify multiple particles are wrapped independently.

    Confirms that each particle coordinate is processed separately,
    particles already inside the search space remain unchanged, and
    velocities are preserved.
    """
    particles = np.array([
        [[0.2],  [0.1]],
        [[-0.3], [0.4]],
        [[1.2],  [-0.5]],
    ])

    returned_particles, mask, _ = periodic(particles)

    expected_positions = np.array([0.2, 0.7, 0.2])
    expected_velocities = np.array([0.1, 0.4, -0.5])

    assert_array_equal(mask, np.array([True, True, True]))
    assert_allclose(returned_particles[:, 0, 0], expected_positions)
    assert_allclose(returned_particles[:, 1, 0], expected_velocities)


def test_periodic_boundary_values():
    """
    @brief Verify periodic behavior at the search-space boundaries.

    The lower boundary, 0, remains 0. The upper boundary, 1, is
    equivalent to the lower boundary in a periodic search space and
    therefore wraps to 0.
    """
    particles = np.array([
        [
            [0.0, 1.0],
            [0.2, -0.3],
        ]
    ])

    returned_particles, _, _ = periodic(particles)

    assert_allclose(returned_particles[0, 0, 0], 0.0)
    assert_allclose(returned_particles[0, 0, 1], 0.0)

    # Periodic wrapping does not modify velocity.
    assert_allclose(returned_particles[0, 1, 0], 0.2)
    assert_allclose(returned_particles[0, 1, 1], -0.3)


def test_periodic_velocity_unchanged():
    """
    @brief Verify velocity components are never modified by periodic
    boundary handling.

    Unlike reflecting-wall boundary handling, periodic wrapping does
    not reverse or otherwise modify particle velocities.
    """
    particles = np.array([
        [[-0.2, 1.2, 2.7, -3.4], [0.5, -0.6, 1.2, -1.3]]
    ])

    original_velocities = particles[:, 1, :].copy()

    returned_particles, _, _ = periodic(particles)

    assert_allclose(
        returned_particles[:, 1, :],
        original_velocities,
    )


def test_periodic_multiple_dimensions():
    """
    @brief Verify periodic wrapping is applied independently to each
    coordinate of a multidimensional particle.

    Each coordinate should be wrapped independently while preserving
    its corresponding velocity component.
    """
    particles = np.array([
        [
            [-0.25, 0.25, 1.25, 2.25],
            [0.1, -0.2, 0.3, -0.4],
        ]
    ])

    returned_particles, mask, _ = periodic(particles)

    expected_positions = np.array([0.75, 0.25, 0.25, 0.25])
    expected_velocities = np.array([0.1, -0.2, 0.3, -0.4])

    assert_array_equal(mask, np.array([True]))
    assert_allclose(
        returned_particles[0, 0, :],
        expected_positions,
    )
    assert_allclose(
        returned_particles[0, 1, :],
        expected_velocities,
    )


def test_periodic_fitness_initialized_to_inf():
    """
    @brief Verify fitness values are initialized to infinity.

    The periodic boundary condition always requests fitness evaluation
    for every particle, so these values are expected to be overwritten
    later.
    """
    particles = np.random.default_rng(42).random((5, 2, 3))

    _, mask, fitness = periodic(particles)

    assert_array_equal(mask, np.ones(5, dtype=bool))
    assert fitness.shape == (5,)
    assert np.all(np.isinf(fitness))


def test_periodic_positions_inside_search_space():
    """
    @brief Verify all wrapped positions lie within the periodic
    search-space interval.

    Since the periodic implementation represents the upper boundary
    as the lower boundary, all resulting coordinates must satisfy
    0 <= x < 1.
    """
    particles = np.array([
        [[-10.5, -2.1, 0.0, 0.5, 1.0, 3.7, 10.2],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    ])

    returned_particles, _, _ = periodic(particles)

    positions = returned_particles[:, 0, :]

    assert np.all(positions >= 0.0)
    assert np.all(positions < 1.0)

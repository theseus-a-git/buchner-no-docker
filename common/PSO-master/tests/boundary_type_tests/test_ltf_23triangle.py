# -*- coding: utf-8 -*-
"""
@file test_ltf_23triangle.py
@brief Unit tests for the "ltf_23triangle" boundary handling routine found in

@code
PSO.options.boundary_types.ltf_23triangle
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

from options.boundary_types.ltf_23triangle import LTF_23Triangle

def test_ltf_23triangle_particle_inside_triangle():
    """
    @brief Verify a particle inside both the unit hypercube and the
    triangular subspace remains eligible for fitness evaluation.

    The particle satisfies:

    @code
    x[2] >= x[3]
    @endcode

    which corresponds to the allowed triangular half of the standardized
    square.
    """
    particles = np.array([
        [
            [0.2, 0.5, 0.8, 0.3],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])

    returned_particles, mask, fitness = LTF_23Triangle(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, particles)


def test_ltf_23triangle_particle_outside_triangle():
    """
    @brief Verify a particle outside the triangular subspace is excluded
    from fitness evaluation.

    The particle remains inside the standardized hypercube but violates
    the triangle constraint.
    """
    particles = np.array([
        [
            [0.2, 0.5, 0.3, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, fitness = LTF_23Triangle(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])


def test_ltf_23triangle_particle_outside_hypercube():
    """
    @brief Verify particles outside the standardized search space are
    excluded regardless of the triangle constraint.
    """
    particles = np.array([
        [
            [1.2, 0.5, 0.8, 0.3],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, _ = LTF_23Triangle(particles)

    assert_array_equal(mask, np.array([False]))


def test_ltf_23triangle_boundary_line_is_inside():
    """
    @brief Verify particles exactly on the triangle boundary remain
    inside the admissible region.

    The boundary corresponds to

    @code
    x[2] == x[3]
    @endcode
    """
    particles = np.array([
        [
            [0.1, 0.9, 0.6, 0.6],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, _ = LTF_23Triangle(particles)

    assert_array_equal(mask, np.array([True]))


def test_ltf_23triangle_mixed_population():
    """
    @brief Verify particles are correctly classified when the swarm
    contains particles inside the triangle, outside the triangle, and
    outside the standardized search space.
    """
    particles = np.array([
        [[0.2, 0.3, 0.9, 0.4], [0.0, 0.0, 0.0, 0.0]],   # inside
        [[0.2, 0.3, 0.2, 0.7], [0.0, 0.0, 0.0, 0.0]],   # outside triangle
        [[1.1, 0.3, 0.9, 0.4], [0.0, 0.0, 0.0, 0.0]],   # outside cube
        [[0.4, 0.5, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0]],   # boundary
    ])

    _, mask, _ = LTF_23Triangle(particles)

    expected = np.array([True, False, False, True])

    assert_array_equal(mask, expected)


def test_ltf_23triangle_does_not_modify_particles():
    """
    @brief Verify the triangle boundary handler does not modify the
    particle positions or velocities.

    Confirms the current implementation only performs boundary
    classification.
    """
    particles = np.random.default_rng(456).random((6, 2, 5))

    original = particles.copy()

    returned_particles, _, _ = LTF_23Triangle(particles)

    assert_allclose(returned_particles, original)
    assert_allclose(particles, original)


def test_ltf_23triangle_fitness_initialized_to_inf():
    """
    @brief Verify fitness values are initialized to infinity.

    All entries should be initialized to infinity regardless of whether
    particles remain inside the admissible search region.
    """
    particles = np.random.default_rng(789).random((5, 2, 4))

    _, _, fitness = LTF_23Triangle(particles)

    assert fitness.shape == (5,)
    assert np.all(np.isinf(fitness))
    
    
def test_ltf_23triangle_zero_denominator():
    """
    @brief Verify particles on the edge x[3]=0 are handled correctly.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, _ = LTF_23Triangle(particles)

    assert_array_equal(mask, np.array([True]))

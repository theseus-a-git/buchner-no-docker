# -*- coding: utf-8 -*-
"""
@file test_ltf_23triangle_refl.py
@brief Unit tests for the "ltf_23triangle_refl" boundary handling routine found in

@code
PSO.options.boundary_types.ltf_23triangle_refl
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

from options.boundary_types.ltf_23triangle_refl import LTF_23Triangle_Refl

def test_ltf_23triangle_refl_particle_already_inside():
    """
    @brief Verify particles already inside the admissible triangular
    region remain unchanged.

    Particles satisfying

    @code
    x[2] >= x[3]
    @endcode

    should not have their positions or velocities modified.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = LTF_23Triangle_Refl(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, original)


def test_ltf_23triangle_refl_reflects_position_and_velocity():
    """
    @brief Verify particles entering the forbidden triangular region are
    reflected across the x=y line.

    Both the position and corresponding velocity components should be
    exchanged.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.3, 0.8],
            [1.0, 2.0, 30.0, 40.0],
        ]
    ])

    returned_particles, mask, _ = LTF_23Triangle_Refl(particles)

    assert_array_equal(mask, np.array([True]))

    assert_allclose(returned_particles[0,0,2], 0.8)
    assert_allclose(returned_particles[0,0,3], 0.3)

    assert_allclose(returned_particles[0,1,2], 40.0)
    assert_allclose(returned_particles[0,1,3], 30.0)


def test_ltf_23triangle_refl_outside_hypercube_still_reflects_triangle():
    """
    @brief Verify particles outside the hypercube are excluded from
    fitness evaluation while still being reflected into the canonical
    triangular ordering.
    """
    particles = np.array([
        [
            [1.2, 0.4, 0.2, 0.7],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ])

    returned_particles, mask, fitness = LTF_23Triangle_Refl(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])

    assert_allclose(returned_particles[0,0,2], 0.7)
    assert_allclose(returned_particles[0,0,3], 0.2)

    assert_allclose(returned_particles[0,1,2], 4.0)
    assert_allclose(returned_particles[0,1,3], 3.0)


def test_ltf_23triangle_refl_boundary_line_unchanged():
    """
    @brief Verify particles exactly on the triangular boundary are not
    modified.

    The boundary corresponds to

    @code
    x[2] == x[3]
    @endcode
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.6, 0.6],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = LTF_23Triangle_Refl(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles, original)

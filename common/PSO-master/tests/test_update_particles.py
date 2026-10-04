# -*- coding: utf-8 -*-
"""
@file test_update_particles.py
@brief Unit tests for the PSO particle-update kernel.

This module contains unit tests for the particle-update factory function
defined in:

@code
PSO.core.update_particles
@endcode

The tests verify:

- Correct particle-array dimensions
- Correct velocity updates
- Correct position updates
- Proper velocity clamping
- Correct inertia-weight scheduling
- Correct use of particle-specific random vectors
- Preservation of the input particle array
- MATLAB cross-validation against reference data

The tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-06-30
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from numba import njit

from core.update_particles import make_update_particles
from core.dynamical_equations import make_update_velocity


# --------------------------------------------------------------------------
# Mock configuration
# --------------------------------------------------------------------------

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight PSO configuration used for testing.

    Members:
    - N_x
    - N_iterations
    - N_particles
    - clamp_velocity
    - vmax
    - C1
    - C2
    - eq_type
    """

    N_x: int
    N_iterations: int
    N_particles: int
    clamp_velocity: bool
    vmax: float
    C1: float = 2.0
    C2: float = 2.0
    eq_type: str = "inertia"


# --------------------------------------------------------------------------
# Deterministic velocity kernel
# --------------------------------------------------------------------------

@njit
def simple_velocity_update(
    w,
    prev_x,
    prev_v,
    R1,
    R2,
    pbest,
    lbest,
):
    """
    @brief Deterministic velocity-update kernel used for testing.
    """

    return (
        w * prev_v
        + 2.0 * R1 * (pbest - prev_x)
        + 2.0 * R2 * (lbest - prev_x)
    )


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------

def build_config(
    *,
    N_particles=2,
    N_x=2,
    N_iterations=10,
    clamp_velocity=False,
    vmax=1.0,
):
    return MockPSOConfig(
        N_x=N_x,
        N_iterations=N_iterations,
        N_particles=N_particles,
        clamp_velocity=clamp_velocity,
        vmax=vmax,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_particle_tensor_shape():
    """
    @brief Verify updated particle tensor preserves shape.
    """

    config = build_config()

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.zeros((2, 2, 3))

    out = update_particles(
        particles,
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        1,
    )

    assert out.shape == particles.shape


def test_velocity_update():
    """
    @brief Verify velocity updates match the analytical expression.
    """

    config = build_config()

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.array(
        [
            [[1., 2.], [0.5, 0.5]],
            [[3., 4.], [1.0, 1.0]],
        ]
    )

    pbest = np.array([[2., 3.], [4., 5.]])
    lbest = np.array([[3., 4.], [5., 6.]])

    R1 = np.ones((2, 2))
    R2 = np.ones((2, 2))

    out = update_particles(
        particles,
        pbest,
        lbest,
        R1,
        R2,
        1,
    )

    w = 0.9

    expected = (
        w * particles[:, 1]
        + 2 * (pbest - particles[:, 0])
        + 2 * (lbest - particles[:, 0])
    )

    assert_allclose(out[:, 1], expected)


def test_position_update():
    """
    @brief Verify positions are updated using the newly computed
    velocities.
    """

    config = build_config()

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.array(
        [
            [[1., 1.], [1., 1.]],
            [[2., 2.], [2., 2.]],
        ]
    )

    zeros = np.zeros((2, 2))

    out = update_particles(
        particles,
        particles[:, 0],
        particles[:, 0],
        zeros,
        zeros,
        1,
    )

    expected_v = 0.9 * particles[:, 1]
    expected_x = particles[:, 0] + expected_v

    assert_allclose(out[:, 1], expected_v)
    assert_allclose(out[:, 0], expected_x)


def test_velocity_clamping():
    """
    @brief Verify velocities are clamped to +/- vmax.
    """

    config = build_config(
        clamp_velocity=True,
        vmax=0.5,
    )

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.zeros((2, 2, 2))

    pbest = np.full((2, 2), 10.0)
    lbest = np.full((2, 2), -10.0)

    R1 = np.ones((2, 2))
    R2 = np.ones((2, 2))

    out = update_particles(
        particles,
        pbest,
        lbest,
        R1,
        R2,
        1,
    )

    assert np.all(out[:, 1] <= 0.5)
    assert np.all(out[:, 1] >= -0.5)


def test_particle_specific_random_vectors():
    """
    @brief Verify each particle uses its own random vectors.
    """

    config = build_config()

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.zeros((2, 2, 2))

    pbest = np.ones((2, 2))
    lbest = np.ones((2, 2))

    R1 = np.array([[1., 1.], [2., 2.]])
    R2 = np.array([[3., 3.], [4., 4.]])

    out = update_particles(
        particles,
        pbest,
        lbest,
        R1,
        R2,
        1,
    )

    assert not np.allclose(out[0, 1], out[1, 1])


def test_input_particles_not_modified():
    """
    @brief Verify the input particle array is not modified in-place.
    """

    config = build_config()

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.random.rand(2, 2, 2)
    original = particles.copy()

    update_particles(
        particles,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        1,
    )

    assert_allclose(particles, original)


def test_single_iteration_uses_w_point_four():
    """
    @brief Verify the special case N_iterations==1 uses w=0.4.
    """

    config = build_config(
        N_iterations=1,
    )

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.array([[[0.0], [1.0]], [[0.0], [2.0]]])

    zeros = np.zeros((2, 1))

    out = update_particles(
        particles,
        zeros,
        zeros,
        zeros,
        zeros,
        1,
    )

    assert_allclose(out[:, 1, 0], [0.4, 0.8])


def test_final_iteration_uses_w_point_four():
    """
    @brief Verify the final iteration uses the minimum inertia weight.
    """

    config = build_config(
        N_iterations=11,
    )

    update_particles = make_update_particles(
        config,
        simple_velocity_update,
    )

    particles = np.array([[[0.0], [1.0]], [[0.0], [2.0]]])

    zeros = np.zeros((2, 1))

    out = update_particles(
        particles,
        zeros,
        zeros,
        zeros,
        zeros,
        11,
    )

    assert_allclose(out[:, 1, 0], [0.4, 0.8])


def test_matlab_cross_validation():
    """
    @brief Verify one particle-update iteration matches MATLAB reference
    data exactly.
    """

    base_dir = Path(__file__).resolve().parent / "data"

    N_particles = 5
    N_x = 4

    config = MockPSOConfig(
        N_x=N_x,
        N_iterations=2,
        N_particles=N_particles,
        clamp_velocity=True,
        vmax=0.5,
    )

    def read_particle_file(filename):
        vals = list(np.loadtxt(base_dir / filename))
        arr = np.empty((N_particles, 2, N_x))
        for j in range(2):
            for x in range(N_x):
                for i in range(N_particles):
                    arr[i, j, x] = vals.pop(0)
        return arr

    def read_position_file(filename):
        vals = list(np.loadtxt(base_dir / filename))
        arr = np.empty((N_particles, N_x))
        for x in range(N_x):
            for i in range(N_particles):
                arr[i, x] = vals.pop(0)
        return arr

    particles_before = read_particle_file(
        "Initialized_particles_matlab_5p_4x.txt"
    )

    particles_after = read_particle_file(
        "Iterated_particles_matlab_5p_4x.txt"
    )

    pbest = read_position_file(
        "pbest_positions_matlab_5p_4x.txt"
    )

    lbest = read_position_file(
        "lbest_positions_matlab_5p_4x.txt"
    )

    R1 = np.loadtxt(base_dir / "R1_vals_5p_4x.txt").reshape(
        N_particles,
        N_x,
    )

    R2 = np.loadtxt(base_dir / "R2_vals_5p_4x.txt").reshape(
        N_particles,
        N_x,
    )

    update_velocity = make_update_velocity(config)

    update_particles = make_update_particles(
        config,
        update_velocity,
    )

    calculated = update_particles(
        particles_before,
        pbest,
        lbest,
        R1,
        R2,
        1,
    )

    # compare against matlab
    for j in range(2):
        for x in range(N_x):
            for i in range(N_particles):
                assert calculated[i, j, x] == particles_after[i, j, x]



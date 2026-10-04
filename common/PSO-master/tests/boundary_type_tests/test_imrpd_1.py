# -*- coding: utf-8 -*-
"""
@file test_imrpd_1.py
@brief Unit tests for the "IMRPD-1" boundary handling routine found in

@code
PSO.options.boundary_types.imrpd_1
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

from options.boundary_types.imrpd_1 import IMRPD_1

def test_imrpd_1_particle_inside_all_constraints():
    """
    @brief Verify a particle satisfying all IMRPD-1 constraints remains
    eligible for fitness evaluation and is not modified.

    The particle satisfies:

    @code
    0 <= x[k] <= 1
    x[2] >= x[3]
    |x[4] - 0.5| + |x[5] - 0.5| <= 0.5
    @endcode

    No boundary handling should be required for this particle.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, 0.7, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.all(np.isinf(fitness))
    assert_allclose(returned_particles, original)
    assert_allclose(particles, original)


def test_imrpd_1_reflects_mass_position_and_velocity():
    """
    @brief Verify the mass coordinates and velocities are exchanged when
    the particle enters the forbidden triangular half of the mass
    subspace.

    A particle satisfying

    @code
    x[2] < x[3]
    @endcode

    should have dimensions 2 and 3 exchanged for both position and
    velocity.

    The particle should remain eligible for fitness evaluation after
    this reflection.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.3, 0.8, 0.6, 0.4],
            [1.0, 2.0, 30.0, 40.0, 5.0, 6.0],
        ]
    ])

    returned_particles, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert np.isinf(fitness[0])

    assert_allclose(returned_particles[0, 0, 2:4], [0.8, 0.3])
    assert_allclose(returned_particles[0, 1, 2:4], [40.0, 30.0])

    # Other dimensions should remain unchanged.
    assert_allclose(returned_particles[0, 0, [0, 1, 4, 5]],
                    [0.2, 0.4, 0.6, 0.4])
    assert_allclose(returned_particles[0, 1, [0, 1, 4, 5]],
                    [1.0, 2.0, 5.0, 6.0])


def test_imrpd_1_mass_boundary_line_is_unchanged():
    """
    @brief Verify a particle exactly on the mass ordering boundary is
    not modified.

    The mass boundary corresponds to

    @code
    x[2] == x[3]
    @endcode

    Equality should not trigger the reflection operation.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.6, 0.6, 0.5, 0.5],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles, original)
    assert_allclose(particles, original)


def test_imrpd_1_rotated_square_center_is_inside():
    """
    @brief Verify the center of the rotated square is inside the
    admissible spin region.

    At the center,

    @code
    x[4] = x[5] = 0.5
    @endcode

    and therefore the diamond constraint is satisfied.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles, original)


@pytest.mark.parametrize(
    "spin_position",
    [
        [0.0, 0.5],
        [0.5, 1.0],
        [1.0, 0.5],
        [0.5, 0.0],
    ],
)
def test_imrpd_1_rotated_square_vertices_are_inside(spin_position):
    """
    @brief Verify all four vertices of the rotated square are treated as
    valid boundary points.

    The admissible spin region has vertices at

    @code
    (0, 0.5)
    (0.5, 1)
    (1, 0.5)
    (0.5, 0)
    @endcode

    Points exactly on these boundaries should remain eligible for fitness
    evaluation.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, spin_position[0], spin_position[1]],
            [0.0, 0.0, 0.8, 0.3, 1.0, -1.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles, original)


@pytest.mark.parametrize(
    "spin_position",
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.1, 0.1],
        [0.9, 0.9],
    ],
)
def test_imrpd_1_rotated_square_outside(spin_position):
    """
    @brief Verify points outside the rotated square are excluded from
    fitness evaluation.

    The diamond constraint is

    @code
    |x[4] - 0.5| + |x[5] - 0.5| <= 0.5
    @endcode

    Each parameterized point violates this condition while remaining
    inside the unit hypercube.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, spin_position[0], spin_position[1]],
            [0.0, 0.0, 0.8, 0.3, 1.0, -1.0],
        ]
    ])

    _, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])


def test_imrpd_1_outside_lower_hypercube():
    """
    @brief Verify a particle with a coordinate below zero is excluded
    from fitness evaluation.

    The particle otherwise satisfies the IMRPD-1 mass and spin
    constraints.
    """
    particles = np.array([
        [
            [-0.1, 0.4, 0.8, 0.3, 0.6, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])


def test_imrpd_1_outside_upper_hypercube():
    """
    @brief Verify a particle with a coordinate above one is excluded
    from fitness evaluation.

    The particle otherwise satisfies the IMRPD-1 mass and spin
    constraints.
    """
    particles = np.array([
        [
            [0.2, 1.1, 0.8, 0.3, 0.6, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ])

    _, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])


def test_imrpd_1_unit_hypercube_boundaries_are_inside():
    """
    @brief Verify coordinates exactly equal to zero or one are treated
    as valid hypercube boundaries.

    The mass and spin coordinates are selected so the particle satisfies
    the additional IMRPD-1 constraints.
    """
    particles = np.array([
        [
            [0.0, 1.0, 1.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))
    assert_allclose(returned_particles, original)


def test_imrpd_1_outside_spin_region_does_not_modify_particle():
    """
    @brief Verify a particle outside the rotated spin square is not
    otherwise modified by the IMRPD-1 handler.

    In particular, only the mass ordering operation should modify
    dimensions 2 and 3. The spin dimensions should remain unchanged
    even when they cause the particle to be rejected.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, 0.1, 0.1],
            [1.0, 2.0, 30.0, 40.0, 50.0, 60.0],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])

    assert_allclose(returned_particles[0, 0, 4:6], original[0, 0, 4:6])
    assert_allclose(returned_particles[0, 1, 4:6], original[0, 1, 4:6])


def test_imrpd_1_outside_hypercube_still_reflects_mass_ordering():
    """
    @brief Verify mass ordering is canonicalized even when another
    coordinate places the particle outside the unit hypercube.

    The implementation performs the mass reflection before checking the
    hypercube constraint. Therefore dimensions 2 and 3 should still be
    exchanged even though the particle is ultimately excluded from
    fitness evaluation.
    """
    particles = np.array([
        [
            [1.2, 0.4, 0.2, 0.7, 0.6, 0.4],
            [1.0, 2.0, 30.0, 40.0, 5.0, 6.0],
        ]
    ])

    returned_particles, mask, fitness = IMRPD_1(particles)

    assert_array_equal(mask, np.array([False]))
    assert np.isinf(fitness[0])

    assert_allclose(returned_particles[0, 0, 2:4], [0.7, 0.2])
    assert_allclose(returned_particles[0, 1, 2:4], [40.0, 30.0])


def test_imrpd_1_mixed_population():
    """
    @brief Verify IMRPD-1 correctly handles a mixed particle population.

    The population contains:

    - A particle satisfying all constraints.
    - A particle requiring mass reflection.
    - A particle outside the rotated spin square.
    - A particle outside the unit hypercube.
    - A particle on the mass and spin boundaries.

    Only the particles satisfying all constraints after mass reflection
    should be marked for fitness evaluation.
    """
    particles = np.array([
        # Inside all constraints.
        [
            [0.2, 0.3, 0.9, 0.4, 0.8, 0.3],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ],

        # Mass ordering violation; should be reflected and accepted.
        [
            [0.2, 0.3, 0.2, 0.7, 0.8, 0.3],
            [0.0, 0.0, 10.0, 20.0, 3.0, 4.0],
        ],

        # Outside rotated spin square.
        [
            [0.2, 0.3, 0.9, 0.4, 0.1, 0.1],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ],

        # Outside unit hypercube.
        [
            [1.2, 0.3, 0.9, 0.4, 0.8, 0.3],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ],

        # On mass boundary and spin-square boundary.
        [
            [0.2, 0.3, 0.6, 0.6, 0.0, 0.5],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ],
    ])

    returned_particles, mask, fitness = IMRPD_1(particles)

    expected_mask = np.array([True, True, False, False, True])

    assert_array_equal(mask, expected_mask)
    assert np.all(np.isinf(fitness))

    # Verify the mass reflection occurred for the second particle.
    assert_allclose(returned_particles[1, 0, 2:4], [0.7, 0.2])
    assert_allclose(returned_particles[1, 1, 2:4], [20.0, 10.0])

    # Verify the other particles were not unexpectedly modified.
    assert_allclose(
        returned_particles[0],
        particles[0],
    )
    assert_allclose(
        returned_particles[2],
        particles[2],
    )
    assert_allclose(
        returned_particles[3],
        particles[3],
    )
    assert_allclose(
        returned_particles[4],
        particles[4],
    )


def test_imrpd_1_fitness_initialized_to_inf():
    """
    @brief Verify the IMRPD-1 handler initializes every fitness value
    to positive infinity.

    This should hold regardless of whether the corresponding particle
    passes or fails the boundary checks.
    """
    particles = np.array([
        [[0.2, 0.3, 0.9, 0.4, 0.8, 0.3], [0.0] * 6],
        [[0.2, 0.3, 0.2, 0.7, 0.1, 0.1], [0.0] * 6],
        [[1.2, 0.3, 0.9, 0.4, 0.8, 0.3], [0.0] * 6],
        [[0.2, 0.3, 0.8, 0.3, 0.5, 0.5], [0.0] * 6],
    ])

    _, mask, fitness = IMRPD_1(particles)

    assert fitness.shape == (4,)
    assert np.all(np.isinf(fitness))
    assert_array_equal(mask, np.array([True, False, False, True]))


def test_imrpd_1_does_not_change_unrelated_dimensions():
    """
    @brief Verify mass reflection changes only dimensions 2 and 3.

    When the mass ordering constraint is violated, dimensions 2 and 3
    should be exchanged along with their corresponding velocities.
    All other dimensions must remain unchanged.
    """
    particles = np.array([
        [
            [0.11, 0.22, 0.30, 0.80, 0.44, 0.55],
            [1.1, 2.2, 3.3, 8.8, 4.4, 5.5],
        ]
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True]))

    assert_allclose(returned_particles[0, 0, 0:2],
                    original[0, 0, 0:2])
    assert_allclose(returned_particles[0, 0, 2:4],
                    [0.80, 0.30])
    assert_allclose(returned_particles[0, 0, 4:6],
                    original[0, 0, 4:6])

    assert_allclose(returned_particles[0, 1, 0:2],
                    original[0, 1, 0:2])
    assert_allclose(returned_particles[0, 1, 2:4],
                    [8.8, 3.3])
    assert_allclose(returned_particles[0, 1, 4:6],
                    original[0, 1, 4:6])


def test_imrpd_1_particle_array_is_modified_only_when_mass_reflection_required():
    """
    @brief Verify the handler mutates particle data only when mass
    canonicalization is required.

    This test compares a particle that already satisfies the mass ordering
    against a particle that violates it.
    """
    particles = np.array([
        [
            [0.2, 0.4, 0.8, 0.3, 0.6, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        [
            [0.2, 0.4, 0.3, 0.8, 0.6, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
    ])

    original = particles.copy()

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True, True]))

    # First particle already had canonical mass ordering.
    assert_allclose(returned_particles[0], original[0])

    # Second particle should have dimensions 2 and 3 exchanged.
    assert_allclose(returned_particles[1, 0, 2:4], [0.8, 0.3])
    assert_allclose(returned_particles[1, 1, 2:4], [4.0, 3.0])

    # All non-mass dimensions remain unchanged.
    assert_allclose(
        returned_particles[1, :, [0, 1, 4, 5]],
        original[1, :, [0, 1, 4, 5]],
    )


def test_imrpd_1_multiple_particles_with_mass_reflection():
    """
    @brief Verify mass reflection is performed independently for each
    particle in a swarm.

    Particles requiring reflection should have their mass coordinates and
    velocities exchanged independently without affecting other particles.
    """
    particles = np.array([
        [
            [0.2, 0.3, 0.1, 0.9, 0.6, 0.4],
            [0.0, 0.0, 10.0, 20.0, 1.0, 2.0],
        ],
        [
            [0.2, 0.3, 0.8, 0.2, 0.6, 0.4],
            [0.0, 0.0, 30.0, 40.0, 3.0, 4.0],
        ],
        [
            [0.2, 0.3, 0.7, 0.7, 0.6, 0.4],
            [0.0, 0.0, 50.0, 60.0, 5.0, 6.0],
        ],
    ])

    returned_particles, mask, _ = IMRPD_1(particles)

    assert_array_equal(mask, np.array([True, True, True]))

    assert_allclose(returned_particles[0, 0, 2:4], [0.9, 0.1])
    assert_allclose(returned_particles[0, 1, 2:4], [20.0, 10.0])

    assert_allclose(returned_particles[1, 0, 2:4], [0.8, 0.2])
    assert_allclose(returned_particles[1, 1, 2:4], [30.0, 40.0])

    assert_allclose(returned_particles[2, 0, 2:4], [0.7, 0.7])
    assert_allclose(returned_particles[2, 1, 2:4], [50.0, 60.0])

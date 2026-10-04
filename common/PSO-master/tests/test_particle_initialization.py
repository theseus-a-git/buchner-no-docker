# -*- coding: utf-8 -*-
"""
@file test_particle_initialization.py
@brief Unit tests for particle initialization in Particle Swarm Optimization.

This module contains unit tests for the
`initialize_particles` function defined in:

@code
PSO.core.particle_initialization
@endcode

The tests verify:

- Correct particle array dimensions
- Proper initialization of particle positions
- Proper initialization of particle velocities
- Correct application of velocity clamping
- Correct behavior when velocity clamping is disabled
- Correct row-major ordering of generated values
- Correct handling of box topology
- Correct transformation for spherical-surface topology
- Correct handling of triangular initialization
- Correct handling of IMRPD-1 spin transformations
- Correct use of the run index for premade values
- Regression against Matlab reference output

The tests use deterministic mock random-number generators to ensure
repeatable and predictable results.

@author
Ben Carlson

@date
2026-09-09
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pytest

from core.particle_initialization import initialize_particles
from utils.dispatch_tables import VALUE_GENERATORS

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object for testing.

    This class provides the configuration members required by
    `initialize_particles` and the premade value generator.
    """
    space_topology: str
    boundary_type: str
    N_x: int
    N_particles: int
    clamp_initial_velocity: bool
    vmax: float
    position_init_value_type: str = "premade"
    velocity_init_value_type: str = "premade"
    position_init_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0]])
    )
    velocity_init_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0]])
    )
    update_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0]])
    )
    
    
def make_config(
        *,
        space_topology="box",
        boundary_type="let_them_fly",
        N_x=1,
        N_particles=1,
        clamp_initial_velocity=False,
        vmax=1.0,
        position_randoms=None,
        velocity_randoms=None,
        update_randoms=None,
    ):
    """
    @brief Create a deterministic mock PSO configuration.

    @param space_topology
    Search-space topology.

    @param boundary_type
    Boundary-handling or initialization type.

    @param N_x
    Number of search-space dimensions.

    @param N_particles
    Number of particles.

    @param clamp_initial_velocity
    Whether initial velocities should be clamped.

    @param vmax
    Maximum initial velocity magnitude.

    @param position_randoms
    Premade values used by the position generator.

    @param velocity_randoms
    Premade values used by the velocity generator.

    @param update_randoms
    Premade values reserved for the update generator.

    @return
    Configured `MockPSOConfig` instance.
    """
    if position_randoms is None:
        position_randoms = np.zeros((1, 1))

    if velocity_randoms is None:
        velocity_randoms = np.zeros((1, 1))

    if update_randoms is None:
        update_randoms = np.zeros((1, 1))

    return MockPSOConfig(
        space_topology=space_topology,
        boundary_type=boundary_type,
        N_x=N_x,
        N_particles=N_particles,
        clamp_initial_velocity=clamp_initial_velocity,
        vmax=vmax,
        position_init_premade_vals=np.asarray(
            position_randoms
        )[None, ...]
        if np.asarray(position_randoms).ndim == 1
        else np.asarray(position_randoms)[None, ...],
        velocity_init_premade_vals=np.asarray(
            velocity_randoms
        )[None, ...]
        if np.asarray(velocity_randoms).ndim == 1
        else np.asarray(velocity_randoms)[None, ...],
        update_premade_vals=np.asarray(
            update_randoms
        )[None, ...]
        if np.asarray(update_randoms).ndim == 1
        else np.asarray(update_randoms)[None, ...],
    )



def test_initialize_particles_shape():
    """
    @brief Verify returned particle array has the expected dimensions.

    The returned array should have shape:

    @code
    (N_particles, 2, N_x)
    @endcode
    """
    config = make_config(
        N_x=3,
        N_particles=4,
        position_randoms=np.full((4, 3), 0.5),
        velocity_randoms=np.full((4, 3), 0.5),
    )

    particles = initialize_particles(config, 0)

    assert particles.shape == (4, 2, 3)


def test_initialize_particles_positions():
    """
    @brief Verify particle positions are initialized correctly.

    Confirms that all position coordinates are assigned directly from the
    supplied random values.
    """
    position_randoms = np.full((2, 2), 0.25)

    config = make_config(
        N_x=2,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 2), 0.25),
    )

    particles = initialize_particles(config, 0)

    np.testing.assert_allclose(
        particles[:, 0, :],
        position_randoms,
    )


def test_initialize_particles_velocities_no_clamp():
    """
    @brief Verify velocity initialization when clamping is disabled.

    Velocities should equal:

    @code
    velocity_random - position
    @endcode
    """
    position_randoms = np.array([
        [0.2],
        [0.2],
    ])

    velocity_randoms = np.array([
        [0.8],
        [0.8],
    ])

    config = make_config(
        N_x=1,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.6],
        [0.6],
    ])

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected,
    )


def test_initialize_particles_velocity_clamping():
    """
    @brief Verify positive velocity clamping.

    Velocities larger than `vmax` should be limited to `vmax`.
    """
    position_randoms = np.array([
        [0.1],
        [0.1],
    ])

    velocity_randoms = np.array([
        [0.9],
        [0.9],
    ])

    config = make_config(
        N_x=1,
        N_particles=2,
        clamp_initial_velocity=True,
        vmax=0.3,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.3],
        [0.3],
    ])

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected,
    )


def test_initialize_particles_negative_velocity_clamping():
    """
    @brief Verify negative velocity clamping.

    Negative velocities whose magnitude exceeds `vmax`
    should be limited to `-vmax`.
    """
    position_randoms = np.array([
        [0.9],
    ])

    velocity_randoms = np.array([
        [0.1],
    ])

    config = make_config(
        N_x=1,
        N_particles=1,
        clamp_initial_velocity=True,
        vmax=0.2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [-0.2],
    ])

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected,
    )


def test_row_major_ordering():
    """
    @brief Verify premade values preserve row-major ordering.

    Values should be organized so that all coordinates for particle 0
    appear first, followed by all coordinates for particle 1, and so on.

    For example:

    @code
    [0, 1, 2, 3, 4, 5]
    @endcode

    should produce:

    @code
    [[0, 1],
     [2, 3],
     [4, 5]]
    @endcode
    """
    position_randoms = np.array([
        [0., 1.],
        [2., 3.],
        [4., 5.],
    ])

    config = make_config(
        N_x=2,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=np.arange(6, dtype=float).reshape(3, 2),
    )

    particles = initialize_particles(config, 0)

    np.testing.assert_allclose(
        particles[:, 0, :],
        position_randoms,
    )

    
def test_initialize_particles_spherical_surface_theta_transform():
    """
    @brief Verify theta coordinates are transformed for spherical surfaces.

    The first coordinate should be mapped according to:

    @code
    theta = arccos(2u - 1) / pi
    @endcode

    where `u` is the uniformly distributed random value on [0,1].
    """
    position_randoms = np.array([
        [0.0, 0.3],
        [0.5, 0.3],
        [1.0, 0.3],
    ])
    
    config = make_config(
        space_topology="spherical_surface",
        N_x=2,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=np.full((3, 2), 0.5),
    )
    
    particles = initialize_particles(config, 0)
    
    expected_theta = np.array([1.0, 0.5, 0.0])
    
    np.testing.assert_allclose(
        particles[:, 0, 0],
        expected_theta,
    )
    
    
def test_initialize_particles_spherical_surface_preserves_other_coordinates():
    """
    @brief Verify only the theta coordinate is transformed.

    All remaining coordinates should remain identical to the originally
    generated random values.
    """
    position_randoms = np.array([
        [0.2, 0.4, 0.1],
        [0.8, 0.6, 0.9],
    ])
    
    config = make_config(
        space_topology="spherical_surface",
        N_x=3,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 3), 0.5),
    )
    
    particles = initialize_particles(config, 0)
    
    np.testing.assert_allclose(
        particles[:, 0, 1:],
        position_randoms[:, 1:],
    )
    
    
def test_initialize_particles_box_topology_leaves_positions_unchanged():
    """
    @brief Verify box topology performs no position transformation.

    The default box topology should leave all initialized position
    coordinates unchanged.
    """

    position_randoms = np.array([
        [0.25, 0.40],
        [0.75, 0.60],
    ])

    config = make_config(
        space_topology="box",
        N_x=2,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 2), 0.5),
    )

    particles = initialize_particles(config, 0)

    np.testing.assert_allclose(
        particles[:, 0, :],
        position_randoms,
    )
    
    
@pytest.mark.parametrize(
    "boundary_type",
    ["ltf_23triangle", "ltf_23triangle_refl"],
)
def test_initialize_particles_ltf_23triangle_enforces_triangle_constraint(boundary_type,):
    """
    @brief Verify initialized particles satisfy the triangular search
    region.

    The triangle initialization should ensure

    @code
    x[2] >= x[3]
    @endcode

    for every initialized particle.
    """
    position_randoms = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.9, 0.3, 0.8],
    ])

    config = make_config(
        boundary_type=boundary_type,
        N_x=4,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=np.full((3, 4), 0.5),
    )

    particles = initialize_particles(config, 0)

    assert np.all(
        particles[:, 0, 2] >= particles[:, 0, 3]
    )
    
    
@pytest.mark.parametrize(
    "boundary_type",
    ["ltf_23triangle", "ltf_23triangle_refl"],
)
def test_initialize_particles_ltf_23triangle_swaps_only_when_required(boundary_type):
    """
    @brief Verify only particles violating the triangle constraint are
    modified.

    Particles already satisfying the constraint should remain unchanged,
    while particles outside the triangle should have the two coordinates
    exchanged.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.8, 0.4],
        [0.2, 0.4, 0.2, 0.9],
    ])

    config = make_config(
        boundary_type=boundary_type,
        N_x=4,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 4), 0.5),
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.8, 0.4],
        [0.9, 0.2],
    ])

    np.testing.assert_allclose(
        particles[:, 0, 2:],
        expected,
    )


@pytest.mark.parametrize(
    "boundary_type",
    ["ltf_23triangle", "ltf_23triangle_refl"],
)
def test_initialize_particles_ltf_23triangle_preserves_other_coordinates(boundary_type):
    """
    @brief Verify triangle initialization affects only coordinates two
    and three.

    All remaining position coordinates should remain identical to their
    initially generated random values.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.2, 0.9, 0.5],
        [0.2, 0.4, 0.8, 0.1, 0.6],
    ])

    config = make_config(
        boundary_type=boundary_type,
        N_x=5,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 5), 0.5),
    )

    particles = initialize_particles(config, 0)

    expected = position_randoms[:, [0, 1, 4]]

    np.testing.assert_allclose(
        particles[:, 0][:, [0, 1, 4]],
        expected,
    )
    
    
@pytest.mark.parametrize(
    "boundary_type",
    ["ltf_23triangle", "ltf_23triangle_refl"],
)
def test_initialize_particles_ltf_23triangle_boundary_is_preserved(boundary_type):
    """
    @brief Verify particles already on the triangle boundary remain
    unchanged.

    Coordinates satisfying

    @code
    x[2] == x[3]
    @endcode

    should not be modified.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.6, 0.6],
        [0.2, 0.4, 0.7, 0.7],
    ])

    config = make_config(
        boundary_type=boundary_type,
        N_x=4,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 4), 0.5),
    )

    particles = initialize_particles(config, 0)

    np.testing.assert_allclose(
        particles[:, 0, 2:],
        position_randoms[:, 2:],
    )
    
    
def test_initialize_particles_matches_matlab_reference():
    """
    @brief Verify particle initialization matches Matlab reference output.

    This regression test compares the initialized particle array against
    a reference data set generated by the original Matlab implementation.
    
    The random values in `Randoms_for_particle_initialization.txt` are
    expected to be written in row-major order, matching the default NumPy
    reshape behavior.
    """

    base_dir = Path(__file__).resolve().parent / "data/"

    premade_pos = np.loadtxt(base_dir / "Randoms_for_particle_initialization_pos.txt")
    premade_vel = np.loadtxt(base_dir / "Randoms_for_particle_initialization_vel.txt")

    config = MockPSOConfig(
            space_topology="box",
            boundary_type="let_them_fly",
            N_x=4,
            N_particles=5,
            clamp_initial_velocity=False,
            vmax=0.5,
            position_init_premade_vals=np.array([premade_pos]),
            velocity_init_premade_vals=np.array([premade_vel]),
        )

    particles = initialize_particles(config, 0)

    reference = list(
        np.loadtxt(
            base_dir / "Initialized_particles_matlab_5p_4x.txt"
        )
    )

    for j in range(2):
        for x in range(config.N_x):
            for i in range(config.N_particles):
                assert particles[i, j, x] == reference.pop(0)
                
                
def test_initialize_particles_imrpd1_applies_spin_combination_transform():
    """
    @brief Verify the IMRPD-1 affine transformation.

    For the IMRPD-1 boundary type, the spin parameters are transformed
    according to:

    @code
    x[4] = 0.5 * (chi_a + chi_s)
    x[5] = 0.5 * (chi_s - chi_a + 1.0)
    @endcode

    The test uses deterministic input values so that the transformed
    coordinates can be compared directly against their expected values.
    """
    position_randoms = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.2, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.4, 0.3],
        [0.3, 0.4, 0.5, 0.6, 0.9, 0.1],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=np.full((3, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.5, 0.8],
        [0.35, 0.45],
        [0.5, 0.1],
    ])

    np.testing.assert_allclose(
        particles[:, 0, 4:6],
        expected,
    )


def test_initialize_particles_imrpd1_positions_are_inside_rotated_square():
    """
    @brief Verify IMRPD-1 spin coordinates lie inside the rotated square.

    The physical spin-combination region is the square centered at
    `(0.5, 0.5)` with vertices at:

    @code
    (0.0, 0.5)
    (0.5, 1.0)
    (1.0, 0.5)
    (0.5, 0.0)
    @endcode

    A point `(x, y)` is inside or on the boundary of this square when:

    @code
    abs(x - 0.5) + abs(y - 0.5) <= 0.5
    @endcode

    All deterministically initialized particles should satisfy this
    condition.
    """
    position_randoms = np.array([
        [0.0, 0.1, 0.2, 0.3, 0.0, 0.0],
        [0.2, 0.3, 0.4, 0.5, 0.0, 1.0],
        [0.4, 0.5, 0.6, 0.7, 1.0, 0.0],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.0],
        [0.8, 0.9, 0.1, 0.2, 0.5, 0.5],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=5,
        position_randoms=position_randoms,
        velocity_randoms=np.full((5, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    chi_a = particles[:, 0, 4]
    chi_s = particles[:, 0, 5]

    distance_from_center = (
        np.abs(chi_a - 0.5)
        + np.abs(chi_s - 0.5)
    )

    assert np.all(distance_from_center <= 0.5)


def test_initialize_particles_imrpd1_random_positions_are_inside_rotated_square():
    """
    @brief Verify arbitrary IMRPD-1 initial positions remain inside the
    rotated square.

    Random points from the original `[0,1]^2` parameter domain should all
    map into the physical rotated square.
    """
    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=1000,
        position_randoms=np.random.default_rng(12345).random(
            (1000, 6)
        ),
        velocity_randoms=np.full((1000, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    chi_a = particles[:, 0, 4]
    chi_s = particles[:, 0, 5]

    distance_from_center = (
        np.abs(chi_a - 0.5)
        + np.abs(chi_s - 0.5)
    )

    assert np.all(distance_from_center <= 0.5)


def test_initialize_particles_imrpd1_maps_square_vertices_to_rotated_vertices():
    """
    @brief Verify the four corners map to the four rotated-square vertices.

    The affine transformation maps the corners of the original `[0,1]^2`
    square to the vertices of the physical spin-combination square:

    @code
    (0, 0) -> (0.0, 0.5)
    (0, 1) -> (0.5, 1.0)
    (1, 0) -> (0.5, 0.0)
    (1, 1) -> (1.0, 0.5)
    @endcode
    """
    position_randoms = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.0, 1.0],
        [0.1, 0.1, 0.1, 0.1, 1.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 1.0, 1.0],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=4,
        position_randoms=position_randoms,
        velocity_randoms=np.full((4, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.0, 0.5],
        [0.5, 1.0],
        [0.5, 0.0],
        [1.0, 0.5],
    ])

    np.testing.assert_allclose(
        particles[:, 0, 4:6],
        expected,
    )


def test_initialize_particles_imrpd1_preserves_other_coordinates():
    """
    @brief Verify IMRPD-1 affects only the spin coordinates.

    Coordinates outside the `(4,5)` spin pair should remain equal to the
    originally generated random values after IMRPD-1 initialization.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.9],
        [0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.5],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=7,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=np.full((2, 7), 0.5),
    )

    particles = initialize_particles(config, 0)

    expected = position_randoms[:, [0, 1, 6]]

    np.testing.assert_allclose(
        particles[:, 0][:, [0, 1, 6]],
        expected,
    )


def test_initialize_particles_imrpd1_preserves_rotated_square_boundary():
    """
    @brief Verify particles on the rotated-square boundary remain on it.

    Points on the boundary satisfy:

    @code
    abs(x - 0.5) + abs(y - 0.5) == 0.5
    @endcode

    The four transformed vertices should therefore remain exactly on the
    physical boundary.
    """
    position_randoms = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.0, 1.0],
        [0.1, 0.1, 0.1, 0.1, 1.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 1.0, 1.0],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=4,
        position_randoms=position_randoms,
        velocity_randoms=np.full((4, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    chi_a = particles[:, 0, 4]
    chi_s = particles[:, 0, 5]

    distance_from_center = (
        np.abs(chi_a - 0.5)
        + np.abs(chi_s - 0.5)
    )

    np.testing.assert_allclose(
        distance_from_center,
        0.5,
    )


def test_initialize_particles_imrpd1_also_enforces_23_triangle():
    """
    @brief Verify IMRPD-1 applies the existing `(2,3)` triangle constraint.

    IMRPD-1 is included in the same initialization path as the
    `ltf_23triangle` boundary types. Therefore, initialization should
    enforce:

    @code
    x[2] >= x[3]
    @endcode

    for every particle.
    """
    position_randoms = np.array([
        [0.1, 0.1, 0.2, 0.8, 0.3, 0.4],
        [0.2, 0.2, 0.9, 0.1, 0.5, 0.6],
        [0.3, 0.3, 0.4, 0.7, 0.7, 0.8],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=np.full((3, 6), 0.5),
    )

    particles = initialize_particles(config, 0)

    assert np.all(
        particles[:, 0, 2] >= particles[:, 0, 3]
    )
                
                
def test_initialize_particles_velocities_match_random_minus_position():
    """
    @brief Verify each velocity coordinate is computed from the corresponding
    initialized position.

    For the standard box topology, the velocity for each coordinate should
    be:

    @code
    velocity = velocity_random - position
    @endcode

    This test uses distinct position and velocity random values for every
    coordinate so that incorrect ordering or broadcasting is detected.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6],
    ])

    velocity_randoms = np.array([
        [0.7, 0.9, 0.2],
        [0.8, 1.0, 0.1],
    ])

    config = make_config(
        N_x=3,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected_velocities = np.array([
        [0.6, 0.6, -0.3],
        [0.6, 0.6, -0.5],
    ])

    np.testing.assert_allclose(
        particles[:, 0, :],
        position_randoms,
    )

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected_velocities,
    )


def test_initialize_particles_velocity_clamping_limits_both_signs():
    """
    @brief Verify velocity clamping limits both positive and negative
    velocities.

    When velocity clamping is enabled, every velocity coordinate should
    satisfy:

    @code
    -vmax <= velocity <= vmax
    @endcode

    The test also verifies that values exactly at the limits remain
    unchanged.
    """
    position_randoms = np.array([
        [0.0, 0.25, 0.5, 0.75],
        [1.0, 0.75, 0.5, 0.25],
    ])

    velocity_randoms = np.array([
        [1.0, 0.5, 0.75, 0.5],
        [0.0, 1.0, 0.25, 0.0],
    ])

    config = make_config(
        N_x=4,
        N_particles=2,
        clamp_initial_velocity=True,
        vmax=0.25,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected = np.array([
        [0.25, 0.25, 0.25, -0.25],
        [-0.25, 0.25, -0.25, -0.25],
    ])

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected,
    )

    assert np.all(particles[:, 1, :] <= config.vmax)
    assert np.all(particles[:, 1, :] >= -config.vmax)


def test_initialize_particles_spherical_surface_velocity_uses_transformed_theta():
    """
    @brief Verify spherical-surface velocity initialization uses the
    transformed theta position.

    The theta coordinate is transformed before velocity initialization.
    Therefore the theta velocity must be calculated using the transformed
    value:

    @code
    theta = arccos(2u - 1) / pi
    velocity_theta = velocity_random - theta
    @endcode

    This test ensures the velocity is not incorrectly calculated from the
    original uniform random theta coordinate.
    """
    position_randoms = np.array([
        [0.0, 0.2],
        [0.5, 0.4],
        [1.0, 0.6],
    ])

    velocity_randoms = np.array([
        [0.25, 0.8],
        [0.25, 0.8],
        [0.25, 0.8],
    ])

    config = make_config(
        space_topology="spherical_surface",
        N_x=2,
        N_particles=3,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected_theta = np.array([
        1.0,
        0.5,
        0.0,
    ])

    expected_velocities = np.array([
        [0.25 - 1.0, 0.8 - 0.2],
        [0.25 - 0.5, 0.8 - 0.4],
        [0.25 - 0.0, 0.8 - 0.6],
    ])

    np.testing.assert_allclose(
        particles[:, 0, 0],
        expected_theta,
    )

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected_velocities,
    )


@pytest.mark.parametrize(
    "boundary_type",
    ["ltf_23triangle", "ltf_23triangle_refl"],
)
def test_initialize_particles_ltf_23triangle_velocity_uses_reordered_positions(
    boundary_type,
):
    """
    @brief Verify triangle initialization modifies the positions before
    velocities are calculated.

    For the (2,3) triangular constraint, the initialized positions satisfy:

    @code
    x[2] >= x[3]
    @endcode

    Velocities must then be calculated from these reordered positions rather
    than from the original random values.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.2, 0.8],
        [0.2, 0.4, 0.9, 0.1],
    ])

    velocity_randoms = np.full((2, 4), 0.5)

    config = make_config(
        boundary_type=boundary_type,
        N_x=4,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    expected_positions = np.array([
        [0.1, 0.3, 0.8, 0.2],
        [0.2, 0.4, 0.9, 0.1],
    ])

    expected_velocities = np.array([
        [0.4, 0.2, -0.3, 0.3],
        [0.3, 0.1, -0.4, 0.4],
    ])

    np.testing.assert_allclose(
        particles[:, 0, :],
        expected_positions,
    )

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected_velocities,
    )


def test_initialize_particles_imrpd1_transforms_velocities_with_spin_coordinates():
    """
    @brief Verify IMRPD-1 applies the affine spin transformation to velocities.

    IMRPD-1 first computes velocities using the pre-transformation positions.
    It then applies the affine transformation to both the spin positions and
    spin velocities.

    For the spin velocities, the resulting transformation is:

    @code
    v_a_new = 0.5 * (v_a + v_s)
    v_s_new = 0.5 * (v_s - v_a)
    @endcode

    The constant term in the position transformation does not contribute to
    the velocity transformation.
    """
    position_randoms = np.array([
        [0.1, 0.3, 0.8, 0.2, 0.0, 0.0],
        [0.2, 0.4, 0.9, 0.1, 1.0, 1.0],
    ])

    velocity_randoms = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.8, 0.2],
        [0.5, 0.5, 0.5, 0.5, 0.2, 0.8],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    # IMRPD-1 spin velocities before the affine transform:
    #
    # particle 0:
    #   v_a = 0.8 - 0.0 = 0.8
    #   v_s = 0.2 - 0.0 = 0.2
    #
    # particle 1:
    #   v_a = 0.2 - 1.0 = -0.8
    #   v_s = 0.8 - 1.0 = -0.2
    #
    # Then:
    #   v_a_new = 0.5 * (v_a + v_s)
    #   v_s_new = 0.5 * (v_s - v_a)
    expected_23_positions = np.array([
        [0.8, 0.2],
        [0.9, 0.1],
    ])

    expected_23_velocities = 0.5 - expected_23_positions

    expected_spin_velocities = np.array([
        [0.5, -0.3],
        [-0.5, 0.3],
    ])

    np.testing.assert_allclose(
        particles[:, 0, 2:4],
        expected_23_positions,
    )

    np.testing.assert_allclose(
        particles[:, 1, 2:4],
        expected_23_velocities,
    )

    np.testing.assert_allclose(
        particles[:, 1, 4:6],
        expected_spin_velocities,
    )
    

def test_initialize_particles_imrpd1_velocity_clamping_occurs_before_spin_transform():
    """
    @brief Verify IMRPD-1 applies velocity clamping before the affine
    spin transformation.

    Velocity clamping occurs immediately after the standard velocity
    calculation and before the IMRPD-1 transformation. Therefore the final
    transformed spin velocities are not necessarily bounded by `vmax`.

    This test documents the ordering of these operations and prevents a
    future change from silently moving the clamping step after the affine
    transformation.
    """
    position_randoms = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.0, 1.0],
    ])

    velocity_randoms = np.array([
        [0.5, 0.5, 0.5, 0.5, 1.0, 0.0],
    ])

    config = make_config(
        boundary_type="IMRPD-1",
        N_x=6,
        N_particles=1,
        clamp_initial_velocity=True,
        vmax=0.2,
        position_randoms=position_randoms,
        velocity_randoms=velocity_randoms,
    )

    particles = initialize_particles(config, 0)

    # Before IMRPD-1:
    #
    # v_a = 1.0 - 0.0 = 1.0 -> clamped to  0.2
    # v_s = 0.0 - 1.0 = -1.0 -> clamped to -0.2
    #
    # After the affine transformation:
    #
    # v_a_new = 0.5 * ( 0.2 + -0.2) =  0.0
    # v_s_new = 0.5 * (-0.2 -  0.2) = -0.2

    expected = np.array([
        [0.0, -0.2],
    ])

    np.testing.assert_allclose(
        particles[:, 1, 4:6],
        expected,
    )
    
@pytest.mark.parametrize(
    "run_index",
    [0, 1],
)
def test_initialize_particles_uses_run_index_for_premade_values(run_index):
    """
    @brief Verify `run_val` selects the correct premade-value stream.

    Each run should retrieve its position and velocity values from the
    corresponding row of the premade-value arrays.
    """
    position_premade = np.array([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.7, 0.8], [0.9, 1.0]],
    ])

    velocity_premade = np.array([
        [[0.5, 0.6], [0.7, 0.8]],
        [[0.2, 0.3], [0.4, 0.5]],
    ])

    config = MockPSOConfig(
        space_topology="box",
        boundary_type="let_them_fly",
        N_x=2,
        N_particles=2,
        clamp_initial_velocity=False,
        vmax=1.0,
        position_init_premade_vals=position_premade,
        velocity_init_premade_vals=velocity_premade,
    )

    particles = initialize_particles(config, run_index)

    expected_positions = position_premade[run_index]
    expected_velocities = (
        velocity_premade[run_index]
        - expected_positions
    )

    np.testing.assert_allclose(
        particles[:, 0, :],
        expected_positions,
    )

    np.testing.assert_allclose(
        particles[:, 1, :],
        expected_velocities,
    )


def test_initialize_particles_does_not_share_premade_stream_state_between_runs():
    """
    @brief Verify different runs maintain independent premade streams.

    Constructing and advancing the generators for one run should not alter
    the values returned for another run.

    The premade-value configuration arrays are two-dimensional, with each
    row containing the sequential premade-value stream for one run.
    """
    position_run_0 = np.array([
        0.1, 0.2, 0.3, 0.4,
    ])

    position_run_1 = np.array([
        0.5, 0.6, 0.7, 0.8,
    ])

    velocity_run_0 = np.array([
        0.4, 0.5, 0.6, 0.7,
    ])

    velocity_run_1 = np.array([
        0.8, 0.9, 1.0, 0.1,
    ])

    config = MockPSOConfig(
        space_topology="box",
        boundary_type="let_them_fly",
        N_x=2,
        N_particles=2,
        clamp_initial_velocity=False,
        vmax=1.0,
        position_init_premade_vals=np.array([
            position_run_0,
            position_run_1,
        ]),
        velocity_init_premade_vals=np.array([
            velocity_run_0,
            velocity_run_1,
        ]),
    )

    # Create generators for both runs independently.
    position_generator_run_0 = VALUE_GENERATORS["premade"](
        config,
        0,
        0,
    )
    position_generator_run_1 = VALUE_GENERATORS["premade"](
        config,
        1,
        0,
    )

    velocity_generator_run_0 = VALUE_GENERATORS["premade"](
        config,
        0,
        1,
    )
    velocity_generator_run_1 = VALUE_GENERATORS["premade"](
        config,
        1,
        1,
    )

    # Advance run 0 before retrieving any values from run 1.
    first_run_0_positions = position_generator_run_0(1, 2)
    second_run_0_positions = position_generator_run_0(1, 2)

    first_run_0_velocities = velocity_generator_run_0(1, 2)
    second_run_0_velocities = velocity_generator_run_0(1, 2)

    first_run_1_positions = position_generator_run_1(2, 2)
    first_run_1_velocities = velocity_generator_run_1(2, 2)

    np.testing.assert_array_equal(
        first_run_0_positions,
        [[0.1, 0.2]],
    )

    np.testing.assert_array_equal(
        second_run_0_positions,
        [[0.3, 0.4]],
    )

    np.testing.assert_array_equal(
        first_run_0_velocities,
        [[0.4, 0.5]],
    )

    np.testing.assert_array_equal(
        second_run_0_velocities,
        [[0.6, 0.7]],
    )

    np.testing.assert_array_equal(
        first_run_1_positions,
        [
            [0.5, 0.6],
            [0.7, 0.8],
        ],
    )

    np.testing.assert_array_equal(
        first_run_1_velocities,
        [
            [0.8, 0.9],
            [1.0, 0.1],
        ],
    )
                
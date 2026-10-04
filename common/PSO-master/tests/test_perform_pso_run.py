# -*- coding: utf-8 -*-
"""
@file test_perform_pso_run.py
@brief Integration tests for a complete Particle Swarm Optimization run.

This module contains pytest tests for the `perform_run` function defined in
`perform_pso_run.py`.

The tests verify:

- Correct return types and array shapes.
- Correct operation with and without history saving.
- Deterministic behavior when identical random-number streams are used.
- Independent handling of position, velocity, and update random generators.
- Proper handling of independent premade random-number streams.
- Correct convergence on a simple deterministic fitness function.
- Proper interaction with the boundary-handling mechanism.
- Result matches MATLAB reference data.

The goal of these tests is to validate that the major PSO components operate
correctly together as a complete optimization run.

@author
Ben Carlson

@date
2026-09-10
"""
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose

from core.perform_pso_run import perform_run
from benchmarks.benchmark_functions import Rastrigin
from parallel_wrappers.wrap_fitness_func import make_get_fitness_wrapper
from utils.config import PSOConfig


def sphere(x):
    """
    @brief Simple sphere objective.

    Global minimum occurs at x = 0.
    """
    return np.sum(x * x)


def fitness_wrapper(particles):
    """
    @brief Compute particle fitness values.

    @param particles
    Particle locations and velocities.

    Accepts either

        particles.shape == (N_particles,2,N_x)

    or

        particles.shape == (N_subset,2,N_x)

    @return
    Array containing the fitness value for each particle of shape
    
        (N_particles,)
    """
    positions = particles[:, 0, :]
    return np.sum(positions ** 2, axis=1)


def build_config(
        *,
        save_history=False,
        position_generator="random-uniform",
        velocity_generator="random-uniform",
        update_generator="random-uniform",
        position_premade=None,
        velocity_premade=None,
        update_premade=None,
        seeds=(1234,),
        N_runs=1,
        N_particles=5,
        N_iterations=5,
        N_x=2,
    ):
    """
    @brief Construct a PSO configuration for testing.

    @param save_history
    Whether particle, fitness, and pbest histories are stored.

    @param position_generator
    Value generator used to initialize particle positions.

    @param velocity_generator
    Value generator used to initialize particle velocities.

    @param update_generator
    Value generator used to generate R1 and R2 during particle updates.

    @param position_premade
    Premade random values for position initialization.

    @param velocity_premade
    Premade random values for velocity initialization.

    @param update_premade
    Premade random values for particle updates.

    @param seeds
    Seed sequence for each random-value generator.

    @param N_runs
    Number of PSO runs represented by the configuration.

    @param N_particles
    Number of particles in the swarm.

    @param N_iterations
    Number of iterations performed by each run.

    @param N_x
    Number of dimensions in the search space.

    @return
    A configured PSOConfig instance.
    """
    boundary_conditions = np.tile(
        np.array([-10.0, 10.0]),
        (N_x, 1),
    )

    # PSOConfig expects one seed setting for each independent value generator:
    # position initialization, velocity initialization, and particle updates.
    seed_values = [
        np.asarray(seeds),
        np.asarray(seeds),
        np.asarray(seeds),
    ]
    
    return PSOConfig(
        N_runs=N_runs,
        N_particles=N_particles,
        N_iterations=N_iterations,
        boundary_conditions=boundary_conditions,

        save_history=save_history,

        position_init_value_type=position_generator,
        velocity_init_value_type=velocity_generator,
        update_value_type=update_generator,

        position_init_premade_vals=position_premade,
        velocity_init_premade_vals=velocity_premade,
        update_premade_vals=update_premade,

        seed_vals=seed_values,

        clamp_velocity=True,
        clamp_initial_velocity=True,
        vmax=0.5,

        C1=2.0,
        C2=2.0,
        eq_type="inertia",

        PSO_type="gbest",
        N_local=3,

        boundary_type="let_them_fly",
        space_topology="box",
    )


def test_perform_run_returns_expected_shapes():
    """
    @brief Verify output array shapes.

    Confirms that perform_run() returns correctly shaped outputs when
    history saving is disabled.
    """
    config = build_config()

    gbest, final_f, particle_hist, fitness_hist, pbest_hist = perform_run(
        run_val=0,
        config=config,
        get_fitness_wrapper=fitness_wrapper,
    )

    assert gbest.shape == (config.N_x,)
    assert np.isscalar(final_f)

    assert particle_hist.size == 0
    assert fitness_hist.size == 0
    assert pbest_hist.size == 0


def test_perform_run_history_lengths():
    """
    @brief Verify saved histories contain one entry per iteration.

    History should contain the initial swarm plus every iteration.
    """

    config = build_config(seeds=(10,), save_history=True)

    (_,
     _,
     particle_hist,
     fitness_hist,
     pbest_hist,
        )    = perform_run(
                        run_val=0,
                        config=config,
                        get_fitness_wrapper=fitness_wrapper,
                    )

    expected = config.N_iterations + 1

    assert particle_hist.shape[0] == expected
    assert fitness_hist.shape[0] == expected
    assert pbest_hist.shape[0] == expected


def test_perform_run_is_reproducible():
    """
    @brief Verify identical seeds produce identical runs.
    
    All three random-value generators receive identical seed settings.
    Repeated executions should therefore produce identical results.
    """

    config = build_config(seeds=(123,), save_history=True)

    result1 = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    result2 = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    for a, b in zip(result1, result2):
        assert_allclose(a, b)


def test_perform_run_with_independent_premade_random_numbers():
    """
    @brief Verify independent premade random-number streams.

    Position initialization, velocity initialization, and particle updates
    each receive their own premade random-number stream. Two runs with
    identical streams should produce identical results even though the
    configuration contains multiple runs.
    """
    N_particles = 5
    N_x = 2
    N_iterations = 5

    position_randoms = np.linspace(
        0.05,
        0.30,
        N_particles * N_x,
    )

    velocity_randoms = np.linspace(
        0.35,
        0.60,
        N_particles * N_x,
    )

    update_randoms = np.linspace(
        0.05,
        0.95,
        2 * N_particles * N_x * N_iterations,
    )

    # Each row corresponds to one PSO run. The two rows are intentionally
    # identical so that run 0 and run 1 receive identical random streams.
    position_premade = np.array([
        position_randoms,
        position_randoms,
    ])

    velocity_premade = np.array([
        velocity_randoms,
        velocity_randoms,
    ])

    update_premade = np.array([
        update_randoms,
        update_randoms,
    ])

    config = build_config(
        N_runs=2,
        position_generator="premade",
        velocity_generator="premade",
        update_generator="premade",
        position_premade=position_premade,
        velocity_premade=velocity_premade,
        update_premade=update_premade,
        seeds=(1, 99999),
        save_history=True,
    )

    result1 = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    result2 = perform_run(
        1,
        config,
        fitness_wrapper,
    )

    for first, second in zip(result1, result2):
        assert_allclose(first, second)


def test_perform_run_final_fitness_is_best_particle():
    """
    @brief Verify returned fitness equals the fitness of gbest.

    The returned global-best location should have a fitness equal to the
    final fitness value returned by perform_run().
    """
    config = build_config(
        seeds=(7,),
    )

    gbest, final_f, *_ = perform_run(
        run_val=0,
        config=config,
        get_fitness_wrapper=fitness_wrapper,
    )

    assert_allclose(final_f, sphere(gbest))


def test_perform_run_boundary_handler_keeps_fitness_finite():
    """
    @brief Verify optimization completes successfully when particles leave
    the search space.

    This primarily exercises the boundary-handler integration.
    """
    config = build_config(
        seeds=(42,),
    )

    # Use aggressive acceleration and velocity limits to encourage particles
    # to leave the configured search region.
    config = PSOConfig(
        N_runs=config.N_runs,
        N_particles=config.N_particles,
        N_iterations=config.N_iterations,
        boundary_conditions=config.boundary_conditions,

        save_history=config.save_history,

        position_init_value_type=config.position_init_value_type,
        velocity_init_value_type=config.velocity_init_value_type,
        update_value_type=config.update_value_type,

        position_init_premade_vals=config.position_init_premade_vals,
        velocity_init_premade_vals=config.velocity_init_premade_vals,
        update_premade_vals=config.update_premade_vals,

        seed_vals=config.seed_vals,

        clamp_velocity=True,
        clamp_initial_velocity=True,
        vmax=5.0,

        C1=5.0,
        C2=5.0,
        eq_type=config.eq_type,

        PSO_type=config.PSO_type,
        N_local=config.N_local,

        boundary_type=config.boundary_type,
        space_topology=config.space_topology,
    )

    gbest, final_f, *_ = perform_run(
        run_val=0,
        config=config,
        get_fitness_wrapper=fitness_wrapper,
    )

    assert np.isfinite(final_f)
    assert np.all(np.isfinite(gbest))
    

def test_perform_run_position_init_premade_update_random():
    """
    @brief Verify position initialization is independent of update generation.

    Particle positions are initialized using premade values while velocities
    and particle updates use random-uniform values. The initial particle
    positions must exactly match the supplied position initialization stream.
    """
    N_particles = 5
    N_x = 2

    position_randoms = np.linspace(
        0.1,
        0.4,
        N_particles * N_x,
    )

    config = build_config(
        position_generator="premade",
        velocity_generator="random-uniform",
        update_generator="random-uniform",
        position_premade=np.array([position_randoms]),
        seeds=(123,),
        save_history=True,
    )

    (
        gbest,
        final_f,
        particle_hist,
        fitness_hist,
        pbest_hist,
    ) = perform_run(
        run_val=0,
        config=config,
        get_fitness_wrapper=fitness_wrapper,
    )

    expected_initial_locations = position_randoms.reshape(
        (N_particles, N_x)
    )

    assert_allclose(
        particle_hist[0, :, 0, :],
        expected_initial_locations,
    )

    assert gbest.shape == (N_x,)
    assert np.isscalar(final_f)
    assert particle_hist.shape[0] == config.N_iterations + 1
    assert fitness_hist.shape[0] == config.N_iterations + 1
    assert pbest_hist.shape[0] == config.N_iterations + 1


def test_perform_run_velocity_init_premade_is_independent_of_position_init():
    """
    @brief Verify velocity initialization has its own value generator.

    Two runs use identical position initialization and update settings but
    different premade velocity streams. The initial particle positions should
    therefore be identical while the initial velocities should differ.

    This verifies that velocity initialization consumes values from its own
    configured generator independently of position initialization.
    """
    N_particles = 5
    N_x = 2

    position_randoms = np.linspace(
        0.1,
        0.4,
        N_particles * N_x,
    )

    velocity_randoms_a = np.linspace(
        0.2,
        0.7,
        N_particles * N_x,
    )

    velocity_randoms_b = np.linspace(
        0.7,
        0.2,
        N_particles * N_x,
    )

    config_a = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="random-uniform",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms_a]),
        seeds=(123,),
        save_history=True,
    )

    config_b = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="random-uniform",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms_b]),
        seeds=(123,),
        save_history=True,
    )

    result_a = perform_run(
        run_val=0,
        config=config_a,
        get_fitness_wrapper=fitness_wrapper,
    )

    result_b = perform_run(
        run_val=0,
        config=config_b,
        get_fitness_wrapper=fitness_wrapper,
    )

    particle_hist_a = result_a[2]
    particle_hist_b = result_b[2]

    # Position initialization is identical between the two configurations.
    assert_allclose(
        particle_hist_a[0, :, 0, :],
        particle_hist_b[0, :, 0, :],
    )

    # The velocity generator receives different premade values, so the
    # resulting initial velocities should differ.
    assert not np.allclose(
        particle_hist_a[0, :, 1, :],
        particle_hist_b[0, :, 1, :],
    )


def test_perform_run_update_premade_changes_trajectory():
    """
    @brief Verify the update generator controls particle trajectories.

    Two runs use identical position and velocity initialization streams but
    different premade update streams. Their initial particle states should
    therefore be identical, while their states after the first iteration
    should differ.
    """
    N_particles = 5
    N_x = 2
    N_iterations = 5

    position_randoms = np.linspace(
        0.1,
        0.4,
        N_particles * N_x,
    )

    velocity_randoms = np.linspace(
        0.2,
        0.5,
        N_particles * N_x,
    )

    update_randoms_a = np.full(
        2 * N_particles * N_x * N_iterations,
        0.1,
    )

    update_randoms_b = np.full(
        2 * N_particles * N_x * N_iterations,
        0.9,
    )

    config_a = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="premade",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms]),
        update_premade=np.array([update_randoms_a]),
        seeds=(123,),
        save_history=True,
    )

    config_b = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="premade",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms]),
        update_premade=np.array([update_randoms_b]),
        seeds=(123,),
        save_history=True,
    )

    result_a = perform_run(
        0,
        config_a,
        fitness_wrapper,
    )

    result_b = perform_run(
        0,
        config_b,
        fitness_wrapper,
    )

    particle_hist_a = result_a[2]
    particle_hist_b = result_b[2]

    # Initialization is identical because both configurations use the same
    # position and velocity streams.
    assert_allclose(
        particle_hist_a[0],
        particle_hist_b[0],
    )

    # The different update streams must alter the trajectory.
    assert not np.allclose(
        particle_hist_a[1],
        particle_hist_b[1],
    )
    
def test_perform_run_mixed_generators_reproducible():
    """
    @brief Verify mixed generator configurations remain reproducible.

    Repeated runs with the same configuration, seeds, and premade values
    should produce identical results when the three random-value generators
    are independently selected.
    """
    N_particles = 5
    N_x = 2

    position_randoms = np.linspace(
        0.1,
        0.4,
        N_particles * N_x,
    )

    velocity_randoms = np.linspace(
        0.2,
        0.5,
        N_particles * N_x,
    )

    config = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="random-uniform",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms]),
        seeds=(123,),
        save_history=True,
    )

    result1 = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    result2 = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    for first, second in zip(result1, result2):
        assert_allclose(first, second)
        
        
def test_perform_run_update_premade_stream_consumed_each_iteration():
    """
    @brief Verify the update premade stream supplies two matrices per
    iteration.

    The update generator must consume values for both R1 and R2 at every
    iteration. A stream containing exactly the required number of values
    should therefore allow the complete run to finish successfully.
    """
    N_particles = 5
    N_x = 2
    N_iterations = 5

    position_randoms = np.linspace(
        0.1,
        0.4,
        N_particles * N_x,
    )

    velocity_randoms = np.linspace(
        0.2,
        0.5,
        N_particles * N_x,
    )

    update_randoms = np.linspace(
        0.05,
        0.95,
        2 * N_particles * N_x * N_iterations,
    )

    config = build_config(
        position_generator="premade",
        velocity_generator="premade",
        update_generator="premade",
        position_premade=np.array([position_randoms]),
        velocity_premade=np.array([velocity_randoms]),
        update_premade=np.array([update_randoms]),
        seeds=(123,),
        save_history=True,
    )

    (
        gbest,
        final_f,
        particle_hist,
        fitness_hist,
        pbest_hist,
    ) = perform_run(
        0,
        config,
        fitness_wrapper,
    )

    assert gbest.shape == (N_x,)
    assert np.isfinite(final_f)
    assert np.all(np.isfinite(gbest))

    assert particle_hist.shape[0] == N_iterations + 1
    assert fitness_hist.shape[0] == N_iterations + 1
    assert pbest_hist.shape[0] == N_iterations + 1


###############################################################################
###############################################################################
    
def test_run_results_matlab_cross_validation():
    """
    @brief Verify one run matches MATLAB reference data.
    
    @note
    gbest and best_fitness are checked exactly.
    
    @note
    particle positions are checked within tolerance 3 ULP.
    
    @note
    One set of particle positions is slightly different: 
        0.5520485052185232
        0.5520485052185229
    which differ by about 1.5 x machine epsilon. All other numbers are exact.    
    """
    base_dir = Path(__file__).resolve().parent / "data"
    N_p = 5
    N_x = 4
    N_i = 200
    
    # Load reference data
    reference = list(np.loadtxt(base_dir / "particles_after_run_matlab.txt"))
    reference_locations = np.empty((N_p, N_x))
    for x in range(N_x):
        for i in range(N_p):
            reference_locations[i, x] = reference.pop(0)
    reference_fitness = 3.0071998229902661
    reference_gbest = reference_locations[0]
    
    # Calculate using python
    bcs = np.array([
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0]
                    ])
    premade_pos = np.array([np.loadtxt(base_dir / "Randoms_for_particle_initialization_pos.txt")])
    premade_vel = np.array([np.loadtxt(base_dir / "Randoms_for_particle_initialization_vel.txt")])
    premade_upd = np.array([np.loadtxt(base_dir / "Randoms_for_perform_run_test.txt")])
    config = PSOConfig(
                       N_runs=1,
                       N_particles=N_p,
                       N_iterations=N_i,
                       boundary_conditions=bcs,
                       save_history=True,
                       position_init_value_type="premade",
                       velocity_init_value_type="premade",
                       update_value_type="premade",
                       position_init_premade_vals=premade_pos,
                       velocity_init_premade_vals=premade_vel,
                       update_premade_vals=premade_upd,
                       )
    
    get_fitness_wrapper = make_get_fitness_wrapper(config, Rastrigin)
    
    gbest, best_f, particle_h, fitness_h, pbest_h = perform_run(0,
                                                                config,
                                                                get_fitness_wrapper)
    last_locations = particle_h[-1]
    
    # check particle positions
    for p in range(N_p):
        for x in range(N_x):
            np.testing.assert_array_max_ulp(last_locations[p, 0, x], 
                                            reference_locations[p, x], 
                                            maxulp=3)
# =============================================================================
#             np.testing.assert_allclose(
#                 last_locations[p, 0, x],
#                 reference_locations[p, x],
#                 rtol=1e-15,
#                 atol=0.0
#             )
# =============================================================================
    
    # check gbest
    for x in range(N_x):
        assert gbest[x] == reference_gbest[x]
    
    # check fitness
    assert best_f == reference_fitness
    
    

# -*- coding: utf-8 -*-
"""
@file test_pso_config.py

@brief Unit tests for the PSOConfig dataclass.

@author
Ben Carlson

@date
2026-09-10
"""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from utils.config import PSOConfig

# -------------------------------------------------------------------------
# Optional CuPy support
# -------------------------------------------------------------------------

try:
    import cupy as cp

    try:
        CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        CUPY_AVAILABLE = False

except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def valid_bcs():
    """
    @brief Return valid boundary conditions.
    """
    return np.array(
        [
            [-5.0, 5.0],
            [-2.0, 2.0],
        ]
    )


def make_config(**kwargs):
    """
    @brief Construct a valid configuration.
    """
    params = dict(
        N_runs=4,
        N_iterations=20,
        boundary_conditions=valid_bcs(),
    )

    params.update(kwargs)

    return PSOConfig(**params)


# ============================================================================
# Construction
# ============================================================================

def test_default_configuration():
    """
    @brief Verify default configuration constructs successfully.
    """
    cfg = make_config()

    assert cfg.N_runs == 4
    assert cfg.N_iterations == 20
    assert cfg.N_x == 2
    assert cfg.N_particles == 40
    assert cfg.N_local == 3
    assert cfg.PSO_type == "lbest"
    assert cfg.boundary_type == "let_them_fly"
    assert cfg.space_topology == "box"

    assert cfg.C1 == 2.0
    assert cfg.C2 == 2.0
    assert cfg.eq_type == "inertia"

    assert cfg.vmax == 0.5
    assert cfg.clamp_initial_velocity is False
    assert cfg.clamp_velocity is True

    assert cfg.position_init_value_type == "random-uniform"
    assert cfg.velocity_init_value_type == "random-uniform"
    assert cfg.update_value_type == "random-uniform"

    assert cfg.run_parallel_mode == "series"
    assert cfg.fitness_parallel_mode == "series"

    assert cfg.verbose is False
    assert cfg.save_history is False


def test_boundary_conditions_converted():
    """
    @brief Verify boundary conditions become float64.
    """
    cfg = make_config(
        boundary_conditions=[[-1, 1]]
    )

    assert cfg.boundary_conditions.dtype == np.float64
    assert cfg.boundary_conditions.shape == (1, 2)
    assert cfg.N_x == 1


def test_n_x_is_derived_from_boundary_conditions():
    """
    @brief Verify N_x is derived from the number of search dimensions.
    """
    boundary_conditions = np.array([
                                    [-1.0, 1.0],
                                    [-2.0, 2.0],
                                    [-3.0, 3.0],
                                    [-4.0, 4.0],
                                    ])
    
    cfg = make_config(
        boundary_conditions=boundary_conditions
    )
    
    assert cfg.N_x == 4
    
def test_n_x_cannot_be_overridden():
    """
    @brief Verify N_x is an init=False derived field.
    """
    with pytest.raises(TypeError):
        PSOConfig(
            N_runs=4,
            N_iterations=20,
            boundary_conditions=valid_bcs(),
            N_x=2,
            )
        
def test_boundary_conditions_are_read_only():
    """
    @brief Verify validated boundary conditions are immutable.
    """
    cfg = make_config()
    
    assert cfg.boundary_conditions.flags.writeable is False
    
    with pytest.raises(ValueError):
        cfg.boundary_conditions[0, 0] = -10.0

def test_boundary_conditions_are_contiguous():
    """
    @brief Verify validated boundary conditions are contiguous.
    """
    cfg = make_config()
    
    assert cfg.boundary_conditions.flags.c_contiguous is True

def test_configuration_is_frozen():
    """
    @brief Verify configuration is immutable.
    """
    cfg = make_config()

    with pytest.raises(FrozenInstanceError):
        cfg.N_particles = 100


# ============================================================================
# Required arguments
# ============================================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("N_runs", 0),
        ("N_runs", -1),
        ("N_iterations", 0),
        ("N_iterations", -5),
    ],
)
def test_invalid_required_arguments(field, value):
    """
    @brief Verify invalid required arguments raise.
    """
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        make_config(**kwargs)

@pytest.mark.parametrize(
    "field,value",
    [
        ("N_runs", 1.0),
        ("N_runs", "4"),
        ("N_iterations", 1.0),
        ("N_iterations", "20"),
    ],
)
def test_invalid_required_argument_types(field, value):
    """
    @brief Verify required arguments reject invalid types.
    """
    with pytest.raises(TypeError):
        make_config(**{field: value})

# ============================================================================
# Swarm parameters
# ============================================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("N_particles", 0),
        ("N_particles", -1),
        ("N_local", 0),
        ("N_local", 40),
        ("PSO_type", "bad"),
        ("boundary_type", "bad"),
        ("space_topology", "bad"),
    ],
)
def test_invalid_swarm_parameters(field, value):
    """
    @brief Verify swarm parameter validation.
    """
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        make_config(**kwargs)

@pytest.mark.parametrize(
    "field,value",
    [
        ("N_particles", 10.0),
        ("N_local", 2.0),
        ("PSO_type", 1),
        ("boundary_type", 1),
        ("space_topology", 1),
    ],
)
def test_invalid_swarm_parameter_types(field, value):
    """
    @brief Verify swarm parameters reject invalid types.
    """
    kwargs = {field: value}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)


def test_n_local_must_be_less_than_n_particles():
    """
    @brief Verify N_local must be within [1, N_particles - 1].
    """
    with pytest.raises(ValueError):
        make_config(
        N_particles=10,
        N_local=10,
        )


@pytest.mark.parametrize(
    "boundary_type",
    [
        "ltf_23triangle",
        "ltf_23triangle_refl",
    ],
)
def test_triangle_boundary_requires_four_dimensions(boundary_type):
    """
    @brief Verify 23-triangle boundary types require at least four dimensions.
    """
    with pytest.raises(ValueError):
        make_config(
        boundary_type=boundary_type,
        boundary_conditions=[
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        ],
        )

@pytest.mark.parametrize(
    "boundary_type",
    [
        "ltf_2345triangle",
        "ltf_2345triangle_refl",
        "IMRPD-1",
    ],
)
def test_high_dimension_boundary_requires_six_dimensions(boundary_type):
    """
    @brief Verify high-dimensional boundary types require at least six
    dimensions.
    """
    with pytest.raises(ValueError):
        make_config(
        boundary_type=boundary_type,
        boundary_conditions=[
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        ],
    )
        
def test_four_dimension_boundary_is_accepted():
    """
    @brief Verify 23-triangle boundary types work with four dimensions.
    """
    bcs = [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    
    cfg = make_config(
        boundary_type="ltf_23triangle",
        boundary_conditions=bcs,
    )
    
    assert cfg.N_x == 4
    
def test_six_dimension_boundary_is_accepted():
    """
    @brief Verify six-dimensional boundary types work with six dimensions.
    """
    bcs = [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
    
    cfg = make_config(
        boundary_type="IMRPD-1",
        boundary_conditions=bcs,
    )
    
    assert cfg.N_x == 6
    

# ============================================================================
# Dynamical parameters
# ============================================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("C1", 0),
        ("C1", -1),
        ("C2", 0),
        ("C2", -1),
        ("vmax", 0),
        ("vmax", -1),
        ("eq_type", "bad"),
    ],
)
def test_invalid_dynamics(field, value):
    """
    @brief Verify dynamics validation.
    """
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        make_config(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("C1", "2.0"),
        ("C2", "2.0"),
        ("vmax", "0.5"),
        ("eq_type", 1),
    ],
)
def test_invalid_dynamics_types(field, value):
    """
    @brief Verify dynamics parameters reject invalid types.
    """
    kwargs = {field: value}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)

def test_valid_dynamics_parameters():
    """
    @brief Verify valid dynamics parameters are accepted.
    """
    cfg = make_config(
        C1=1.5,
        C2=2.5,
        vmax=0.25,
        eq_type="constriction",
    )
    
    assert cfg.C1 == 1.5
    assert cfg.C2 == 2.5
    assert cfg.vmax == 0.25
    assert cfg.eq_type == "constriction"
    
    
# ============================================================================
# Velocity control
# ============================================================================

@pytest.mark.parametrize(
    "field",
    [
        "clamp_velocity",
        "clamp_initial_velocity",
    ],
)
def test_boolean_validation(field):
    """
    @brief Verify velocity-control boolean parameters reject non-bools.
    """
    kwargs = {field: "yes"}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)

def test_velocity_control_options():
    """
    @brief Verify valid velocity-control options are accepted.
    """
    cfg = make_config(
        clamp_velocity=False,
        clamp_initial_velocity=True,
        )
    
    assert cfg.clamp_velocity is False
    assert cfg.clamp_initial_velocity is True
    

# =============================================================================
# Random parameters
# =============================================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("position_init_value_type", "bad"),
        ("velocity_init_value_type", "bad"),
        ("update_value_type", "bad"),
    ],
)
def test_invalid_value_generation_types(field, value):
    """
    @brief Verify value generator types.
    """
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        make_config(**kwargs)
        
        
@pytest.mark.parametrize(
    "field,value",
    [
        ("position_init_value_type", 1),
        ("velocity_init_value_type", 1),
        ("update_value_type", 1),
    ],
)
def test_invalid_value_generation_type_types(field, value):
    """
    @brief Verify value generator parameters reject non-string types.
    """
    kwargs = {field: value}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)
        
        
@pytest.mark.parametrize(
    "value_type",
    [
        "halton",
        "random-uniform",
        "premade",
        "sobol",
    ],
)
def test_supported_position_value_generation_types(value_type):
    """
    @brief Verify supported position value generators are accepted.
    
    @note
    When the "premade" generator is selected, valid pre-computed
    position values are supplied to satisfy configuration validation.
    """
    kwargs = {
        "position_init_value_type": value_type,
    }

    if value_type == "premade":
        n_values = 2 * 40
        kwargs["position_init_premade_vals"] = np.zeros((4, n_values))

    cfg = make_config(**kwargs)

    assert cfg.position_init_value_type == value_type
    
    
@pytest.mark.parametrize(
    "value_type",
    [
        "halton",
        "random-uniform",
        "premade",
        "sobol",
    ],
)
def test_supported_velocity_value_generation_types(value_type):
    """
    @brief Verify supported velocity value generators are accepted.
    
    @note
    When the "premade" generator is selected, valid pre-computed
    velocity values are supplied to satisfy configuration validation.
    """
    kwargs = {
        "velocity_init_value_type": value_type,
    }

    if value_type == "premade":
        n_values = 2 * 40
        kwargs["velocity_init_premade_vals"] = np.zeros((4, n_values))

    cfg = make_config(**kwargs)

    assert cfg.velocity_init_value_type == value_type


@pytest.mark.parametrize(
    "value_type",
    [
        "halton",
        "random-uniform",
        "premade",
        "sobol",
    ],
)
def test_supported_update_value_generation_types(value_type):
    """
    @brief Verify supported update value generators are accepted.
    
    @note
    When the "premade" generator is selected, valid pre-computed
    update values are supplied to satisfy configuration validation.
    """
    kwargs = {
        "update_value_type": value_type,
    }

    if value_type == "premade":
        n_values = 2 * 20 * 2 * 40
        kwargs["update_premade_vals"] = np.zeros((4, n_values))

    cfg = make_config(**kwargs)

    assert cfg.update_value_type == value_type


# ============================================================================
# Premade random values
# ============================================================================

def test_position_premade_values_require_premade_generator():
    """
    @brief Verify position premade values are ignored unless the position
    generator is 'premade'.
    """
    vals = np.full(
        (4, 80),
        0.5,
    )
    
    cfg = make_config(
        position_init_premade_vals=vals,
    )
    
    assert cfg.position_init_premade_vals is None

def test_velocity_premade_values_require_premade_generator():
    """
    @brief Verify velocity premade values are ignored unless the velocity
    generator is 'premade'.
    """
    vals = np.full(
        (4, 80),
        0.5,
    )
    
    cfg = make_config(
        velocity_init_premade_vals=vals,
    )
    
    assert cfg.velocity_init_premade_vals is None

def test_update_premade_values_require_premade_generator():
    """
    @brief Verify update premade values are ignored unless the update
    generator is 'premade'.
    """
    vals = np.full(
        (4, 1600),
        0.5,
    )
    
    cfg = make_config(
        update_premade_vals=vals,
    )
    
    assert cfg.update_premade_vals is None

def test_default_premade_values_are_none():
    """
    @brief Verify premade random-value arrays default to None.
    """
    cfg = make_config()
    
    assert cfg.position_init_premade_vals is None
    assert cfg.velocity_init_premade_vals is None
    assert cfg.update_premade_vals is None

def test_position_premade_values_are_accepted():
    """
    @brief Verify correctly-sized position premade values are accepted.
    """
    correct_length = 2 * 40
    vals = np.full(
        (4, correct_length),
        0.5,
    )
    
    cfg = make_config(
        position_init_value_type="premade",
        position_init_premade_vals=vals,
    )
    
    assert cfg.position_init_premade_vals.dtype == np.float64
    assert cfg.position_init_premade_vals.shape == (4, correct_length)

def test_velocity_premade_values_are_accepted():
    """
    @brief Verify correctly-sized velocity premade values are accepted.
    """
    correct_length = 2 * 40
    vals = np.full(
        (4, correct_length),
        0.5,
    )
    
    cfg = make_config(
        velocity_init_value_type="premade",
        velocity_init_premade_vals=vals,
    )
    
    assert cfg.velocity_init_premade_vals.dtype == np.float64
    assert cfg.velocity_init_premade_vals.shape == (4, correct_length)

def test_update_premade_values_are_accepted():
    """
    @brief Verify correctly-sized update premade values are accepted.
    """
    correct_length = 2 * 20 * 2 * 40
    vals = np.full(
        (4, correct_length),
        0.5,
    )
    
    cfg = make_config(
        update_value_type="premade",
        update_premade_vals=vals,
    )
    
    assert cfg.update_premade_vals.dtype == np.float64
    assert cfg.update_premade_vals.shape == (4, correct_length)


@pytest.mark.parametrize(
    "field,value_type,correct_length",
    [
        (
        "position_init_premade_vals",
        "position_init_value_type",
        2 * 40,
        ),
        (
        "velocity_init_premade_vals",
        "velocity_init_value_type",
        2 * 40,
        ),
        (
        "update_premade_vals",
        "update_value_type",
        2 * 20 * 2 * 40,
        ),
    ],
)
def test_premade_values_reject_too_few_values(
    field,
    value_type,
    correct_length,
    ):
    """
    @brief Verify premade arrays reject insufficient values.
    """
    field_type = {
        "position_init_premade_vals": "premade",
        "velocity_init_premade_vals": "premade",
        "update_premade_vals": "premade",
    }
    
    vals = np.full(
        (4, correct_length - 1),
        0.5,
    )
    
    with pytest.raises(ValueError):
        make_config(
            **{
                value_type: field_type[field],
                field: vals,
            }
        )

# ============================================================================
# Seeds
# ============================================================================

def test_default_seed_configuration():
    """
    @brief Verify the default seed configuration generates three seed
    collections.
    """
    cfg = make_config()
    
    assert len(cfg.seed_vals) == 3
    
    for seeds in cfg.seed_vals:
        assert len(seeds) == 4

def test_seed_true_configuration():
    """
    @brief Verify True generates deterministic seeds for all generators.
    """
    cfg = make_config(seed_vals=True)
    
    assert len(cfg.seed_vals) == 3
    
    for seeds in cfg.seed_vals:
        assert len(seeds) == 4

def test_seed_false_configuration():
    """
    @brief Verify False generates nondeterministic SeedSequence objects.
    """
    cfg = make_config(seed_vals=False)
    
    assert len(cfg.seed_vals) == 3
    
    for seeds in cfg.seed_vals:
        assert len(seeds) == 4
        assert all(
            isinstance(seed, np.random.SeedSequence)
            for seed in seeds
        )

def test_explicit_integer_seeds():
    """
    @brief Verify explicit integer seeds are accepted for all generators.
    """
    seeds = [
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
    ]
    
    cfg = make_config(seed_vals=seeds)
    
    for actual, expected in zip(cfg.seed_vals, seeds):
        np.testing.assert_array_equal(
            actual,
            np.asarray(expected, dtype=np.uint64),
        )

def test_mixed_seed_specifications():
    """
    @brief Verify different seed specification types can be used for each
    generator.
    """
    seed_vals = [
        True,
        [10, 20, 30, 40],
        False,
    ]
    
    cfg = make_config(seed_vals=seed_vals)
    
    assert len(cfg.seed_vals) == 3
    assert len(cfg.seed_vals[0]) == 4
    assert len(cfg.seed_vals[1]) == 4
    assert len(cfg.seed_vals[2]) == 4
    
    np.testing.assert_array_equal(
        cfg.seed_vals[1],
        np.array([10, 20, 30, 40], dtype=np.uint64),
    )

def test_seedsequence_objects():
    """
    @brief Verify explicit SeedSequence objects are accepted.
    """
    seeds = [
        np.random.SeedSequence(1),
        np.random.SeedSequence(2),
        np.random.SeedSequence(3),
        np.random.SeedSequence(4),
    ]
    
    cfg = make_config(
        seed_vals=[
            seeds,
            seeds,
            seeds,
        ]
    )
    
    for generator_seeds in cfg.seed_vals:
        assert generator_seeds == seeds


# ============================================================================
# Parallelization
# ============================================================================

@pytest.mark.parametrize(
    "field,value",
    [
        ("run_parallel_mode", "bad"),
        ("fitness_parallel_mode", "bad"),
        ("run_n_jobs", 0),
        ("fitness_n_jobs", 0),
    ],
)
def test_invalid_parallelization(field, value):
    """
    @brief Verify parallelization validation.
    """
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        make_config(**kwargs)
        
@pytest.mark.parametrize(
    "field,value",
    [
        ("run_parallel_mode", 1),
        ("fitness_parallel_mode", 1),
        ("run_n_jobs", 1.0),
        ("fitness_n_jobs", 1.0),
    ],
)
def test_invalid_parallelization_types(field, value):
    """
    @brief Verify parallelization parameters reject invalid types.
    """
    kwargs = {field: value}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)

@pytest.mark.parametrize(
    "mode",
    [
        "series",
        "cpu",
        "mpi",
    ],
)
def test_supported_run_parallel_modes(mode):
    """
    @brief Verify supported run parallelization modes are accepted.
    """
    cfg = make_config(
        run_parallel_mode=mode
    )
    
    assert cfg.run_parallel_mode == mode
    
@pytest.mark.parametrize(
    "mode",
    [
        "series",
        "cpu",
    ],
)
def test_supported_fitness_parallel_modes(mode):
    """
    @brief Verify supported fitness parallelization modes are accepted.
    """
    cfg = make_config(
        fitness_parallel_mode=mode
    )
    
    assert cfg.fitness_parallel_mode == mode
    
@pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CUDA-capable CuPy device not available",
)
def test_supported_fitness_parallel_modes():
    """
    @brief Verify supported fitness parallelization modes are accepted.
    """
    cfg = make_config(
        fitness_parallel_mode="cuda-cupy"
    )
    
    assert cfg.fitness_parallel_mode == "cuda-cupy"

def test_parallel_job_counts_are_accepted():
    """
    @brief Verify positive parallel job counts are accepted.
    """
    cfg = make_config(
        run_n_jobs=2,
        fitness_n_jobs=2,
    )
    
    assert cfg.run_n_jobs == 2
    assert cfg.fitness_n_jobs == 2
    
    
# ============================================================================
# Logging / output
# ============================================================================

@pytest.mark.parametrize(
    "field",
    [
        "verbose",
        "save_history",
    ],
)
def test_logging_boolean_validation(field):
    """
    @brief Verify logging/output boolean parameters reject non-bools.
    """
    kwargs = {field: "yes"}
    
    with pytest.raises(TypeError):
        make_config(**kwargs)

def test_logging_options():
    """
    @brief Verify valid logging/output options are accepted.
    """
    cfg = make_config(
        verbose=True,
        save_history=True,
    )
    
    assert cfg.verbose is True
    assert cfg.save_history is True

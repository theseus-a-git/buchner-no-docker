# -*- coding: utf-8 -*-
"""
@file test_config_validation.py

@brief Unit tests for the validation helper functions in config.py.

Tests:
    - validate_boundary_conditions()
    - validate_random_sequence()
    - validate_seeds()
    - validate_entropy_vals()
    - validate_parallelization()

@author
Ben Carlson

@date
2026-09-10
"""
import numpy as np
import pytest
from unittest.mock import patch

from utils.config import (
    validate_boundary_conditions,
    validate_random_sequence,
    validate_entropy_vals,
    validate_seeds,
    validate_parallelization,
)


# ============================================================================
# Shared fixtures
# ============================================================================

N_RUNS = 4
N_ITER = 5
N_X = 2
N_PARTICLES = 3


def position_matrix():
    """
    @brief Return a correctly-sized position initialization matrix.
    """
    n_values = N_X * N_PARTICLES
    
    return np.full(
        (N_RUNS, n_values),
        0.5,
    )

def velocity_matrix():
    """
    @brief Return a correctly-sized velocity initialization matrix.
    """
    n_values = N_X * N_PARTICLES
    
    return np.full(
        (N_RUNS, n_values),
        0.5,
    )

def update_matrix():
    """
    @brief Return a correctly-sized particle update matrix.
    """
    n_values = 2 * N_ITER * N_X * N_PARTICLES
    
    return np.full(
        (N_RUNS, n_values),
        0.5,
    )

# ============================================================================
# validate_boundary_conditions
# ============================================================================

def test_validate_boundary_conditions_success():
    """
    @brief Verify valid boundary conditions are accepted.
    """
    bcs = [[-5, 5], [0, 10]]

    result = validate_boundary_conditions(bcs)

    assert result.dtype == np.float64
    assert result.shape == (2, 2)
    assert result.flags.writeable is False
    assert result.flags.c_contiguous is True

def test_validate_boundary_conditions_conversion():
    """
    @brief Verify boundary conditions are converted to float64.
    """
    bcs = [
        [-5, 5],
        [0, 10],
    ]
    
    result = validate_boundary_conditions(bcs)
    
    assert result.dtype == np.float64
    
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [-5.0, 5.0],
                [0.0, 10.0],
            ]
        ),
    )

@pytest.mark.parametrize(
    "bcs",
    [
        [],
        [1, 2],
        [[1, 2, 3]],
        [[5, 5]],
        [[5, 4]],
        [[np.nan, 1]],
        [[0, np.inf]],
        [[-np.inf, 1]],
    ],
)
def test_validate_boundary_conditions_invalid(bcs):
    """
    @brief Verify invalid boundary conditions raise ValueError.
    """
    with pytest.raises(ValueError):
        validate_boundary_conditions(bcs)
        
def test_validate_boundary_conditions_nonconvertible():
    """
    @brief Verify non-numeric boundary conditions raise TypeError.
    """
    with pytest.raises(TypeError):
        validate_boundary_conditions([["low", "high"]])


# ============================================================================
# validate_random_sequence
# ============================================================================

def test_validate_random_sequence_none():
    """
    @brief Verify None is accepted when the corresponding generator is not
    'premade'.
    """
    result = validate_random_sequence(
        None,
        "random-uniform",
        "random-uniform",
        "random-uniform",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        0,
    )
    
    assert result is None

@pytest.mark.parametrize(
    "generator_val,matrix_factory",
    [
        (0, position_matrix),
        (1, velocity_matrix),
        (2, update_matrix),
    ],
)
def test_validate_random_sequence_success(generator_val, matrix_factory):
    """
    @brief Verify correctly-sized premade matrices are accepted.
    """
    arr = matrix_factory()
    
    result = validate_random_sequence(
        arr,
        "premade",
        "premade",
        "premade",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        generator_val,
    )
    
    assert result.dtype == np.float64
    assert result.shape == arr.shape
    
    np.testing.assert_allclose(result, arr)

def test_validate_random_sequence_position_length():
    """
    @brief Verify position initialization requires N_x * N_particles values
    per run.
    """
    correct_length = N_X * N_PARTICLES
    
    result = validate_random_sequence(
        np.full((N_RUNS, correct_length), 0.5),
        "premade",
        "premade",
        "premade",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        0,
    )
    
    assert result.shape == (N_RUNS, correct_length)

def test_validate_random_sequence_update_length():
    """
    @brief Verify particle updates require 2 * N_iterations * N_x *
    N_particles values per run.
    """
    correct_length = 2 * N_ITER * N_X * N_PARTICLES
    
    result = validate_random_sequence(
        np.full((N_RUNS, correct_length), 0.5),
        "premade",
        "premade",
        "premade",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        2,
    )
    
    assert result.shape == (N_RUNS, correct_length)

def test_validate_random_sequence_missing_premade_position_values():
    """
    @brief Verify position premade values are required when the position
    generator is 'premade'.
    """
    with pytest.raises(ValueError):
        validate_random_sequence(
            None,
            "premade",
            "random-uniform",
            "random-uniform",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            0,
        )

def test_validate_random_sequence_missing_premade_velocity_values():
    """
    @brief Verify velocity premade values are required when the velocity
    generator is 'premade'.
    """
    with pytest.raises(ValueError):
        validate_random_sequence(
            None,
            "random-uniform",
            "premade",
            "random-uniform",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            1,
        )
        
def test_validate_random_sequence_missing_premade_update_values():
    """
    @brief Verify update premade values are required when the update
    generator is 'premade'.
    """
    with pytest.raises(ValueError):
        validate_random_sequence(
            None,
            "random-uniform",
            "random-uniform",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            2,
        )
        
def test_validate_random_sequence_wrong_run_count():
    """
    @brief Verify the number of rows must equal N_runs.
    """
    arr = np.full(
        (N_RUNS - 1, N_X * N_PARTICLES),
        0.5,
    )
    
    with pytest.raises(ValueError):
        validate_random_sequence(
            arr,
            "premade",
            "premade",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            0,
        )

def test_validate_random_sequence_too_few_values():
    """
    @brief Verify insufficient random values are rejected.
    """
    arr = np.full(
        (N_RUNS, N_X * N_PARTICLES - 1),
        0.5,
    )
    
    with pytest.raises(ValueError):
        validate_random_sequence(
            arr,
            "premade",
            "premade",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            0,
        )

def test_validate_random_sequence_extra_values_warning(capsys):
    """
    @brief Verify extra random values produce a warning and are retained.
    """
    correct_length = N_X * N_PARTICLES
    
    arr = np.full(
        (N_RUNS, correct_length + 2),
        0.5,
    )
    
    result = validate_random_sequence(
        arr,
        "premade",
        "premade",
        "premade",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        0,
    )
    
    captured = capsys.readouterr()
    
    assert "WARNING" in captured.out
    assert result.shape == arr.shape

def test_validate_random_sequence_ignores_unused_premade_values(capsys):
    """
    @brief Verify premade values are ignored when the corresponding
    generator is not 'premade'.
    """
    arr = position_matrix()
    
    result = validate_random_sequence(
        arr,
        "random-uniform",
        "random-uniform",
        "random-uniform",
        N_RUNS,
        N_ITER,
        N_X,
        N_PARTICLES,
        0,
    )
    
    captured = capsys.readouterr()
    
    assert result is None
    assert "WARNING" in captured.out

@pytest.mark.parametrize(
    "bad",
    [
        "hello",
        b"hello",
        [[0.1], [0.2, 0.3]],
        np.ones((3, 5)),
        np.array([[1.2]]),
        np.array([[np.nan]]),
    ],
)
def test_validate_random_sequence_invalid(bad):
    """
    @brief Verify invalid random matrices raise.
    """
    with pytest.raises((TypeError, ValueError)):
        validate_random_sequence(
            bad,
            "premade",
            "premade",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            0,
        )
        
def test_validate_random_sequence_inconsistent_row_lengths():
    """
    @brief Verify rows with different lengths are rejected.
    """
    bad = [
        [0.1, 0.2],
        [0.1],
        [0.1, 0.2],
        [0.1, 0.2],
    ]
    
    with pytest.raises(ValueError):
        validate_random_sequence(
            bad,
            "premade",
            "premade",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            0,
        )

def test_validate_random_sequence_unsupported_generator_value():
    """
    @brief Verify unsupported generator_val values raise NotImplementedError.
    """
    arr = np.full(
        (N_RUNS, 1),
        0.5,
    )
    
    with pytest.raises(NotImplementedError):
        validate_random_sequence(
            arr,
            "premade",
            "premade",
            "premade",
            N_RUNS,
            N_ITER,
            N_X,
            N_PARTICLES,
            3,
        )

# ============================================================================
# validate_entropy_vals
# ============================================================================

def test_validate_entropy_vals_success():
    """
    @brief Verify valid entropy values are accepted.
    """
    entropy_vals = [10, 11111, 12345]

    result = validate_entropy_vals(entropy_vals)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.shape == (3,)

    np.testing.assert_array_equal(
        result,
        np.array(
            [10, 11111, 12345],
            dtype=np.int64,
        ),
    )


def test_validate_entropy_vals_numpy_array():
    """
    @brief Verify a numpy array of valid entropy values is accepted.
    """
    entropy_vals = np.array(
        [10, 11111, 12345],
        dtype=np.int64,
    )

    result = validate_entropy_vals(entropy_vals)

    assert result.dtype == np.int64
    assert result.shape == (3,)

    np.testing.assert_array_equal(
        result,
        entropy_vals,
    )


def test_validate_entropy_vals_conversion():
    """
    @brief Verify valid integer sequences are converted to an int64 numpy
    array.
    """
    entropy_vals = (10, 11111, 12345)

    result = validate_entropy_vals(entropy_vals)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64

    np.testing.assert_array_equal(
        result,
        np.array(
            [10, 11111, 12345],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [10],
        [10, 11111],
        [10, 11111, 12345, 99999],
    ],
)
def test_validate_entropy_vals_wrong_number_of_values(bad):
    """
    @brief Verify entropy_vals must contain exactly three values.
    """
    with pytest.raises(ValueError):
        validate_entropy_vals(bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[10, 11111, 12345]]),
        np.array([[10], [11111], [12345]]),
    ],
)
def test_validate_entropy_vals_wrong_dimension(bad):
    """
    @brief Verify entropy_vals must be a one-dimensional sequence.
    """
    with pytest.raises(ValueError):
        validate_entropy_vals(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [0, 11111, 12345],
        [-1, 11111, 12345],
        [10, 0, 12345],
        [10, -5, 12345],
        [10, 11111, 0],
        [10, 11111, -12345],
    ],
)
def test_validate_entropy_vals_non_positive(bad):
    """
    @brief Verify entropy values must all be positive.
    """
    with pytest.raises(ValueError):
        validate_entropy_vals(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [1.5, 11111, 12345],
        [10, 2.5, 12345],
        [10, 11111, 3.5],
        ["10", 11111, 12345],
        [10, "11111", 12345],
        [10, 11111, "12345"],
        [True, 11111, 12345],
        [10, False, 12345],
        [np.bool_(True), 11111, 12345],
        [10, np.bool_(False), 12345],
    ],
)
def test_validate_entropy_vals_non_integer(bad):
    """
    @brief Verify entropy values must all be integers.
    """
    with pytest.raises(TypeError):
        validate_entropy_vals(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "101111112345",
        b"101111112345",
    ],
)
def test_validate_entropy_vals_string(bad):
    """
    @brief Verify strings and bytes are rejected as entropy values.
    """
    with pytest.raises(TypeError):
        validate_entropy_vals(bad)
        
def test_validate_entropy_vals_numpy_integer_types():
    """
    @brief Verify numpy integer scalar types are accepted as entropy values.
    """
    entropy_vals = [
        np.int8(10),
        np.int32(11111),
        np.int64(12345),
    ]

    result = validate_entropy_vals(entropy_vals)

    assert result.dtype == np.int64

    np.testing.assert_array_equal(
        result,
        np.array(
            [10, 11111, 12345],
            dtype=np.int64,
        ),
    )

# ============================================================================
# validate_seeds
# ============================================================================

def test_validate_seeds_boolean_false():
    """
    @brief Verify False generates nondeterministic SeedSequence objects for
    all three generators.
    """
    seeds = validate_seeds(False, N_RUNS, np.array([10, 11111, 12345]))
    
    assert len(seeds) == 3
    
    for generator_seeds in seeds:
        assert len(generator_seeds) == N_RUNS
        assert all(
            isinstance(seed, np.random.SeedSequence)
            for seed in generator_seeds
        )

def test_validate_seeds_boolean_true_is_reproducible():
    """
    @brief Verify True produces reproducible seed sequences for all
    generators.
    """
    seeds_1 = validate_seeds(True, N_RUNS, np.array([10, 11111, 12345]))
    seeds_2 = validate_seeds(True, N_RUNS, np.array([10, 11111, 12345]))
    
    assert len(seeds_1) == 3
    assert len(seeds_2) == 3
    
    for first, second in zip(seeds_1, seeds_2):
        assert [
            seed.entropy
            for seed in first
        ] == [
            seed.entropy
            for seed in second
        ]

def test_validate_seeds_explicit_integer_seeds():
    """
    @brief Verify explicit integer seeds are accepted independently for each
    generator.
    """
    seed_vals = [
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    ]
    
    result = validate_seeds(seed_vals, N_RUNS, np.array([10, 11111, 12345]))
    
    assert len(result) == 3
    
    for actual, expected in zip(result, seed_vals):
        np.testing.assert_array_equal(
            actual,
            np.asarray(expected, dtype=np.uint64),
        )

def test_validate_seeds_seedsequence_objects():
    """
    @brief Verify explicit SeedSequence objects are accepted.
    """
    seed_sequences = [
        np.random.SeedSequence(1),
        np.random.SeedSequence(2),
        np.random.SeedSequence(3),
        np.random.SeedSequence(4),
    ]
    
    result = validate_seeds(
        [
            seed_sequences,
            seed_sequences,
            seed_sequences,
        ],
        N_RUNS,
        np.array([10, 11111, 12345])
    )
    
    assert len(result) == 3
    
    for generator_seeds in result:
        assert generator_seeds == seed_sequences

def test_validate_seeds_mixed_generator_specifications():
    """
    @brief Verify different seed specification types can be used for
    different generators.
    """
    result = validate_seeds(
        [
            True,
            [10, 20, 30, 40],
            False,
        ],
        N_RUNS,
        np.array([10, 11111, 12345])
    )
    
    assert len(result) == 3
    assert len(result[0]) == N_RUNS
    assert len(result[1]) == N_RUNS
    assert len(result[2]) == N_RUNS
    
    np.testing.assert_array_equal(
        result[1],
        np.array(
            [10, 20, 30, 40],
            dtype=np.uint64,
        ),
    )

@pytest.mark.parametrize(
    "bad",
    [
        [],
        [1, 2],
        [True, False],
        [1, 2, 3, 4],
    ],
)
def test_validate_seeds_wrong_number_of_generators(bad):
    """
    @brief Verify seed_vals must contain exactly three generator
    specifications.
    """
    with pytest.raises(ValueError):
        validate_seeds(bad, N_RUNS, np.array([10, 11111, 12345]))
    
@pytest.mark.parametrize(
"bad",
    [
        [1, 2],
        [1.5, 2, 3, 4],
        [-1, 2, 3, 4],
        ["bad", 2, 3, 4],
    ],
)
def test_validate_seeds_invalid_integer_specification(bad):
    """
    @brief Verify invalid integer seed specifications raise.
    """
    with pytest.raises((TypeError, ValueError)):
        validate_seeds(
            [bad, bad, bad],
            N_RUNS,
        )

def test_validate_seeds_wrong_seed_count():
    """
    @brief Verify each generator must provide one seed per run.
    """
    with pytest.raises(ValueError):
        validate_seeds(
            [
                [1, 2, 3],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            N_RUNS,
            np.array([10, 11111, 12345])
        )

def test_validate_seeds_empty_generator_specification():
    """
    @brief Verify empty generator seed specifications are rejected.
    """
    with pytest.raises(ValueError):
        validate_seeds(
            [
                [],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            N_RUNS,
            np.array([10, 11111, 12345])
        )

def test_validate_seeds_mixed_seed_types_within_generator():
    """
    @brief Verify integer and SeedSequence values cannot be mixed within
    one generator.
    """
    mixed = [
        1,
        np.random.SeedSequence(2),
        3,
        4,
    ]
    
    with pytest.raises(TypeError):
        validate_seeds(
            [
                mixed,
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            N_RUNS,
            np.array([10, 11111, 12345])
        )


# ============================================================================
# validate_parallelization
# ============================================================================

@patch("os.cpu_count", return_value=8)
def test_validate_parallelization_series(mock_cpu):
    """
    @brief Verify default series mode.
    """
    result = validate_parallelization(
        "series",
        None,
        "series",
        None,
        4,
        8,
    )

    assert result == ("series", None, "series", None)


@patch("os.cpu_count", return_value=8)
def test_validate_parallelization_cpu(mock_cpu):
    """
    @brief Verify automatic CPU job selection.
    """
    run_mode, run_jobs, fit_mode, fit_jobs = (
        validate_parallelization(
            "cpu",
            None,
            "series",
            None,
            4,
            8,
        )
    )

    assert run_mode == "cpu"
    assert run_jobs >= 1
    assert fit_mode == "series"
    assert fit_jobs is None


@patch("os.cpu_count", return_value=8)
def test_validate_parallelization_cpu_fitness(mock_cpu):
    """
    @brief Verify CPU fitness auto-selection.
    """
    result = validate_parallelization(
        "series",
        None,
        "cpu",
        None,
        4,
        8,
    )

    assert result[0] == "series"
    assert result[2] == "cpu"
    assert result[3] >= 1
    

@patch("os.cpu_count", return_value=8)
def test_validate_parallelization_explicit_cpu_jobs(mock_cpu):
    """
    @brief Verify explicit CPU job counts are retained when within the
    hardware limit.
    """
    result = validate_parallelization(
        "cpu",
        2,
        "series",
        None,
        4,
        8,
    )
    
    assert result[0] == "cpu"
    assert result[1] == 2
    assert result[2] == "series"
    assert result[3] is None

    
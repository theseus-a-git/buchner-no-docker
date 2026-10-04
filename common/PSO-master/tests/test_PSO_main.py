# -*- coding: utf-8 -*-
"""
@file test_PSO_main.py

@brief Unit tests for PSO_main.py.

These tests verify that run_PSO() correctly constructs the PSO
configuration, validates the fitness function, dispatches execution
to the run wrapper, converts standardized coordinates back to
physical coordinates, and returns the expected values.

External dependencies are mocked so these tests execute quickly and
only verify orchestration logic.

@author
Ben Carlson

@date
2026-07-02
"""

import numpy as np

from dataclasses import dataclass
from unittest.mock import patch

from PSO_main import run_PSO


# -------------------------------------------------------------------------
# Mock configuration
# -------------------------------------------------------------------------

@dataclass
class MockPSOConfig:
    """
    @brief Minimal configuration object used by run_PSO().
    """

    N_runs: int
    N_iterations: int
    boundary_conditions: np.ndarray

    verbose: bool = False

    run_parallel_mode: str = "series"
    run_n_jobs: int = 1

    fitness_parallel_mode: str = "series"
    fitness_n_jobs: int = 1


# -------------------------------------------------------------------------
# Dummy fitness
# -------------------------------------------------------------------------

def dummy_fitness(x):
    """
    @brief Simple deterministic fitness function.
    """
    return np.sum(x ** 2)


# -------------------------------------------------------------------------
# Fake run wrapper
# -------------------------------------------------------------------------

def fake_wrapper(fitness_func, config):
    """
    @brief Replacement for perform_run_wrapper().
    """

    best_f = np.array([3.0, 1.0])

    best_std = np.array([
        [0.2, 0.8],
        [0.5, 0.1],
    ])

    particle_history = np.array([])
    fitness_history = np.array([])
    pbest_history = np.array([])

    return (
        best_f,
        best_std,
        particle_history,
        fitness_history,
        pbest_history,
    )


# -------------------------------------------------------------------------
# Fake un_standardize
# -------------------------------------------------------------------------

def fake_unstandardize(x, bounds):
    """
    @brief Simple predictable coordinate conversion.
    """
    return x + 10.0


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

@patch("PSO_main.un_standardize", side_effect=fake_unstandardize)
@patch("PSO_main.perform_run_wrapper", side_effect=fake_wrapper)
@patch("PSO_main.check_fitness_func")
@patch("PSO_main.PSOConfig", side_effect=MockPSOConfig)
def test_run_pso_returns_expected_values(
    mock_config,
    mock_check,
    mock_wrapper,
    mock_unstd,
):
    """
    @brief Verify run_PSO() returns expected values.
    """

    bounds = np.array([[0.0, 1.0],
                       [0.0, 1.0]])

    best_f, best_x, hist_p, hist_f, hist_pb = run_PSO(
        N_runs=2,
        N_iterations=5,
        boundary_conditions=bounds,
        fitness_func=dummy_fitness,
    )

    assert np.allclose(best_f, [3.0, 1.0])

    assert np.allclose(
        best_x,
        np.array([
            [10.2, 10.8],
            [10.5, 10.1],
        ]),
    )

    assert hist_p.size == 0
    assert hist_f.size == 0
    assert hist_pb.size == 0

    mock_check.assert_called_once()
    mock_wrapper.assert_called_once()
    mock_unstd.assert_called_once()


@patch("PSO_main.check_fitness_func")
@patch("PSO_main.perform_run_wrapper", side_effect=fake_wrapper)
@patch("PSO_main.un_standardize", side_effect=fake_unstandardize)
@patch("PSO_main.PSOConfig", side_effect=MockPSOConfig)
def test_verbose_prints(
    mock_config,
    mock_unstd,
    mock_wrapper,
    mock_check,
    capsys,
):
    """
    @brief Verify verbose mode prints summary information.
    """

    bounds = np.array([[0.0, 1.0]])

    run_PSO(
        N_runs=1,
        N_iterations=2,
        boundary_conditions=bounds,
        fitness_func=dummy_fitness,
        verbose=True,
    )

    out = capsys.readouterr().out

    assert "run_PSO began running" in out
    assert "FINAL RESULTS" in out
    assert "best function value" in out


def test_required_kwargs_cannot_be_overridden():
    """
    @brief Verify required positional arguments cannot be passed
    through kwargs.
    """

    bounds = np.array([[0.0, 1.0]])

    try:
        run_PSO(
            1,
            1,
            bounds,
            dummy_fitness,
            N_runs=5,
        )
    except TypeError:
        return

    assert False


@patch("PSO_main.PSOConfig", side_effect=MockPSOConfig)
@patch("PSO_main.check_fitness_func")
@patch("PSO_main.perform_run_wrapper", side_effect=fake_wrapper)
@patch("PSO_main.un_standardize", side_effect=fake_unstandardize)
def test_boundary_conditions_forwarded(
    mock_unstd,
    mock_wrapper,
    mock_check,
    mock_config,
):
    """
    @brief Verify boundary conditions are passed to
    un_standardize().
    """

    bounds = np.array([
        [-5.0, 5.0],
        [2.0, 8.0],
    ])

    run_PSO(
        2,
        3,
        bounds,
        dummy_fitness,
    )

    args = mock_unstd.call_args[0]

    assert np.array_equal(args[1], bounds)


@patch("PSO_main.PSOConfig", side_effect=MockPSOConfig)
@patch("PSO_main.perform_run_wrapper", side_effect=fake_wrapper)
@patch("PSO_main.un_standardize", side_effect=fake_unstandardize)
@patch("PSO_main.check_fitness_func")
def test_validation_called(
    mock_check,
    mock_unstd,
    mock_wrapper,
    mock_config,
):
    """
    @brief Verify fitness-function validation is performed.
    """

    bounds = np.array([[0.0, 1.0]])

    run_PSO(
        1,
        1,
        bounds,
        dummy_fitness,
    )

    mock_check.assert_called_once()
    
# -*- coding: utf-8 -*-
"""
@file test_wrap_perform_pso_run.py
@brief Unit tests for wrappers executing multiple PSO runs.

This module contains pytest tests for the functions defined in
`wrap_perform_pso_run.py`.

The tests verify:

- Correct serial execution.
- Correct CPU parallel execution.
- Preservation of run ordering.
- Proper collection of histories.
- Correct dispatch through helper wrapper functions.
- Proper exception handling for unsupported execution modes.

The expensive optimization itself is mocked so these tests focus on the
wrapper logic rather than the PSO algorithm, which is tested elsewhere.

@author
Ben Carlson

@date
2026-06-30
"""

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from parallel_wrappers.wrap_perform_pso_run import (
        perform_run_wrapper,
        _perform_run_wrapper2,
        _run_single,
    )


@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object used for testing.

    Members:
    - N_runs
    - N_x
    - seed_vals
    - run_parallel_mode
    - run_n_jobs
    - fitness_parallel_mode
    - fitness_n_jobs
    """

    N_runs: int
    N_x: int

    seed_vals: list

    run_parallel_mode: str
    run_n_jobs: int

    fitness_parallel_mode: str
    fitness_n_jobs: int


def dummy_fitness(x):
    """
    @brief Simple objective function.
    """
    return np.sum(x)


def fake_perform_run(run, wrapper, config):
    """
    @brief Deterministic replacement for perform_run().

    Each run returns values containing the run index so ordering can be
    verified.
    """

    best = np.full(config.N_x, float(run))

    return (
        best,
        float(run),
        np.array([run]),
        np.array([run + 10]),
        np.array([run + 20]),
    )


def build_config(mode="series"):
    """
    @brief Construct a minimal configuration.
    """

    return MockPSOConfig(
        N_runs=4,
        N_x=3,

        seed_vals=[10, 20, 30, 40],

        run_parallel_mode=mode,
        run_n_jobs=2,

        fitness_parallel_mode="series",
        fitness_n_jobs=2,
    )

class DummyExecutor:
    """
    @brief Mock ProcessPoolExecutor used for unit testing.

    This mock executes submitted jobs sequentially in the current
    process while preserving the interface of ProcessPoolExecutor.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def map(self, func, jobs):
        """
        @brief Execute each submitted job sequentially.

        @param func
        Function normally executed by the worker pool.

        @param jobs
        Iterable of job tuples.

        @return
        List containing one result per submitted job.
        """
        return [func(job) for job in jobs]


@patch("parallel_wrappers.wrap_perform_pso_run.make_get_fitness_wrapper")
@patch("parallel_wrappers.wrap_perform_pso_run.perform_run")
def test_wrapper2_series_dispatch(
    mock_perform,
    mock_get_wrapper,
):
    """
    @brief Verify wrapper2 dispatches correctly for serial fitness
    evaluation.
    """

    config = build_config()

    mock_get_wrapper.return_value = object()
    mock_perform.return_value = ("a", "b", "c", "d", "e")

    result = _perform_run_wrapper2(
        0,
        dummy_fitness,
        config,
    )

    assert result == ("a", "b", "c", "d", "e")

    mock_get_wrapper.assert_called_once()
    mock_perform.assert_called_once()


def test_run_single_forwards_arguments():
    """
    @brief Verify _run_single simply forwards its tuple.
    """

    config = build_config()

    with patch(
        "parallel_wrappers.wrap_perform_pso_run._perform_run_wrapper2"
    ) as wrapped:

        wrapped.return_value = 123

        result = _run_single(
            (
                1,
                dummy_fitness,
                config,
            )
        )

        assert result == 123

        wrapped.assert_called_once()


@patch(
    "parallel_wrappers.wrap_perform_pso_run._perform_run_wrapper2",
    side_effect=fake_perform_run,
)
def test_serial_execution(mock_run):
    """
    @brief Verify serial execution returns one result per run.
    """

    config = build_config("series")

    fs, gbests, hp, hf, hpb = perform_run_wrapper(
        dummy_fitness,
        config,
    )

    assert fs.shape == (4,)
    assert gbests.shape == (4, 3)

    assert hp.shape[0] == 4
    assert hf.shape[0] == 4
    assert hpb.shape[0] == 4

    assert mock_run.call_count == 4


@patch(
    "parallel_wrappers.wrap_perform_pso_run._perform_run_wrapper2",
    side_effect=fake_perform_run,
)
def test_serial_result_order(mock_run):
    """
    @brief Verify returned results preserve run ordering.
    """

    config = build_config("series")

    fs, gbests, *_ = perform_run_wrapper(
        dummy_fitness,
        config,
    )

    assert np.allclose(fs, [0, 1, 2, 3])

    for i in range(4):
        assert np.allclose(
            gbests[i],
            np.full(3, i),
        )


@patch(
    "parallel_wrappers.wrap_perform_pso_run.ProcessPoolExecutor",
    DummyExecutor,
)
@patch(
    "parallel_wrappers.wrap_perform_pso_run._run_single",
    side_effect=lambda args: fake_perform_run(*args),
)
def test_cpu_parallel_execution(mock_run_single):
    """
    @brief Verify CPU parallel execution branch.

    Confirms that:

    - The CPU-parallel branch is selected.
    - All runs are executed.
    - Returned results preserve run ordering.
    - The ProcessPoolExecutor interface is used correctly.
    """

    config = build_config("cpu")

    fs, gbests, hp, hf, hpb = perform_run_wrapper(
        dummy_fitness,
        config,
    )

    assert mock_run_single.call_count == config.N_runs

    assert fs.shape == (config.N_runs,)
    assert gbests.shape == (config.N_runs, config.N_x)

    assert np.allclose(fs, np.arange(config.N_runs))

    for i in range(config.N_runs):
        assert np.allclose(gbests[i], np.full(config.N_x, float(i)))


def test_invalid_parallel_mode():
    """
    @brief Verify unsupported execution modes raise an exception.
    """

    config = build_config("banana")

    with pytest.raises(NotImplementedError):
        perform_run_wrapper(
            dummy_fitness,
            config,
        )
        
        
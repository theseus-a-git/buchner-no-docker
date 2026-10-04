# -*- coding: utf-8 -*-
"""
@file test_input_validation.py

@brief Unit tests for validation of user-supplied PSO fitness functions.

This module contains pytest tests for the `check_fitness_func` function
defined in `input_validation.py`.

The tests verify:

- Valid module-level fitness functions are accepted.
- Non-callable objects raise `TypeError`.
- Nested functions are rejected when multiprocessing requires pickling.
- Nested functions are accepted when pickling is not required.
- MPI execution enforces the same pickling requirements as CPU execution.

These tests intentionally use lightweight mock configuration objects since
only the parallelization options are required by `check_fitness_func`.

@author
Ben Carlson

@date
2026-06-30
"""

from dataclasses import dataclass

import pytest

from utils.input_validation import check_fitness_func


@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object used for testing.

    Only the configuration members accessed by
    `check_fitness_func()` are included.

    Members:
    - run_parallel_mode
    - fitness_parallel_mode
    """

    run_parallel_mode: str
    fitness_parallel_mode: str


def module_scope_fitness(x):
    """
    @brief Simple module-level fitness function.

    This function is intentionally defined at module scope so that it
    satisfies multiprocessing pickling requirements.
    """
    return x


def test_valid_module_scope_function():
    """
    @brief Verify a valid module-level function passes validation.

    Confirms that no exception is raised when the supplied fitness
    function is callable and pickleable.
    """

    config = MockPSOConfig(
        run_parallel_mode="serial",
        fitness_parallel_mode="serial",
    )

    check_fitness_func(module_scope_fitness, config)


def test_non_callable_raises_type_error():
    """
    @brief Verify non-callable objects are rejected.

    Confirms that supplying a non-callable object raises a
    `TypeError`.
    """

    config = MockPSOConfig(
        run_parallel_mode="serial",
        fitness_parallel_mode="serial",
    )

    with pytest.raises(TypeError):
        check_fitness_func(12345, config)


def test_nested_function_rejected_for_cpu_parallel():
    """
    @brief Verify nested functions are rejected for CPU execution.

    Confirms that multiprocessing-safe validation rejects functions
    that are not defined at module scope.
    """

    config = MockPSOConfig(
        run_parallel_mode="cpu",
        fitness_parallel_mode="serial",
    )

    def nested_function(x):
        return x

    with pytest.raises(TypeError):
        check_fitness_func(nested_function, config)


def test_nested_function_rejected_for_mpi_parallel():
    """
    @brief Verify nested functions are rejected for MPI execution.

    Confirms that MPI execution enforces the same module-scope
    requirement as CPU execution.
    """

    config = MockPSOConfig(
        run_parallel_mode="mpi",
        fitness_parallel_mode="serial",
    )

    def nested_function(x):
        return x

    with pytest.raises(TypeError):
        check_fitness_func(nested_function, config)


def test_nested_function_rejected_for_cpu_fitness_parallel():
    """
    @brief Verify nested functions are rejected for CPU fitness
    parallelization.

    Confirms that enabling CPU-based fitness evaluation also requires
    module-scope pickleable functions.
    """

    config = MockPSOConfig(
        run_parallel_mode="serial",
        fitness_parallel_mode="cpu",
    )

    def nested_function(x):
        return x

    with pytest.raises(TypeError):
        check_fitness_func(nested_function, config)


def test_nested_function_allowed_without_pickling():
    """
    @brief Verify nested functions are accepted when pickling is not
    required.

    Confirms that nested functions may be used during purely serial
    execution.
    """

    config = MockPSOConfig(
        run_parallel_mode="serial",
        fitness_parallel_mode="serial",
    )

    def nested_function(x):
        return x

    check_fitness_func(nested_function, config)
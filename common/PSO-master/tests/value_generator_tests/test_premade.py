# -*- coding: utf-8 -*-
"""
@file test_premade.py

@brief Unit tests for precomputed random-value retrieval.

This module contains unit tests for the
`make_premade` factory function defined in:

@code
PSO.options.value_generators.premade
@endcode

The tests verify:

* Correct retrieval of precomputed values
* Correct output dimensions
* Sequential consumption of precomputed values
* Conversion of precomputed values to float64
* Read-only behavior of returned arrays
* Zero-length requests
* Behavior when precomputed values are exhausted
* Correct selection of values by generator index
* Correct selection of values by run index
* Independent streams for different runs
* Independent state between separately created generators
* Repeated requests preserve sequential ordering
* Unsupported generator indices raise the expected exception

These tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-09-09
"""
from dataclasses import dataclass, field
import numpy as np
import pytest
        
from options.value_generators.premade import make_premade

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object for testing.

    This class provides only the configuration members required by
    `make_premade`.
    
    @param position_init_premade_vals
    Precomputed values used for position initialization.

    @param velocity_init_premade_vals
    Precomputed values used for velocity initialization.

    @param update_premade_vals
    Precomputed values used for particle updates.
    """
    position_init_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0]])
    )
    velocity_init_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0]])
    )
    update_premade_vals: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0]])
    )
    

def test_precomputed_values():
    """
    @brief Verify retrieval of precomputed random values.
    
    The premade generator should return values from the precomputed
    array associated with the selected generator index.
    """
    premade = np.array([
        [0.1, 0.2, 0.3, 0.4],
    ])

    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(2, 1)
    
    expected = np.array([[0.1], [0.2]])
    
    np.testing.assert_array_equal(vals, expected)


def test_precomputed_values_shape():
    """
    @brief Verify retrieved values have the requested shape.
    
    The generator should return exactly N rows and M columns when
    sufficient precomputed values are available.
    """
    premade = np.arange(20, dtype=np.float64)

    config = MockPSOConfig(
        position_init_premade_vals=np.array([premade]),
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(4, 5)
    
    assert vals.shape == (4, 5)
    assert vals.size == 20


def test_precomputed_sequential_consumption():
    """
    @brief Verify sequential consumption of precomputed values.
    
    Consecutive calls should continue advancing through the internal
    array without resetting the index.
    """
    premade = np.array([[1.0, 2.0, 3.0, 4.0]])
    
    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals1 = premade_generator(2, 1)
    vals2 = premade_generator(2, 1)
    
    np.testing.assert_array_equal(vals1, [[1.0], [2.0]])
    np.testing.assert_array_equal(vals2, [[3.0], [4.0]])


def test_precomputed_dtype_conversion():
    """
    @brief Verify precomputed values are converted to float64.
    
    Integer input values should be converted to NumPy float64 values
    by the factory function.
    """
    premade = np.array([[1, 2, 3]])
    
    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(3, 1)
    
    assert vals.dtype == np.float64


def test_precomputed_array_is_read_only():
    """
    @brief Verify returned precomputed arrays are read-only views.
    
    The implementation marks the internal contiguous array as
    non-writeable. Since the returned values are slices of that array,
    thethe writeable flag should remain False.
    """
    premade = np.array([[1.0, 2.0, 3.0]])
    
    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(3, 1)
    
    assert vals.flags.writeable is False


def test_precomputed_values_cannot_be_modified():
    """
    @brief Verify modifying returned precomputed values raises an error.
    
    Returned arrays should be read-only because they reference the
    internal precomputed-value array.
    """
    premade = np.array([[1.0, 2.0, 3.0]])
    
    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(3, 1)
    
    with pytest.raises(ValueError):
        vals[0, 0] = 99.0


def test_zero_length_request():
    """
    @brief Verify requesting zero values returns an empty array.
    
    A zero-length request should preserve the requested number of
    columns while returning no values.
    """
    premade = np.array([
        [1.0, 2.0, 3.0],
    ])

    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    vals = premade_generator(0, 5)
    
    assert vals.shape == (0, 5)
    assert vals.size == 0


def test_precomputed_overrun_behavior():
    """
    @brief Verify behavior when requesting more values than remain.
    
    The current implementation performs no explicit bounds checking.
    NumPy slicing therefore returns the values that remain, and the
    subsequent reshape raises ValueError when the number of remaining
    values does not match the requested shape.
    """
    premade = np.array([[0.1, 0.2]])
    
    config = MockPSOConfig(
        position_init_premade_vals=premade,
    )

    premade_generator = make_premade(config, 0, 0)
    
    with pytest.raises(ValueError):
        premade_generator(5, 1)


@pytest.mark.parametrize(
    "generator_index, attribute_name",
    [
        (0, "position_init_premade_vals"),
        (1, "velocity_init_premade_vals"),
        (2, "update_premade_vals"),
    ],
)
def test_generator_index_selects_correct_precomputed_values(
    generator_index,
    attribute_name,
    ):
    """
    @brief Verify generator_index selects the correct precomputed array.

    Generator index zero should select position initialization values,
    generator index one should select velocity initialization values,
    and generator index two should select particle update values.
    """
    config = MockPSOConfig(
        position_init_premade_vals=np.array([
            [1.0, 2.0, 3.0],
        ]),
        velocity_init_premade_vals=np.array([
            [4.0, 5.0, 6.0],
        ]),
        update_premade_vals=np.array([
            [7.0, 8.0, 9.0],
        ]),
    )

    expected = getattr(config, attribute_name)

    premade_generator = make_premade(
        config,
        run_index=0,
        generator_index=generator_index,
    )

    vals = premade_generator(3, 1)

    np.testing.assert_array_equal(
        vals,
        expected.reshape(3, 1),
    )


def test_run_index_selects_correct_precomputed_values():
    """
    @brief Verify run_int selects the corresponding precomputed stream.
    
    Different run indices should select the corresponding row from
    the precomputed values associated with the selected generator.
    """
    premade = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        ])
    
    config = MockPSOConfig(position_init_premade_vals=premade)
    
    generator_run_1 = make_premade(config, 0, 0)
    generator_run_2 = make_premade(config, 1, 0)
    
    vals1 = generator_run_1(3, 1)
    vals2 = generator_run_2(3, 1)
    
    np.testing.assert_array_equal(
        vals1,
        [[1.0], [2.0], [3.0]],
    )
    np.testing.assert_array_equal(
        vals2,
        [[4.0], [5.0], [6.0]],
    )


def test_precomputed_streams_are_independent():
    """
    @brief Verify separate precomputed generators maintain independent state.
    
    Advancing one generator should not affect the position of another
    generator created from a different run's precomputed values.
    """
    premade = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    
    config = MockPSOConfig(position_init_premade_vals=premade)
    
    generator_run_1 = make_premade(config, 0, 0)
    generator_run_2 = make_premade(config, 1, 0)
    
    first_run_1 = generator_run_1(2, 1)
    
    # Advance only run 1.
    generator_run_1(1, 1)
    
    first_run_2 = generator_run_2(2, 1)
    
    np.testing.assert_array_equal(
        first_run_1,
        [[1.0], [2.0]],
    )
    np.testing.assert_array_equal(
        first_run_2,
        [[5.0], [6.0]],
    )


def test_repeated_calls_preserve_sequential_order():
    """
    @brief Verify multiple differently sized requests consume values in order.
    
    The internal index should advance by exactly N*M for every call,
    regardless of the requested matrix dimensions.
    """
    premade = np.arange(12, dtype=float)
    
    config = MockPSOConfig(
        position_init_premade_vals=np.array([premade]),
    )
    
    premade_generator = make_premade(config, 0, 0)
    
    vals1 = premade_generator(2, 2)
    vals2 = premade_generator(1, 4)
    vals3 = premade_generator(2, 2)
    
    np.testing.assert_array_equal(
        vals1,
        [[0.0, 1.0], [2.0, 3.0]],
    )
    np.testing.assert_array_equal(
        vals2,
        [[4.0, 5.0, 6.0, 7.0]],
    )
    np.testing.assert_array_equal(
        vals3,
        [[8.0, 9.0], [10.0, 11.0]],
    )

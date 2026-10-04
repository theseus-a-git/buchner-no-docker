# -*- coding: utf-8 -*-
"""
@file test_random_uniform.py
@brief Unit tests for uniform random-value generation.

This module contains unit tests for the
`make_random_uniform` factory function defined in:

@code
PSO.options.value_generators.random_uniform
@endcode

The tests verify:
    
* Correct generation of random values with requested dimensions
* Values are within the interval [0, 1)
* Reproducibility using integer seeds
* Independence of RNG streams using different seeds
* Independent state between separately created generators
* Zero-length requests
* Generation across multiple dimensions
* Reproducibility using SeedSequence objects
* Correct selection of the run-specific seed
* Correct selection of the generator-specific seed
* Continuation of the random sequence across multiple calls

These tests are designed for use with pytest.

@author
Ben Carlson

@date
2026-09-09
"""
from dataclasses import dataclass, field
import numpy as np

from options.value_generators.random_uniform import make_random_uniform

@dataclass
class MockPSOConfig:
    """
    @brief Lightweight configuration object for testing.

    This class provides only the configuration members required by
    `make_random_uniform`.
    
    @param seed_vals
    Two-dimensional collection containing the seed associated with each
    generator and PSO run.

    The first index selects the generator and the second index selects
    the PSO run.
    """
    seed_vals: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array([0, 1], dtype=np.uint64),
        ]
    )


def test_random_generation_shape():
    """
    @brief Verify generated random array has the requested shape.
    
    The returned callable should generate an array containing N rows
    and M columns, as specified by the PSO value-generator interface.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    random_uniform = make_random_uniform(config, 0, 0)
    
    vals = random_uniform(10, 3)
    
    assert vals.shape == (10, 3)


def test_random_generation_range():
    """
    @brief Verify generated random values lie in the interval [0, 1).
    
    NumPy's Generator.random() method should produce floating-point
    values greater than or equal to zero and strictly less than one.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    random_uniform = make_random_uniform(config, 0, 0)
    
    vals = random_uniform(100, 5)
    
    assert np.all(vals >= 0.0)
    assert np.all(vals < 1.0)


def test_seed_reproducibility():
    """
    @brief Verify identical seeds produce identical random sequences.
    
    Two generators initialized with the same seed should produce
    identical random values.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([42], dtype=np.uint64),
        ],
    )

    rng1 = make_random_uniform(config, 0, 0)
    rng2 = make_random_uniform(config, 0, 0)
    
    vals1 = rng1(20, 5)
    vals2 = rng2(20, 5)
    
    np.testing.assert_array_equal(vals1, vals2)


def test_seed_independence():
    """
    @brief Verify different seeds produce different random sequences.
    
    Generators initialized with different seeds should not produce
    identical random sequences.
    """
    config1 = MockPSOConfig(
        seed_vals=[
            np.array([1], dtype=np.uint64),
        ],
    )
    config2 = MockPSOConfig(
        seed_vals=[
            np.array([2], dtype=np.uint64),
        ],
    )

    rng1 = make_random_uniform(config1, 0, 0)
    rng2 = make_random_uniform(config2, 0, 0)
    
    vals1 = rng1(20, 5)
    vals2 = rng2(20, 5)
    
    assert not np.array_equal(vals1, vals2)


def test_rng_streams_are_independent():
    """
    @brief Verify generators maintain independent RNG state.
    
    Creating multiple generators with the same configuration should
    create independent NumPy Generator instances. Advancing one
    generator must not affect the sequence produced by another
    generator initialized with the same seed.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    rng1 = make_random_uniform(config, 0, 0)
    rng2 = make_random_uniform(config, 0, 0)
    
    first_vals = rng1(10, 3)
    
    # Advance only rng1.
    rng1(10, 3)
    
    second_vals = rng2(10, 3)
    
    # rng2 should still produce the first sequence.
    np.testing.assert_array_equal(first_vals, second_vals)


def test_zero_length_request():
    """
    @brief Verify requesting zero random values returns an empty array.
    
    A zero-length request should preserve the requested number of
    columns while returning no rows.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    random_uniform = make_random_uniform(config, 0, 0)
    
    vals = random_uniform(0, 5)
    
    assert vals.shape == (0, 5)
    assert vals.size == 0


def test_random_generation_multiple_dimensions():
    """
    @brief Verify generated values correctly fill all requested dimensions.
    
    This test verifies that the generator produces N*M values and
    returns them as the requested two-dimensional array.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    random_uniform = make_random_uniform(config, 0, 0)
    
    vals = random_uniform(4, 7)
    
    assert vals.shape == (4, 7)
    assert vals.size == 28
    assert np.all(vals >= 0.0)
    assert np.all(vals < 1.0)


def test_seed_sequence_reproducibility():
    """
    @brief Verify SeedSequence objects provide reproducible streams.
    
    Two generators initialized from equivalent SeedSequence objects
    should produce identical random sequences.
    """
    seed1 = np.random.SeedSequence(12345)
    seed2 = np.random.SeedSequence(12345)
    
    config1 = MockPSOConfig(
        seed_vals=[
            np.array([seed1], dtype=object),
        ],
    )
    config2 = MockPSOConfig(
        seed_vals=[
            np.array([seed2], dtype=object),
        ],
    )

    rng1 = make_random_uniform(config1, 0, 0)
    rng2 = make_random_uniform(config2, 0, 0)
    
    np.testing.assert_array_equal(rng1(20, 5), rng2(20, 5))
    

def test_run_index_selects_correct_seed():
    """
    @brief Verify run_int selects the corresponding seed.
    
    Different run indices should use their corresponding seeds from
    the selected generator's seed array, resulting in different
    random-number streams.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123, 456], dtype=np.uint64),
        ],
    )

    rng1 = make_random_uniform(config, 0, 0)
    rng2 = make_random_uniform(config, 1, 0)
    
    vals1 = rng1(20, 5)
    vals2 = rng2(20, 5)
    
    assert not np.array_equal(vals1, vals2)
    
    
def test_generator_index_selects_correct_seed_array():
    """
    @brief Verify generator_index selects the corresponding seed array.
    
    Different generator indices should select different seed arrays
    while using the same run index. This verifies that independent
    value generators receive independent random-number streams.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
            np.array([456], dtype=np.uint64),
        ],
    )
    
    rng1 = make_random_uniform(config, 0, 0)
    rng2 = make_random_uniform(config, 0, 1)
    
    vals1 = rng1(20, 5)
    vals2 = rng2(20, 5)
    
    assert not np.array_equal(vals1, vals2)


def test_repeated_calls_continue_rng_sequence():
    """
    @brief Verify consecutive calls continue the same RNG stream.
    
    Calling the returned generator multiple times should advance the
    underlying NumPy Generator rather than restarting the sequence.

    The combined output from two consecutive calls should therefore
    match the output from a single call requesting the same total
    number of values.
    """
    config = MockPSOConfig(
        seed_vals=[
            np.array([123], dtype=np.uint64),
        ],
    )

    rng1 = make_random_uniform(config, 0, 0)
    rng2 = make_random_uniform(config, 0, 0)
    
    first = rng1(10, 3)
    second = rng1(10, 3)
    
    expected = rng2(20, 3)
    
    np.testing.assert_array_equal(
        first,
        expected[:10],
    )
    np.testing.assert_array_equal(
        second,
        expected[10:],
    )

# -*- coding: utf-8 -*-
"""
@file config.py
@brief Configuration container for Particle Swarm Optimization (PSO).

Defines the validated configuration object used throughout the PSO
framework. All optional optimizer parameters are centralized here.

This class is intended to be instantiated from user-provided keyword
arguments inside the public `run_PSO()` API.

Example
-------
cfg = PSOConfig(
    N_runs=4,
    N_iterations=1000,
    boundary_conditions=np.array([ [-10.0, 10.0], [-10.0, 10.0] ])
    N_particles=40,
    eq_type="constriction"
)

@author
Ben Carlson

@date
2026-05-19
"""
from dataclasses import dataclass, field, fields
import numbers
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np
import os

from utils.dispatch_tables import SUPPORTED_BOUNDARY_TYPES, SUPPORTED_VALUE_GENERATORS


def validate_boundary_conditions(boundary_conditions) -> np.ndarray:
    """
    @brief Validate and convert a matrix of boundary conditions.

    @param boundary_conditions
    The matrix boundary conditions to be used in PSO.
    
    @return
    The validated matrix of type np.float64.

    @note
    Requirements for the random number matrix:
        - ordered iterable
        - numeric
        - finite
        - 2D array: (n_dimensions, 2)
        - Each pair must be [lower_bound, higher_bound]
        
    @throws TypeError
    Raised if parameter type is invalid.

    @throws ValueError
    Raised if parameter value is invalid.
    """
    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------
    
    try:
        arr = np.asarray(boundary_conditions, dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            "boundary_conditions could not be converted to float64 ndarray"
        ) from exc
    
    # --------------------------------------------------------
    # DIMENSION CHECK
    # --------------------------------------------------------
    
    if arr.ndim != 2:
        raise ValueError("boundary_conditions must be 2D")
        
    # --------------------------------------------------------
    # SHAPE CHECK
    # --------------------------------------------------------
        
    if arr.shape[1] != 2:
        raise ValueError("boundary_conditions must have shape (N, 2)")
        
    # --------------------------------------------------------
    # EMPTY CHECK
    # --------------------------------------------------------    
    
    if arr.shape[0] == 0:
        raise ValueError("boundary_conditions must contain at least one pair")
    
    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------
    
    if not np.all(np.isfinite(arr)):
        raise ValueError("boundary_conditions must contain only finite values")
    
    # --------------------------------------------------------
    # ORDER CHECK
    # --------------------------------------------------------
    
    lower = arr[:, 0]
    upper = arr[:, 1]
    invalid = lower >= upper
    if np.any(invalid):
        bad_rows = np.where(invalid)[0]
        raise ValueError(
            "boundary_conditions contains invalid bounds "
            f"at rows {bad_rows.tolist()} (lower must be < upper)")
        
    # --------------------------------------------------------
    # RETURN NUMPY ARRAY
    # --------------------------------------------------------
    
    arr = np.ascontiguousarray( arr, dtype=np.float64 )
    arr.flags.writeable = False
    return arr
    

def validate_random_sequence(premade_vals, 
                             pos_init_type: str,
                             vel_init_type: str,
                             iter_type: str,
                             N_runs: int,
                             N_iterations: int,
                             N_x: int,
                             N_particles: int,
                             generator_val: int
                             ) -> np.ndarray:
    """
    @brief Validate and convert a matrix of random values.

    @param premade_vals
    The matrix of pre-computed random numbers to be used in PSO.
    
    @param pos_init_type
    The value generator type for the particle initialization positions.
    
    @param vel_init_type
    The value generator type for the particle initialization velocities.
    
    @param iter_type
    The value generator type for the particle update iterations.
    
    @param N_runs
    The number of PSO runs to perform.
    
    @param N_iterations
    Number of iterations to perform each run.
    
    @param N_x
    Number of dimensions of the search space.
    
    @param N_particles
    Number of particles in the swarm.
    
    @param generator_val
    Index of the generator. Used to access the correct premade_vals inside the
    value_generator factory function and to validate correct length here.

    @return
    The validated matrix of type np.float64.

    @note
    Requirements for the random number matrix:
        - ordered iterable
        - numeric
        - finite
        - values in [0, 1]
        - 2D array: (n_runs, n_random_values)
        - length must equal N_runs
        - Number of random vals must equal number of required vals.
        
    @throws TypeError
    Raised if parameter type is invalid.

    @throws ValueError
    Raised if parameter value is invalid.
    
    @throws NotImplementedError
    Raised if received generator_val is not supported.
    """

    # --------------------------------------------------------
    # NONE CHECK
    # --------------------------------------------------------

    if premade_vals is None:
        if pos_init_type == "premade" and generator_val == 0:
            raise ValueError("position_init_premade_vals must be given if "
                             "position_init_value_type='premade'")
        if vel_init_type == "premade" and generator_val == 1:
            raise ValueError("velocity_init_premade_vals must be given if "
                             "velocity_init_value_type='premade'")
        if iter_type == "premade" and generator_val == 2:
            raise ValueError("update_premade_vals must be given if "
                             "update_value_type='premade'")
        return None

    # --------------------------------------------------------
    # NOT NONE BUT ALSO NOT 'premade' WARNING
    # --------------------------------------------------------

    if pos_init_type != "premade" and generator_val == 0:
        print("WARNING: Received pre-computed values will not be used unless "
              "position_init_value_type='premade'.")
        return None
        
    if vel_init_type != "premade" and generator_val == 1:
        print("WARNING: Received pre-computed values will not be used unless "
              "velocity_init_value_type='premade'.")
        return None
        
    if iter_type != "premade" and generator_val == 2:
        print("WARNING: Received pre-computed values will not be used unless "
              "update_value_type='premade'.")
        return None

    # --------------------------------------------------------
    # REJECT STRINGS
    # --------------------------------------------------------

    if isinstance(premade_vals, (str, bytes)):
        raise TypeError("premade_vals must not be a string")
        
    # --------------------------------------------------------
    # CONSISTENCY CHECK: ALL ROWS SAME LENGTH
    # --------------------------------------------------------
    
    try:
        row_lengths = [len(row) for row in premade_vals]
    except:
        raise ValueError("premade_vals must be 2D and iterable")    
    if len(set(row_lengths)) != 1:
        raise ValueError(
            "All rows in premade_vals must have the same length"
        )

    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------

    try:
        arr = np.asarray(premade_vals, dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            "premade_vals could not be converted to float64 ndarray"
            ) from exc

    # --------------------------------------------------------
    # DIMENSION CHECK
    # --------------------------------------------------------

    if arr.ndim != 2:
        raise ValueError("premade_vals must be 2D matrix of shape "
                         "(N_runs, ...)")

    # --------------------------------------------------------
    # EMPTY CHECK
    # --------------------------------------------------------

    if arr.size == 0:
        raise ValueError("premade_vals must not be empty")
        
    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------
    
    if len(arr) != N_runs:
        raise ValueError(f"premade_vals must have N_runs rows. {N_runs} runs "
                         f"were requested, but {len(arr)} rows were received.")

    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------

    if not np.all(np.isfinite(arr)):
        raise ValueError("premade_vals must contain only finite values")

    # --------------------------------------------------------
    # RANGE CHECK
    # --------------------------------------------------------

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("premade_vals values must be in [0, 1]")
        
    # --------------------------------------------------------
    # AMOUNT CHECK
    # --------------------------------------------------------
    
    actual_length = len(arr[0])
    correct_length = 0
    # position initialization
    if generator_val == 0: 
        correct_length += N_x * N_particles
    # velocity initialization
    elif generator_val == 1: 
        correct_length += N_x * N_particles
    # iteration updates
    elif generator_val == 2:     
        correct_length += 2 * N_iterations * N_x * N_particles
    else:
        raise NotImplementedError("Received generator_val is not supported.")
        
    if (actual_length < correct_length):
        raise ValueError("Incorrect number of pre-computed values received. "
            f"Expected {correct_length} per run, but received {actual_length}")
    elif (actual_length > correct_length):
        print("WARNING: Number of pre-computed values does not match expected"
              f" length. Expected {correct_length} per run, "
              f"but received {actual_length}")

    # --------------------------------------------------------
    # RETURN NUMPY ARRAY
    # --------------------------------------------------------

    return arr

# =============================================================================
#     return np.ascontiguousarray(
#         arr,
#         dtype=np.float64
#     )
# =============================================================================


def validate_entropy_vals(entropy_vals) -> np.ndarray:
    """
    @brief Validate and convert entropy values used to seed RNGs.

    @param entropy_vals
    Sequence containing the entropy values used to seed the value generators.
    Exactly three positive integers are required, with one entropy value for
    each random value generator.

    @return
    The validated entropy values as a one-dimensional numpy.ndarray of type
    np.int64.

    @throws TypeError
    Raised if entropy_vals is not an ordered iterable of integers.

    @throws ValueError
    Raised if entropy_vals does not contain exactly three values or if any
    entropy value is not positive.

    @note
    Requirements for entropy_vals:
        - ordered iterable
        - exactly 3 values
        - integer values
        - positive values (> 0)
        - returned as a 1D numpy array of type np.int64
    """

    # --------------------------------------------------------
    # REJECT STRINGS
    # --------------------------------------------------------

    if isinstance(entropy_vals, (str, bytes)):
        raise TypeError("entropy_vals must not be a string")

    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------

    try:
        vals = np.asarray(entropy_vals, dtype=object)
    except Exception as exc:
        raise TypeError(
            "entropy_vals could not be converted to a numpy array"
        ) from exc

    # --------------------------------------------------------
    # DIMENSION CHECK
    # --------------------------------------------------------

    if vals.ndim != 1:
        raise ValueError("entropy_vals must be a 1D sequence")

    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(vals) != 3:
        raise ValueError(
            "entropy_vals must contain exactly 3 values. "
            f"Received {len(vals)} values."
        )

    # --------------------------------------------------------
    # INTEGER CHECK
    # --------------------------------------------------------

    for i, val in enumerate(vals):
        if isinstance(val, (bool, np.bool_)) or \
            not isinstance(val, (int, np.integer)):
                
            raise TypeError(
                f"entropy_vals[{i}] must be an int. "
                f"Received type {type(val)}"
            )

    # --------------------------------------------------------
    # POSITIVE CHECK
    # --------------------------------------------------------

    if np.any(vals <= 0):
        raise ValueError(
            "entropy_vals must contain only positive integers (> 0)"
        )

    # --------------------------------------------------------
    # CONVERT TO NUMPY ARRAY
    # --------------------------------------------------------

    return np.asarray(vals, dtype=np.int64)


def _validate_generator_seeds(seed_spec,
                              N_runs: int,
                              entropy_val: float
                              ) -> np.ndarray | list[np.random.SeedSequence]:
    """
    @brief Validate and normalize the seed specification for one RNG generator.

    @param seed_spec
    Seed specification for one generator.
        False:
            Generate nondeterministic SeedSequence objects.

        True:
            Generate deterministic SeedSequence objects.

        sequence of ints:
            One explicit integer seed per run.

        sequence of SeedSequence:
            One explicit SeedSequence per run.

    @param N_runs
    Number of PSO runs.

    @param entropy_val
    The default entropy value for the master spawn used when `seed_spec=True`.

    @return
    Array of normalized seeds, one per PSO run.
    
    @throws TypeError
    Raised if parameter type is invalid.

    @throws ValueError
    Raised if parameter value is invalid.
    """

    # --------------------------------------------------------
    # BOOLEAN CASE
    # --------------------------------------------------------

    if isinstance(seed_spec, (bool, np.bool_)):

        if not seed_spec:
            # No reproducible seed requested.
            master = np.random.SeedSequence()
            return master.spawn(N_runs)

        # Reproducible automatically-generated seeds.
        master = np.random.SeedSequence(entropy_val)
        return master.spawn(N_runs)

    # --------------------------------------------------------
    # CONVERT TO ARRAY / SEQUENCE
    # --------------------------------------------------------

    if isinstance(seed_spec, np.ndarray):
        arr = seed_spec
    else:
        try:
            arr = np.asarray(seed_spec, dtype=object)
        except Exception as exc:
            raise TypeError(
                "Seed specification must be a bool, sequence of integer "
                "seeds, or sequence of SeedSequence objects."
            ) from exc

    # --------------------------------------------------------
    # DIMENSION CHECK
    # --------------------------------------------------------

    if arr.ndim != 1:
        raise ValueError(
            "Each generator's seed specification must be 1D."
        )

    # --------------------------------------------------------
    # LENGTH CHECK
    # --------------------------------------------------------

    if len(arr) != N_runs:
        raise ValueError(
            f"Each generator must have N_runs seeds. "
            f"{N_runs} runs were requested, but {len(arr)} "
            f"seeds were received."
        )

    # --------------------------------------------------------
    # EMPTY CHECK
    # --------------------------------------------------------

    if arr.size == 0:
        raise ValueError("Seed specification must not be empty.")

    # --------------------------------------------------------
    # SEEDSEQUENCE CASE
    # --------------------------------------------------------

    if all(isinstance(seed, np.random.SeedSequence) for seed in arr):
        return list(arr)

    # --------------------------------------------------------
    # MIXED SeedSequence / INTEGER CHECK
    # --------------------------------------------------------

    if any(isinstance(seed, np.random.SeedSequence) for seed in arr):
        raise TypeError(
            "A generator's seeds must contain either all integer seeds "
            "or all numpy.random.SeedSequence objects. Mixing the two "
            "types within one generator is not supported."
        )

    # --------------------------------------------------------
    # INTEGER SEED CASE
    # --------------------------------------------------------

    if not all(isinstance(seed, (int, np.integer)) for seed in arr):
        raise TypeError(
            "Seeds must contain integers or SeedSequence objects."
        )
    
    int_arr = np.asarray(arr, dtype=np.int64)

    # --------------------------------------------------------
    # RANGE CHECK
    # --------------------------------------------------------

    if np.any(int_arr < 0):
        raise ValueError(
            "Seed values must be non-negative."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return int_arr.astype(np.uint64)


def validate_seeds(seed_vals,
                   N_runs: int,
                   entropy_vals: np.ndarray,
                   ) -> tuple[
                        np.ndarray | list[np.random.SeedSequence],
                        np.ndarray | list[np.random.SeedSequence],
                        np.ndarray | list[np.random.SeedSequence],
                    ]:
    """
    @brief Validate and normalize seed specifications for the three
    independent PSO random-value generators.

    @param seed_vals
    Seed specification.

    A single bool applies to all three generators.

    A 3-element sequence specifies the seed behavior independently
    for position initialization, velocity initialization, and
    particle updates.

    Each element may be:

        False
            Use nondeterministic seeds.

        True
            Generate deterministic seeds.

        sequence of integers
            One explicit seed per run.

        sequence of SeedSequence objects
            One explicit SeedSequence per run.

    @param N_runs
    Number of PSO runs.
    
    @param entropy_vals
    The array of entropy values to seed the generators.

    @return
    (position_seeds, velocity_seeds, update_seeds)
    
    @throws TypeError
    Raised if parameter type is invalid.

    @throws ValueError
    Raised if parameter value is invalid.
    """

    # --------------------------------------------------------
    # SINGLE BOOLEAN
    # --------------------------------------------------------

    if isinstance(seed_vals, (bool, np.bool_)):
        specs = [bool(seed_vals)] * 3

    # --------------------------------------------------------
    # THREE GENERATOR SPECIFICATIONS
    # --------------------------------------------------------

    else:
        try:
            specs = list(seed_vals)
        except Exception as exc:
            raise TypeError(
                "seed_vals must be a bool or a sequence containing "
                "three seed specifications."
            ) from exc
        if len(specs) != 3:
            raise ValueError(
                "seed_vals must contain exactly three specifications: "
                "[position_init, velocity_init, update]."
            )

    # --------------------------------------------------------
    # VALIDATE EACH GENERATOR
    # --------------------------------------------------------

    position_seeds = _validate_generator_seeds(
        specs[0],
        N_runs,
        entropy_vals[0]
    )

    velocity_seeds = _validate_generator_seeds(
        specs[1],
        N_runs,
        entropy_vals[1]
    )

    update_seeds = _validate_generator_seeds(
        specs[2],
        N_runs,
        entropy_vals[2]
    )

    return (position_seeds, velocity_seeds, update_seeds)



def validate_parallelization(run_mode, run_jobs, fit_mode, fit_jobs, N_runs, N_particles):

    # --------------------------------------------------------
    # Detect available hardware
    # --------------------------------------------------------
    
    available_cpu_cores = int(os.environ.get(
                                                "SLURM_CPUS_PER_TASK",
                                                os.cpu_count() or 1,)
                                        )
    
    available_gpus = 0
    if fit_mode in ("cuda-cupy", "cuda-torch"):
        import cupy as cp
        try:
            available_gpus = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            available_gpus = 0
        if available_gpus == 0:
            raise RuntimeError(
                "fitness_parallel_mode='cuda-cupy' "
                "requires at least one visible CUDA GPU."
            )

    # --------------------------------------------------------
    # Select Parallelization Parameters
    # --------------------------------------------------------

    # run_mode == "cpu"

    if run_mode == "cpu":
        max_run_jobs = min(N_runs, available_cpu_cores)
        min_batches_achievable = int(np.ceil(N_runs / available_cpu_cores))
        min_core_count = int(np.ceil(N_runs / min_batches_achievable))
        
        # runs and fitness do not compete for cpu cores,
        if fit_mode == "series":
    
            # Choose values automatically
            if run_jobs is None:
                run_jobs = min_core_count
            # Value already chosen by user
            elif (run_jobs > max_run_jobs):
                print(f"WARNING: run_n_jobs changed to {max_run_jobs}.")
                run_jobs = max_run_jobs
            # Otherwise value chosen by user is fine

        elif fit_mode in ("cuda-cupy", "cuda-torch", "gpu"):
            if fit_jobs is not None:
                print("WARNING: fitness_n_jobs ignored for"
                      " GPU fitness evaluation.")
                
            if (available_gpus == 1):
                print("Warning: run_parallel_mode changed to 'series' because"
                      " only 1 gpu available.")
                run_mode = 'series'
# =============================================================================
#             ''' new section '''
#             max_run_jobs = min(N_runs, available_gpus,)
#             elif run_jobs is None:
#                 run_jobs = max_run_jobs
#             elif run_jobs > max_run_jobs:
#                 print(f"WARNING: run_n_jobs changed to {max_run_jobs} "
#                     "because cannot exceed the number of visible GPUs."
#                     f"{available_gpus} GPUs available.")
#                 run_jobs = max_run_jobs
#             ''' end new '''
# =============================================================================
                
        
        # runs and fitness do compete for cpu cores,
        elif fit_mode == "cpu":
    
            # User did not provide fitness_n_jobs
            if fit_jobs is None:
                # Choose both values automatically
                if run_jobs is None:
                    # Prioritize parallel runs if both request `cpu`.
                    run_jobs = min_core_count
                # User provided run_n_jobs. check it is valid.
                elif run_jobs > max_run_jobs:
                    print(f"WARNING: run_n_jobs changed to {max_run_jobs}.")
                    run_jobs = max_run_jobs
                
                # Use remaining cores for fitness
                min_fit_batches_achievable = int(np.ceil(
                    run_jobs*N_particles / (available_cpu_cores-run_jobs)
                    ))
                cores_for_min_batches = int(np.ceil(N_particles / min_fit_batches_achievable))
                max_allowed_by_hardware = (available_cpu_cores - run_jobs) // run_jobs
                min_fit_core_count = min(N_particles, cores_for_min_batches, max_allowed_by_hardware)
                if (min_fit_core_count < 2):
                    print("WARNING: fitness_parallel_mode changed to "
                          "'series' due to hardware limitations.")
                    fit_mode = "series"
                else:
                    max_fitness_jobs = min(N_particles, min_fit_core_count)
                    fit_jobs = max_fitness_jobs
                
            # User provided only fitness_n_jobs
            elif run_jobs is None:
                optimal_run_jobs = (available_cpu_cores - fit_jobs) // fit_jobs
                min_job_count = min(N_runs, optimal_run_jobs)
                if (min_job_count < 2):
                    print("WARNING: run_parallel_mode changed to "
                          "'series' due to hardware limitations.")
                    run_mode = "series"
                else:
                    run_jobs = min_job_count
                    
            # Both values are provided by user
            else: 
                # First check run_n_jobs
                if run_jobs > max_run_jobs:
                    print(f"WARNING: run_n_jobs changed to {max_run_jobs}.")
                    run_jobs = max_run_jobs
                
                # Next check fitness_n_jobs
                min_fit_batches_achievable = int(np.ceil(
                    run_jobs*N_particles / (available_cpu_cores-run_jobs)
                    ))
                cores_for_min_batches = int(np.ceil(N_particles / min_fit_batches_achievable))
                max_allowed_by_hardware = (available_cpu_cores - run_jobs) // run_jobs
                min_fit_core_count = min(cores_for_min_batches, max_allowed_by_hardware)
                max_fitness_jobs = min(N_particles, min_fit_core_count)
                if (min_fit_core_count < 2):
                    print("WARNING: fitness_parallel_mode changed to "
                          "'series' due to hardware limitations.")
                    fit_mode = "series"
                elif (fit_jobs > max_fitness_jobs):
                    print("WARNING: fitness_n_jobs changed to "
                          f"{max_fitness_jobs}")
                    fit_jobs = max_fitness_jobs
                    
                
    # run_mode == "series" or "mpi":
    else: 
        
        if (run_mode == "mpi") and run_jobs is not None:
            print("WARNING: run_n_jobs ignored when using MPI.")
        
        if fit_mode == "cpu":
            max_fitness_jobs = min(N_particles, available_cpu_cores)
            min_batches_achievable = int(np.ceil(N_particles / available_cpu_cores))
            min_core_count = int(np.ceil(N_particles / min_batches_achievable))
        
            # Choose values automatically
            if fit_jobs is None:
                fit_jobs = min_core_count
            # Value already chosen by user
            elif (fit_jobs > max_fitness_jobs):
                print(f"WARNING: fitness_n_jobs changed to {max_fitness_jobs}.")
                fit_jobs = max_fitness_jobs
            # Otherwise value chosen by user is fine
        
        elif fit_mode in ("cuda-cupy", "cuda-torch", "gpu"):
            if fit_jobs is not None:
                print("WARNING: fitness_n_jobs ignored for"
                      " GPU fitness evaluation.")
                

    return run_mode, run_jobs, fit_mode, fit_jobs




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


@dataclass(frozen=True, slots=True)
class PSOConfig:
    """
    @brief Immutable configuration object for PSO execution.

    All optional PSO parameters are centralized in this object.

    Validation is automatically performed during initialization.

    @note
    `frozen=True` makes the configuration immutable after creation.

    @note
    `slots=True` reduces memory overhead and slightly improves
    attribute access speed.
    """

    # ============================================================
    # SWARM PARAMETERS
    # ============================================================

    # The required inputs for run_PSO
    
    N_runs: int
    """
    Number of PSO runs to perform.
    """
    
    N_iterations: int
    """
    Number of iterations to perform each run.
    """
    
    boundary_conditions: np.ndarray
    """
    Array of lower and higher bounds for each search space dimension.
    """
    
    # Hidded parameter (to avoid conflict with boundary_conditions)
    
    N_x: int = field(init=False)
    """
    Number of dimensions of the search space.
    """
    
    # The optional inputs for run_PSO

    N_particles: int = 40
    """
    Number of particles in the swarm.
    """

    N_local: int = 3
    """
    Number of particles per neighborhood.
    Used only when PSO_type = 'lbest'.
    """
    
    PSO_type: str = 'lbest'
    """
    Controls the swarm topology.
    Options are 'lbest' for local best with ring topology and 
    'gbest' for global best.
    """
    
    boundary_type: str = 'let_them_fly'
    """
    Controls how particles that leave the search space are handled.
    Supported Options:
        - 'let_them_fly'
        - 'absorbing_wall'
        - 'reflecting_wall'
        - 'periodic'
        - 'custom'
        - 'ltf_23triangle'
        - 'ltf_23triangle_refl'
        - 'IMRPD-1'
    See SUPPORTED_BOUNDARY_TYPES in utils.dispatch_tables.py
    """
    
    space_topology: str = 'box'
    """
    Denotes the topology of the search space so that the particles are 
    distributed uniformly.
    Supported Options:
        - 'box'
        - 'spherical_surface'
    """

    # ============================================================
    # DYNAMICAL EQUATION PARAMETERS
    # ============================================================

    C1: float = 2.0
    """
    Cognitive acceleration coefficient.
    """

    C2: float = 2.0
    """
    Social acceleration coefficient.
    """

    eq_type: str = "inertia"
    """
    PSO dynamical equation type.
    Supported Options:
        - 'inertia'
        - 'constriction'
    """

    # ============================================================
    # VELOCITY CONTROL
    # ============================================================

    vmax: float = 0.5
    """
    Maximum allowable particle velocity magnitude.
    """

    clamp_initial_velocity: bool = False
    """
    Enable velocity clamping during particle initialization.
    """
    
    clamp_velocity: bool = True
    """
    Enable velocity clamping during particle updates (iterations).
    """

    # ============================================================
    # RANDOMNESS
    # ============================================================

    position_init_value_type: str = "random-uniform"
    """
    Type of values to use when initializing the particles in the search space.
    Supported Options:
        - "halton"
        - "random-uniform"
        - "premade"
        - "sobol"
    See SUPPORTED_VALUE_GENERATORS in utils.dispatch_tables.py
    """
    
    velocity_init_value_type: str = "random-uniform"
    """
    Type of values to use when initializing the particles in the search space.
    Supported Options:
        - "halton"
        - "random-uniform"
        - "premade"
        - "sobol"
    See SUPPORTED_VALUE_GENERATORS in utils.dispatch_tables.py
    """
    
    update_value_type: str = "random-uniform"
    """
    Type of values to use during particle update iterations.
    Supported Options:
        - "halton"
        - "random-uniform"
        - "premade"
        - "sobol"
    See SUPPORTED_VALUE_GENERATORS in utils.dispatch_tables.py
    """

    position_init_premade_vals: Optional[np.ndarray] = None
    """
    Sequence of pre-computed values to be used for position initialization.
    Shape:
        (n_runs, n_values)
    where each row corresponds to one PSO run.
    The values will be converted to numpy float64 array after validation.
    """

    velocity_init_premade_vals: Optional[np.ndarray] = None
    """
    Sequence of pre-computed values to be used for velocity initialization.
    Shape:
        (n_runs, n_values)
    where each row corresponds to one PSO run.
    The values will be converted to numpy float64 array after validation.
    """

    update_premade_vals: Optional[np.ndarray] = None
    """
    Sequence of pre-computed values to update particles each iteration.
    Shape:
        (n_runs, n_values)
    where each row corresponds to one PSO run.
    The values will be converted to numpy float64 array after validation.
    """
    
    entropy_vals: np.ndarray[int] = field(
        default_factory=lambda: np.array([10, 11111, 12345], dtype=np.int64)
    )
    """
    Array of entropy values used to seed the value generators.
    Note: An entropy value will not be used if the corresponding seed_vals is
    set to False.
    """

    seed_vals: Union[bool,
                     Sequence[
                        Union[
                            bool,
                            Sequence[int],
                            Sequence[np.random.SeedSequence],
                            np.ndarray,
                            ]
                        ],
                    ] = False
    """
    Matrix seed values to use for each RNG.
    Supported Options:
        - False: Uses random entropies to seed all generators.
        - True: Uses the default reproducible entropies to seed all generators.
        - [True/False, True/False, True/False]: Individually set each generator
            to use the default entropy or a random entropy.
        - [Sequence[int], Sequence[int], Sequence[int]]: Individually send 
            seeds to each generator, where each seed is an int.
        - [Sequence[np.random.SeedSequence], Sequence[np.random.SeedSequence], 
           Sequence[np.random.SeedSequence]]: idividually send seeds to each 
            generator, where each seed is a SeedSequenceObject.
        - [True/False, Sequence[int], Sequence[np.random.SeedSequence]]: Mix
        of types is supported for individual generator settings.
    After validation, `seed_vals` is converted to [[], [], []] where each inner
    [] is a list or array of provided seeds or generated seeds.
    """

    # ============================================================
    # PARALLELIZATION
    # ============================================================

    run_parallel_mode: str = "series"
    """
    Enable parallelization over runs.
    Supported Options:
        - 'series'
        - 'cpu'
        - 'mpi'
    """
    
    fitness_parallel_mode: str = "series"
    """
    Enable parallelization over fitness function calls.
    Supported Options:
        - 'series'
        - 'cpu'
        - 'cuda-cupy'
        - 'cuda-torch'
        - 'gpu'
    """
    
    run_n_jobs: Optional[int] = None
    """
    Number of PSO runs to perform in parallel.
    None = automatically choose a sensible value.
    """
    
    fitness_n_jobs: Optional[int] = None
    """
    Number of fitness calls to perform in parallel.
    None = automatically choose a sensible value.
    """

    # ============================================================
    # LOGGING / OUTPUT
    # ============================================================

    verbose: bool = False
    """
    Enable console logging.
    """

    save_history: bool = False
    """
    Store optimization history during execution.
    """

    # ============================================================
    # VALIDATION
    # ============================================================

    def __post_init__(self):
        """
        @brief Validate configuration parameters.

        @throws TypeError
        Raised if any parameter type is invalid.

        @throws ValueError
        Raised if any parameter value is invalid.
        """

        # --------------------------------------------------------
        # SWARM PARAMETER VALIDATION
        # --------------------------------------------------------

        # Required inputs
        
        if type(self.N_runs) is not int:
            raise TypeError("N_runs must be an int. "
                            f"Received type {type(self.N_runs)}")
        if (self.N_runs <= 0):
            raise ValueError(f"N_runs must be > 0. Received {self.N_runs}")
            
        if type(self.N_iterations) is not int:
            raise TypeError("N_iterations must be an int. "
                            f"Received type {type(self.N_iterations)}")
        if (self.N_iterations <= 0):
            raise ValueError("N_iterations must be > 0. "
                             f"Received {self.N_iterations}")
        
        validated_bcs = validate_boundary_conditions(self.boundary_conditions)
        object.__setattr__(self, "boundary_conditions", validated_bcs)

        # Hidden input
        
        object.__setattr__(self, "N_x", len(self.boundary_conditions))

        # Optional inputs

        if type(self.N_particles) is not int:
            raise TypeError("N_particles must be an int. "
                            f"Received type {type(self.N_particles)}")
        if self.N_particles <= 0:
            raise ValueError("N_particles must be > 0. "
                             f"Received {self.N_particles}")

        if type(self.N_local) is not int:
            raise TypeError("N_local must be an int. "
                            f"Received type {type(self.N_local)}")
        if (self.N_local < 1) or (self.N_local >= self.N_particles):
            raise ValueError("N_local must be within [1, N_particles-1]. "
                             f"Received {self.N_local}")
            
        if not isinstance(self.PSO_type, str):
            raise TypeError("PSO_type must be a str. "
                            f"Received type {type(self.PSO_type)}")
        if self.PSO_type not in (
                'lbest',
                'gbest'
        ):
            raise ValueError(f"PSO_type {self.PSO_type} not supported. "
                             "Please see README for supported options.")
            
        if not isinstance(self.boundary_type, str):
            raise TypeError("boundary_type must be a str. "
                            f"Received type {type(self.boundary_type)}")
        if self.boundary_type not in SUPPORTED_BOUNDARY_TYPES:
            raise ValueError(f"boundary_type {self.boundary_type} not "
                             "supported. Please check README for supported "
                             "options.")
        if (self.N_x < 4) and self.boundary_type in ("ltf_23triangle", 
                                                     "ltf_23triangle_refl"):
            raise ValueError(f"boundary_type {self.boundary_type} requires at "
                             "least 4 search space dimensions. "
                             f"Received {self.N_x}")
        if (self.N_x < 6) and self.boundary_type in ("ltf_2345triangle", 
                                                     "ltf_2345triangle_refl",
                                                     "IMRPD-1"):
            raise ValueError(f"boundary_type {self.boundary_type} requires at "
                             "least 6 search space dimensions. "
                             f"Received {self.N_x}")
            
        if not isinstance(self.space_topology, str):
            raise TypeError("space_topology must be a str. "
                            f"Received type {type(self.space_topology)}")
        if self.space_topology not in (
                'box',
                'spherical_surface'
                ):
            raise ValueError("space_topology {self.space_topology} not "
                        "supported. Please see README for supported options.")
            
# =============================================================================
#         # After more thought, I'm not sure this check is needed. In principle,
#         # searching over a spherical surface should have at least 2 dimensions,
#         # but the arccos fix can work with only 1 and there may be cases where
#         # that behavior is desired.
#         if (self.space_topology == "spherical_surface") and (N_x < 2):
#             raise ValueError("space_topology == 'spherical_surface' requires "
#                              "at least 2 search space dimensions (θ,φ)")
# =============================================================================
            
        # --------------------------------------------------------
        # DYNAMICAL EQUATION PARAMETER VALIDATION
        # --------------------------------------------------------
            
        if not isinstance(self.C1, numbers.Real):
            raise TypeError("C1 must be a real number")
        if (self.C1 <= 0.0):
            raise ValueError("C1 must be positive")

        if not isinstance(self.C2, numbers.Real):
            raise TypeError("C2 must be a real number")
        if (self.C2 <= 0.0):
            raise ValueError("C2 must be positive")

        if not isinstance(self.eq_type, str):
            raise TypeError("eq_type must be a str. "
                            f"Received type {type(self.eq_type)}")
        if self.eq_type not in (
            "inertia",
            "constriction"
        ):
            raise ValueError(f"eq_type {self.eq_type} not supported. "
                             "Please see README for supported options.")

        # --------------------------------------------------------
        # VELOCITY CONTROL VALIDATION
        # --------------------------------------------------------

        if not isinstance(self.vmax, numbers.Real):
            raise TypeError("vmax must be a real number")
        if (self.vmax <= 0.0):
            raise ValueError("vmax must be positive")

        if not isinstance(self.clamp_initial_velocity, bool):
            raise TypeError("clamp_initial_velocity must be a bool. "
                        f"Received type {type(self.clamp_initial_velocity)}")
        
        if not isinstance(self.clamp_velocity, bool):
            raise TypeError("clamp_velocity must be a bool. "
                            f"Received type {type(self.clamp_velocity)}")
        
        # --------------------------------------------------------
        # RANDOMNESS VALIDATION
        # --------------------------------------------------------

        if not isinstance(self.position_init_value_type, str):
            raise TypeError("position_init_value_type must be a string.")
        if self.position_init_value_type not in SUPPORTED_VALUE_GENERATORS:
            raise ValueError(f"position_init_value_type {self.position_init_value_type} not "
                        "supported. Please see README for supported options.")

        if not isinstance(self.velocity_init_value_type, str):
            raise TypeError("velocity_init_value_type must be a string.")
        if self.velocity_init_value_type not in SUPPORTED_VALUE_GENERATORS:
            raise ValueError(f"velocity_init_value_type {self.velocity_init_value_type} not "
                        "supported. Please see README for supported options.")

        if not isinstance(self.update_value_type, str):
            raise TypeError("update_value_type must be a string.")
        if self.update_value_type not in SUPPORTED_VALUE_GENERATORS:
            raise ValueError(f"update_value_type {self.update_value_type} not "
                        "supported. Please see README for supported options.")

        validated_pos_init = validate_random_sequence(self.position_init_premade_vals, 
                                                self.position_init_value_type,
                                                self.velocity_init_value_type,
                                                self.update_value_type,
                                                self.N_runs,
                                                self.N_iterations,
                                                self.N_x,
                                                self.N_particles,
                                                0)
        object.__setattr__(self, "position_init_premade_vals", validated_pos_init)
            
        validated_vel_init = validate_random_sequence(self.velocity_init_premade_vals, 
                                                self.position_init_value_type,
                                                self.velocity_init_value_type,
                                                self.update_value_type,
                                                self.N_runs,
                                                self.N_iterations,
                                                self.N_x,
                                                self.N_particles,
                                                1)
        object.__setattr__(self, "velocity_init_premade_vals", validated_vel_init)
            
        validated_update = validate_random_sequence(self.update_premade_vals, 
                                                self.position_init_value_type,
                                                self.velocity_init_value_type,
                                                self.update_value_type,
                                                self.N_runs,
                                                self.N_iterations,
                                                self.N_x,
                                                self.N_particles,
                                                2)
        object.__setattr__(self, "update_premade_vals", validated_update)
        
        validated_entropy_vals = validate_entropy_vals(self.entropy_vals)
        object.__setattr__(self, "entropy_vals", validated_entropy_vals)
            
        validated_seed_vals = validate_seeds(self.seed_vals, 
                                             self.N_runs,
                                             self.entropy_vals)
        object.__setattr__(self, "seed_vals", validated_seed_vals)

        # --------------------------------------------------------
        # PARALLELIZATION VALIDATION
        # --------------------------------------------------------

        if not isinstance(self.run_parallel_mode, str):
            raise TypeError("run_parallel_mode must be a str. "
                            f"Received type {type(self.run_parallel_mode)}")
        if self.run_parallel_mode not in (
                "series",
                "cpu",
                "mpi"
                ):
            raise ValueError(f"run_parallel_mode {self.run_parallel_mode} not "
                        "supported. Please see README for supported options.")

        if not isinstance(self.fitness_parallel_mode, str):
            raise TypeError("fitness_parallel_mode must be a str."
                            f"Received type {type(self.fitness_parallel_mode)}")
        if self.fitness_parallel_mode not in (
                "series",
                "cpu",
                "cuda-cupy",
                # "cuda-torch",
                # "gpu"
                ):
            raise ValueError("fitness_parallel_mode "
                             f"{self.fitness_parallel_mode} not supported. "
                             "Please see README for supported options.")
            
        if self.run_n_jobs is not None:
            if type(self.run_n_jobs) is not int:
                raise TypeError("run_n_jobs must be an int. "
                                f"Received type {type(self.run_n_jobs)}")
            if self.run_n_jobs <= 0:
                raise ValueError("run_n_jobs must be > 0. "
                                 f"Received {self.run_n_jobs}")

        if self.fitness_n_jobs is not None:
            if type(self.fitness_n_jobs) is not int:
                raise TypeError("fitness_n_jobs must be an int. "
                                f"Received type {type(self.fitness_n_jobs)}")
            if self.fitness_n_jobs <= 0:
                raise ValueError("fitness_n_jobs must be > 0. "
                                 f"Received {self.fitness_n_jobs}")
            
        (run_mode, 
         run_jobs, 
         fit_mode, 
         fit_jobs) = validate_parallelization(self.run_parallel_mode,
                                              self.run_n_jobs,
                                              self.fitness_parallel_mode,
                                              self.fitness_n_jobs,
                                              self.N_runs,
                                              self.N_particles)
        object.__setattr__(self, "run_parallel_mode", run_mode)
        object.__setattr__(self, "run_n_jobs", run_jobs)
        object.__setattr__(self, "fitness_parallel_mode", fit_mode)
        object.__setattr__(self, "fitness_n_jobs", fit_jobs)
            
        # --------------------------------------------------------
        # LOGGING / OUTPUT VALIDATION
        # --------------------------------------------------------

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a bool. "
                            f"Received type {type(self.verbose)}")
            
        if not isinstance(self.save_history, bool):
            raise TypeError("save_history must be a bool. "
                            f"Received type {type(self.save_history)}")


def print_config(obj):
    for f in fields(obj):
        value = getattr(obj, f.name)
        print(f"{f.name}: {value}")

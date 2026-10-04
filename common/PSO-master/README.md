# Particle Swarm Optimization

## Overview

A Python implementation of Particle Swarm Optimization (PSO) for stochastic parameter-space optimization, with support for CPU, MPI, and CUDA/GPU-accelerated fitness evaluation.

**Version:** `0.6.3`

## Features

- Global-best (`gbest`) and local-best (`lbest`) swarm topologies
- Inertia-weight and constriction-factor dynamics
- Series, multiprocessing, MPI, and CUDA/CuPy execution modes
- Reproducible random-number and quasi-random-number generation and cross-validation support
- Custom boundary handling
- Optional particle-history tracking
- Unit and integration tests with `pytest`

---

## Installation and Dependencies

This repository can be cloned from GitHub via
```
git clone <PSO_GitHub_link>
```

This project requires:

- **Python 3.12**  
- Python standard modules: `collections`, `concurrent`, `dataclasses`, `datetime`, `inspect`, `math`, `numbers`, `os`, `pathlib`, `pickle`, `sys`, `typing`, `unittest`, `__future__`
- Python packages: `matplotlib`, `numba`, `numpy`, `pytest`, `scipy`

- **MPI** (only required if performing runs over MPI workers)

- MPI-specific Python packages: `mpi4py` (only required if using MPI)

- **CUDA version 11.x or 12.x** (only required if using GPU acceleration)
- CUDA-specific Python packages: `cupy`  (only required if using GPU acceleration)


Compatible versions of the non-GPU Python packages are given in `requirements.txt` and can be installed via
```bash
cd PSO

pip install -r requirements.txt
```

The GPU package versions depend on your CUDA version and must be installed separately. 

For CUDA 12 systems:
```bash
pip install cupy-cuda12x==13.3.0
```

For CUDA 11 systems:
```bash
pip install cupy-cuda11x==13.3.0
```

After installing the dependencies, run pytest (below) to confirm the PSO install is working.

---

## Testing

This repository has unit tests which can be run through the terminal with

```bash
cd PSO

python -m pytest
```

If run on a device that does not have a GPU and/or the required GPU packages, the GPU tests will be skipped.

---

## Example Usage

To use Particle Swarm Optimization, import and call the `run_PSO` function from `PSO/PSO_main.py`.

You must pass the 4 required positional arguments detailed below. Optional parameters as described after the example code.

<div align="center">

| Required Arguments | Type | Description   |
|-------------|------|---------------|
| `<N_runs>` | int | The number of PSO runs to compute. |
| `<N_iterations>` | int | The number of iterations to compute each run. |
| `<boundary_conditions>` | ndarray | NumPy array containing the boundary conditions for each dimension of the parameter space. This must be an array of pairs, where the ith pair is [lower_bound, higher_bound] for the ith dimension. |
| `<fitness_function>` | callable | The fitness function to search over. If running on CPU, this function must be of the form: `fitness_function(x_vals_one_particle: ndarray) -> float:`. It must take one argument which is an array of location values for one particle and must return one float value which is the fitness function corresponding to that location array. If running on GPU, this function must be of the form: `fitness_function(x_vals_all_particles: ndarray) -> ndarray:` where it takes a 2D CuPy array containing the `x_vals` arrays of all particles and returns an array containing the fitness values for all particle locations. **Important:** If you want to run fitness computations in parallel, this function must be defined at module scope and be pickleable. If you want to perform fitness computations over GPUs, this function must be vectorized and support CuPy arrays. |

</div>

The `run_PSO` function will return 5 items (The last 3 will be empty for performance unless `save_history=True`).

<div align="center">

| Return | Type | Description   |
|-------------|-----|----------------|
| `fitness_vals` | ndarray | Array containing the best fitness values obtained from each PSO run. |
| `best_locations` | ndarray | Array containing the locations in search space corresponding to `fitness_vals`.  |
| `particle_history` | ndarray | Array containing the locations visited by every particle during the search (empty by default). |
| `fitness_history` | ndarray | Array containing the fitness values at each location visited by each particle (empty by default). |
| `pbest_history` | ndarray | Array containing the pbest locations of every particle throughout the search (empty by default). |

</div>
 
An example is given below. Additionally, see `examples/PSO_Launcher_pc.py` for a complete working example.

```python
# Import the PSO function and a provided fitness function.
from PSO_main import run_PSO
from benchmarks.benchmark_functions import Rastrigin
import numpy as np

# Define the 4 required PSO parameters.
N_runs = 6
N_iterations = 5000
boundary_conditions = np.array([
                                [-5.12, 5.12], # bounds on x1
                                [-5.12, 5.12], # bounds on x2
                                [-5.12, 5.12]  # bounds on x3
                                ])
fitness_function = Rastrigin

# Receive the best fitness values and corresponding locations of each run.
( fitness_vals, 
  best_locations,
          _, _, _) = run_PSO( # Required arguments
                              N_runs,
                              N_iterations,
                              boundary_conditions,
                              fitness_function,
                              # Optional keyword parameters
                              N_particles=40,
                              PSO_type="lbest")

```

---

## Expected Output

All examples in `examples/` are run using the Rastrigin benchmark function by default. The minimum of this function is 0 and occurs when all parameters are 0 in physical coordinates.

The expected output of all example scripts is

```text
Best fitness vals: 
[0. 0. 0. 0. 0. 0.]

Best Locations: 
[[ 3.69442166e-09  7.75095543e-11  3.53333096e-09]
 [-2.20380514e-09 -5.91088956e-10  8.68855210e-10]
 [ 2.45076848e-10 -1.64084302e-09  1.86865368e-09]
 [-3.20768478e-09 -9.21282606e-10 -3.44849216e-09]
 [-1.22900801e-09 -1.74421100e-09 -3.26948513e-09]
 [ 1.76147097e-09  1.53822022e-09  1.43684620e-09]]
```

---

## Optional Parameters

The `run_PSO` function also takes many optional keyword arguments. They are listed here with their default values:

- `N_particles=40` (int): The number of particles in the swarm.

- `N_local=3` (int): The number of particles per neighborhood. Used only when `PSO_type='lbest'`.

- `PSO_type='lbest'` (str): Controls the swarm topology. Supported Options:

<div align="center">

| PSO_type | Implementation   |
|-------------|---------------------|
| 'gbest'        | Global Best: The social term attracts towards the best location discovered by the entire swarm. |
| 'lbest'   | Local Best: The social term attracts towards the best location discovered by the particle's neighborhood. Neighborhoods are structured with a ring topology split evenly for odd N_local and split asymmetrically for even N_local with one more neighbor on the right than on the left. |

</div>

- `boundary_type='let_them_fly'` (str): Determines how particles which leave the allowed search space are handled. Internally, PSO operates in standardized coordinates in [0,1]^N. The supplied `boundary_conditions` define the corresponding physical parameter ranges; conversions between standardized and physical coordinates are handled internally. Supported Options:

<div align="center">

**Standard PSO Options**

| boundary_type | Implementation   |
|-------------|---------------------|
| 'let_them_fly'        | Particles are allowed to leave the search space, but have their fitness value set to infinity at that point. |
| 'absorbing_wall' | Particles that leave the search space have each offending coordinate clamped to the nearest boundary of the [0,1] hypercube and their corresponding velocity components set to zero. |
| 'reflecting_wall' | Particles that leave the search space are reflected back into the [0,1] hypercube elastically and have their corresponding velocity components flipped. |
| 'periodic' | Particles that leave the search space are wrapped periodically into the [0,1] hypercube. Velocity components are unchanged. |
| 'custom' | No boundary conditions are implemented. This should be used in cases where the provided fitness function already includes boundary handling. Particles will still be initialized uniformly into the [0,1] hypercube. |

**Specialized Gravitational Wave Options:**

| boundary_type | Implementation   |
|-------------|---------------------|
| 'ltf_23triangle' | Same as 'let_them_fly', but with the added condition that the 2D subspace of the parameter space for parameters at indices 2 and 3 is only allowed to be a triangular half of the normal [0,1] square, separated by the x=y line. |
| 'ltf_23triangle_refl' | Same as 'ltf_23triangle', but the triangular boundary is reflective. Particles that enter the disallowed triangle are reflected back into the allowed triangle instead of having their fitness set to infinity. | 
| 'IMRPD-1' | Same as 'ltf_23triangle_refl', but the 2D subspace of the parameters at indices 4 and 5 is restricted to a 90 degree rotated square inside the standardized [0,1]^2 search space with vertices at (0, 0.5), (0.5, 1), (1, 0.5), and (0.5, 0). Any particles that exit this region have their fitness set to infinity. |

</div>

- `C1=2.0` (float): Cognitive acceleration coefficient.

- `C2=2.0` (float): Social acceleration coefficient.

- `eq_type='inertia'` (str): PSO dynamical equation type to use. Supported Options:

<div align="center">

| eq_type | Dynamical Equation |
|--------------|-------------------|
| 'inertia' | $v_{new} = wv_{prev} + C_1R_1(x_{pbest}-x_{prev}) + C_2R_2(x_{lbest}-x_{prev})$ <br><br> where: $w = 0.9 - 0.5*(iteration-1)/(N_{iterations}-1)$ |
| 'constriction' | $v_{\mathrm{new}} = K\left(v_{\mathrm{prev}} + C_1R_1(x_{\mathrm{pbest}} - x_{\mathrm{prev}}) + C_2R_2(x_{\mathrm{lbest}} - x_{\mathrm{prev}})\right)$ <br><br> where: $K = \frac{2}{\lvert 2 - \phi - \sqrt{\phi^2 - 4\phi} \rvert}$ <br><br> and: $\phi = C_1 + C_2$ |

<div align="left">

- `vmax=0.5` (float): The max absolute value allowed for the velocities. Used to increase the extrapolation phase. This value is in standardized coordinates and must be inside the interval [0,1].

- `clamp_initial_velocity=False` (bool): True to restrict velocities to vmax during particle initialization. False to allow initial velocities without bound.

- `clamp_velocity=True` (bool): True to restrict velocities to vmax during particle updates (iterations). False to allow velocities without bound.

- `position_init_premade_vals=None` (None or ndarray): A 2D array of values to use when initializing the particle positions instead of generating values. This allows for cross-validation between programming languages that use different random number generators. The array must be 2D, where the ith row of values is used for the ith run. Note: must set `position_init_value_type='premade'`.

- `velocity_init_premade_vals=None` (None or ndarray): A 2D array of values to use when initializing the particle velocities instead of generating values. This allows for cross-validation between programming languages that use different random number generators. The array must be 2D, where the/ ith row of values is used for the ith run. Note: must set `velocity_init_value_type='premade'`.

- `update_premade_vals=None` (None or ndarray): A 2D array of values to use when updating the particle positions each iteration instead of generating values. This allows for cross-validation between programming languages that use different random number generators. The array must be 2D, where the ith row of values is used for the ith run. Note: must set `update_value_type='premade'`.

- `position_init_value_type='random-uniform'` (str): Determines the value generator used to initialize the particle positions. Supported Options in table below:

- `velocity_init_value_type='random-uniform'` (str): Determines the value generator used to initialize the particle velocities. Supported Options in table below:

- `update_value_type='random-uniform'` (str): Determines the value generator used when updating the particle positions each iteration. Supported Options in table below:

<div align="center">

| position_init_value_type <br> velocity_init_value_type <br> update_value_type | Implementation   |
|-------------|---------------------|
| 'halton' | Quasi-random values are drawn from the Halton sequence. |
| 'random-uniform' | Random values are drawn from a uniform distribution. |
| 'premade' | Pre-computed values supplied by the user are taken from `position_init_premade_vals`, `velocity_init_premade_vals`, or `update_premade_vals`, respectively. |
| 'sobol' | Quasi-random values are drawn from the Sobol sequence. Note: The balance properties of Sobol's points require the number of search space dimensions to be a power of 2. |

</div>

- `entropy_vals=[10, 11111, 12345]` (array): Array of entropy values used to seed the value generators. The first entropy is used for position initialization, the second for velocity initialization, and the third for particle updates. Note: A reproducible entropy value will not be used if the corresponding `seed_vals` is set to False.

- `seed_vals=False` (bool or Sequence[_, _, _]): This gives options for how to generate seeds used in the value generators determined by `position_init_value_type`, `velocity_init_value_type`, and `update_value_type`, or allows the user to input custom values. Supported Options:

<div align="center">

| seed_vals | Implementation   |
|-------------|---------------------|
| False | Uses random entropies to seed all generators |
| True | Uses reproducible entropies to seed all generators. The seeds are spawned from SeedSequence Objects with entropies taken from `entropy_vals`. This method guarantees that each sequence of random values is independent. |
| [True/False, True/False, True/False] | Individually set each generator to use a reproducible entropy from `entropy_vals` or a random entropy. |
| [Sequence[int], Sequence[int], Sequence[int]] | Individually send seeds to each generator, where each seed is an int. Each Sequence[int] must contain `N_runs` integers where the ith integer is the seed of the ith run. |
| [Sequence[np.random.SeedSequence], Sequence[np.random.SeedSequence], Sequence[np.random.SeedSequence]] | Idividually send seeds to each generator, where each seed is a SeedSequence Object. Each Sequence[np.random.SeedSequence] must contain `N_runs` SeedSequence Objects where the ith SeedSequence Object is the seed of the ith run. |
| [True/False, Sequence[int], Sequence[np.random.SeedSequence]] | Mixed seed types is supported for individual generator settings. |

</div>


- `run_parallel_mode='series'` (str): Flag which determines whether to perform each PSO run in series or in parallel over CPU cores. Supported Options:

<div align="center">

| run_parallel_mode | Implementation   |
|-------------|---------------------|
| 'series'        | Runs are performed in series. |
| 'cpu'   | Runs are performed in parallel over CPU cores. |
| 'mpi'   | Runs are performed in parallel over MPI workers with memory sharing across nodes. |

</div>

- `run_n_jobs=None` (None or int): Number of PSO runs to perform in parallel. Only used when `run_parallel_mode='cpu'`. If left with the default `None` option, the algorithm will choose the optimal value automatically based on your hardware and `N_runs`. If `run_parallel_mode='mpi'`, the parallelization of runs is determined by how many tasks are requested in the slurm script.

- `fitness_parallel_mode='series'` (str): Flag which determines whether to perform fitness function calls in series or in parallel. Supported Options:

<div align="center">

| fitness_parallel_mode | Implementation   |
|-------------|---------------------|
| 'series'        | Fitness calls are performed in series over CPU cores. |
| 'cpu'   | Fitness calls are performed in parallel over CPU cores. |
| 'cuda-cupy' | Fitness calls are performed over CUDA-capable GPUs using the CuPy backend. |

</div>

- `fitness_n_jobs=None` (None or int): Number of fitness function calls to perform in parallel. Only used when `fitness_parallel_mode` is not `'series'`. If left with the default `None` option, the algorithm will choose the optimal value automatically based on your hardware and desired parameters.

- `verbose=False` (bool): Flag to print information about the job while running.

- `save_history=False` (bool): Flag to save the particles' history. By default, `particle_history`, `fitness_history`, and `pbest_history` are returned as empty arrays. If set to `True`, then `particle_history`, `fitness_history`, and `pbest_history` will contain the locations visited by every particle, the fitness values at those locations, and the pbest locations of every particle during the routine.

---

## Notes Regarding Parallelization

`run_PSO` can perform particle swarm optimization with various combinations of parallelization methods. These methods are chosen by the `run_parallel_mode` and `fitness_parallel_mode` parameters detailed above. By default, the program will perform PSO runs and fitness evaluations in series for maximum hardware compatibility. A summary table of configurations is given here, with detailed recomendations given below:

<div align="center">

| run_parallel_mode | fitness_parallel_mode | Recommended Use   |
|-------------|-----|-----------------|
| 'series' | 'series' | Maximum compatibility / debugging |
| 'cpu' | 'series' | Personal computer; Multiple independent PSO runs |
| 'series' | 'cpu' | Personal computer; Single PSO run |
| 'series' | 'cuda-cupy' | Personal computer with GPU|
| 'cpu' | 'cpu' | Single-node CPU cluster |
| 'cpu' | 'cuda-cupy' | Single-node GPU cluster |
| 'mpi' | 'cpu' | Multi-node CPU cluster |
| 'mpi' | 'cuda-cupy' | Multi-node GPU cluster |

</div>

**Parallelization on a Personal Computer**

For most personal computers, the recommended settings are `run_parallel_mode='cpu'` and `fitness_parallel_mode='series'`. Example usage can be found in `examples/PSO_Launcher_pc.py`.

Batching is supported. For example: If 8 runs are requested on a machine with 4 CPU cores and `run_parallel_mode='cpu'`, it will perform 2 batches of 4 parallel PSO runs.

Batching of both PSO runs and fitness evaluations is also supported. For example: If 8 runs are requested on a machine with 16 CPU cores and `run_parallel_mode='cpu'`, `fitness_parallel_mode='cpu'`, `run_n_jobs=4`, and `fitness_n_jobs=3` it will perform 2 batches of 4 parallel PSO runs where each PSO run performs fitness evaluations in batches of 3. Although this is supported, overhead dominates on small machines and it is not recommended. The algorithm is smart enough to avoid this and will instead perform the 8 runs in parallel and revert back to `fitness_parallel_mode='series'` if `run_n_jobs` and `fitness_n_jobs` are not specified. 

Most personal computers do not have enough CPU cores to perform both PSO runs and fitness evaluations in parallel over CPU cores without oversubscription or excessive batching. Oversubscribing the CPU cores results in significantly worse performance and if attempted, the program will revert back to series to avoid it.

For personal computers that have a CUDA-capable GPU, it may be optimal to use  `run_parallel_mode='series'` and `fitness_parallel_mode='cuda-cupy'`.

Acceleration of the fitness evaluations over GPU is supported, but requires the fitness function to be CUDA-compatible and vectorized. For simple fitness functions, the overhead of loading data onto and off of the GPU may dominate and `run_parallel_mode='cpu'` and `fitness_parallel_mode='series'` could result in better performance. Note that `run_parallel_mode='cpu'` and `fitness_parallel_mode='cuda-cupy'` is not possible unless the machine has more than 1 GPU and will revert back to `run_parallel_mode='series'`.

**Parallelization on a Computing Cluster**

For fitness functions that are NOT GPU-compatible, the recommended settings are `run_parallel_mode='mpi'` and `fitness_parallel_mode='cpu'`. For fitness functions that are GPU-compatible, the recommended settings are `run_parallel_mode='mpi'` and `fitness_parallel_mode='cuda-cupy'`. Example usage can be found in `examples/PSO_Launcher_cc.py`.

Using `run_parallel_mode='cpu'` is optimal on a personal computer and can work if a single node is requested from the partition, but it cannot handle jobs across multiple independent nodes. Using `run_parallel_mode='mpi'` handles this situation and will create a parallel MPI worker for each PSO run using the required amount of the allocation. For cases where `N_runs` is greater than the number of MPI workers, the PSO runs are performed in round robin batches.

When running on computing clusters, resource allocation is typically handled by a scheduler such as Slurm. Example slurm scripts for the high performance computing clusters TACC lonestar6 and UTRGV CRADLE are also given in `examples/`.


---

## Repository Structure
```text
PSO/
├── .gitignore
├── __init__.py
├── Doxyfile
├── PSO_main.py
├── pytest.ini
├── README.md
├── requirements.txt
|
├── benchmarks/
|    ├── __init__.py
|    ├── benchmark_functions.py
|    ├── create_random_values.m
|    ├── cross_validation_tool.py
|    └── plot_benchmark_funcs.py
|
├── core/
|    ├── __init__.py
|    ├── bests.py
|    ├── boundary_handling.py
|    ├── coordinate_standardization.py
|    ├── dynamical_equations.py
|    ├── particle_initialization.py
|    ├── perform_pso_run.py
|    └── update_particles.py
|
├── examples/
|    ├── PSO_Launcher_cc.py
|    ├── PSO_Launcher_pc.py
|    ├── slurm_CRADLE.sh
|    ├── slurm_TACC_cpu.sh
|    └── slurm_TACC_gpu.sh
|
├── options/
|    ├── __init__.py
|    ├── boundary_types/
|    |      ├── __init__.py
|    |      ├── absorbing_wall.py
|    |      ├── custom.py
|    |      ├── imrpd_1.py
|    |      ├── let_them_fly.py
|    |      ├── ltf_23triangle.py
|    |      ├── ltf_23triangle_refl.py
|    |      ├── periodic.py
|    |      └── reflecting_wall.py
|    └── value_generators/
|           ├── __init__.py
|           ├── halton.py
|           ├── premade.py
|           ├── random_uniform.py
|           └── sobol.py
|
├── parallel_wrappers/
|    ├── __init__.py
|    ├── wrap_fitness_func.py
|    └── wrap_perform_pso_run.py
|
├── tests/
|    ├── __init__.py
|    ├── test_benchmark_functions.py
|    ├── test_bests.py
|    ├── test_boundary_handling.py
|    ├── test_config_validation.py
|    ├── test_coordinate_standardization.py
|    ├── test_dispatch_tables.py
|    ├── test_dynamical_equations.py
|    ├── test_input_validation.py
|    ├── test_particle_initialization.py
|    ├── test_perform_pso_run.py
|    ├── test_pso_config.py
|    ├── test_PSO_main.py
|    ├── test_update_particles.py
|    ├── test_wrap_fitness_func.py
|    ├── test_wrap_perform_pso_run.py
|    ├── boundary_type_tests/
|    |      ├── __init__.py
|    |      ├── test_absorbing_wall.py
|    |      ├── test_custom.py
|    |      ├── test_imrpd_1.py
|    |      ├── test_let_them_fly.py
|    |      ├── test_ltf_23triangle.py
|    |      ├── test_ltf_23triangle_refl.py
|    |      ├── test_periodic.py
|    |      └── test_reflecting_wall.py
|    ├── data/
|    └── value_generator_tests/
|           ├── __init__.py
|           ├── test_halton.py
|           ├── test_premade.py
|           ├── test_random_uniform.py
|           └── test_sobol.py
|
└── utils/
     ├── __init__.py
     ├── config.py    
     ├── dispatch_tables.py
     └── input_validation.py
```

- `PSO_main.py` contains `run_PSO` and is the entry point of the algorithm.

- `benchmarks/` contains benchmark fitness functions for testing the algorithm and tools for cross-validation with MATLAB.

- `core/` contains the algorithm elements of particle swarm optimization.

- `examples/` contains example scripts for using `run_PSO` and submitting jobs via Slurm.

- `options/` contains files for PSO variations that are controlled by optional keyword parameters.

- `parallel_wrappers/` contains wrapper functions that handle parallelization of runs and fitness evaluations.

- `tests/` contains unit and integration tests to be run with pytest for pipeline validation.

- `utils/` contains helper functions used by the algorithm functions. These utility functions handle validation of user inputs, PSO parameter storage, and transferring parameters between functions.

---

## How It Works

Each PSO run initializes a swarm of particles within the supplied
parameter-space boundaries. At each iteration, particle velocities and
positions are updated according to the selected dynamical equation and
swarm topology. Particle fitness values are evaluated, and each
particle's personal best (`pbest`) and the relevant neighborhood/global
best (`lbest`/`gbest`) are updated.

The implementation separates:

1. Particle initialization
2. Fitness evaluation
3. Best-location tracking
4. Velocity/position updates
5. Boundary handling
6. Parallel execution

---

## Documentation

Detailed API documentation is available via Doxygen.

To generate it locally from terminal:

```bash
cd PSO

doxygen Doxyfile
```

---

## Cross-Validation with MATLAB

This Python implementation of Particle Swarm Optimization was designed to allow cross-validation with the MATLAB implementation of PSO found in `CODES/MatlabCodes/crcbpso.m` from the https://github.com/mohanty-sd/SDMBIGDAT19 repository.

Since Python and MATLAB use different implementations for random number generation, a MATLAB script `benchmarks/create_random_values.m` is provided which will generate random values in the same order as `crcbpso.m` and store them in 3 `.txt` files. These values can be passed to `run_PSO` through the `premade_vals` parameters to ensure identical behavior between the Python and MATLAB codes. See `benchmarks/cross_validation_tool.py` for an example.

It is important to run `crcbpso.m` and `run_PSO` with the same parameters for the comparison and to make sure `crcbpso.m` and `create_random_values.m` are run with the same seed.

---

## References

S. D. Mohanty, *Swarm Intelligence Methods for Statistical
Regression*. CRC Press, 2019. doi: [10.1201/b22461](https://doi.org/10.1201/b22461).

---

## Author

Ben Carlson

- GitHub: `@luxluinil`
- Email: CarlsonBen@protonmail.com
- Affiliation: The University of Texas Rio Grande Valley

---

## License

This repository is currently private and is not licensed for redistribution or public use. Access is restricted to authorized collaborators only.

An open-source license will be provided upon public release.

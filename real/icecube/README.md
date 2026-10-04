# IceCube 3-year neutrino benchmark

This is the `icecube-neutrinos` real problem from Johannes Buchner's
`space-of-inference-spaces` collection.

The benchmark uses the PISA three-year high-statistics DeepCore oscillation
configuration (Sample B): two model pipelines, neutrinos and atmospheric
muons, plus the fixed observed-data pipeline. The published PISA example uses
an 8 x 8 x 2 reconstructed-energy/reconstructed-coszenith/PID binning and a
16-parameter fit.

## Setup

Run the normal project setup:

```bash
./setup.sh real/icecube
```

The environment installs PISA from the IceCube GitHub repository. PISA ships
the three benchmark configs and their associated event/resources under
`pisa_examples/resources`. `post_setup.sh` verifies all three are available.

No separate `settings/` checkout or manual PISA clone is required.

## Run

```bash
./run.sh real/icecube ultranest-safe
```

The Python problem code resolves the configs through
`pisa.utils.resources.find_resource()`, so it is independent of the current
working directory and works with PISA's installed package resources.

The public IceCube three-year high-statistics neutrino oscillation release is
91 MB and covers the 5.6--56 GeV samples used by the benchmark:
https://icecube.wisc.edu/data-releases/2019/05/three-year-high-statistics-neutrino-oscillation-samples/

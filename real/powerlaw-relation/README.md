# Power-law relation — Tully-Fisher benchmark

A 3-parameter real inference problem corresponding to Buchner's
`powerlaw-relation` benchmark.

The implementation follows the UltraNest "fitting a line" tutorial using 42
Kormendy & Ho measurements of bulge mass and velocity dispersion. Each data
point is represented by a deterministic Monte-Carlo cloud of 400 samples,
and the likelihood averages the intrinsic-scatter Gaussian over that cloud.

Parameters:

- `slope` ~ Uniform(-3, 3)
- `offset` ~ LogUniform(10, 1000) in km/s (represented internally as log10)
- `scatter` ~ LogUniform(0.001, 10)

Run:

```bash
./setup.sh real/powerlaw-relation
./run.sh real/powerlaw-relation testsampler
./run.sh real/powerlaw-relation ultranest-safe
```

The 42-row source values are stored in `data/tully_fisher.csv` so the benchmark
is self-contained. The data/model setup follows the current UltraNest line-fitting
tutorial and the `powerlaw-relation` definition in Buchner's benchmark paper.

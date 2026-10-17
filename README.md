# Sim2Science: carbon validation

Structures, analysis scripts and data accompanying the carbon validation figure.
Includes MACE-MP-0b3, MACE-MH-1, MACE-OMat and ORB-v3, with five runs each for
pure-carbon and OH-containing structures.

- Model directories: structures grouped by composition and run.
- `analysis_scripts/`: ring, pair-distribution and pore-size analyses.
- `raw_data/`: per-run numerical results.
- `plot_data/`: ensemble averages and standard errors.
- `reference_data/`: digitised Forse and GAP-22 reference curves.
- `figures/`: combined figure and structure image.
- `manifest.json`: index connecting structures and results.

Radial distribution functions are calculated using
[cuRDF](https://github.com/joehart2001/cuRDF).

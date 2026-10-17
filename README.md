# Sim2Science: carbon validation

Structures, analysis scripts and data accompanying the carbon validation figure.
Includes MACE-MP-0b3, MACE-MH-1, MACE-OMat and ORB-v3, with five runs each for
pure-carbon and OH-containing structures.

- Model directories: structures grouped by composition and run.
- `analysis_scripts/`: `rings.py`, `pdf.py` and `psd.py`, each providing a
  calculation function returning numerical arrays. Rings and PSD accept ASE Atoms;
  PDF accepts a structure filename.
- `raw_data/`: per-run numerical results.
- `plot_data/`: ensemble averages and standard errors.
- `reference_data/`: digitised Forse and GAP-22 reference curves.
- `figures/`: combined figure and structure image.

Radial distribution functions are calculated using
[cuRDF](https://github.com/joehart2001/cuRDF).

Pore-size distributions are calculated using
[PoreTorch](https://github.com/joehart2001/poretorch), with sphere-volume weighting.

# dmost
DEIMOS python 1D velocity redux 
... get dee-most out of your data.

All rawdata goes into a folder rawdata_[year]/
1. Run run_mkplanfile to create .plans for setup below.

To run a single mask:
1. Conda activate pypeit
2. Run `dmost_setup` to generate directories and run pypeit setup files.
3. `run_pypeit mask.pypeit -o`
4.  Run collate1d and create marz file:  `dmost_collate1d_marz.py`
5.  Identify galaxies using marz
6. Run dmost:  flexure, telluric, template and emcee



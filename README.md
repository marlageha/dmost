# dmost
DEIMOS python 1D velocity redux 
... get dee-most out of your data.

All rawdata goes into a folder rawdata_[year]/
1. Run `run_dmost_planfiles` to create .plans in directory.  Edit the .plan file.  These files/calibrations will be used in the reduction.

To run a single mask:
1. Conda activate pypeit
2. Run `run_dmost_mask_setup --mask [maskname]` to generate directories and run pypeit setup files.
3. `run_pypeit mask.pypeit -o`
4.  Run collate1d and create marz file:  `run_collate1d_marz.py`
5.  Identify galaxies using marz
6.  python run_dmost.py 27StrmDRun 



# dmost
DEIMOS python 1D redux
... get dee-most out of your data.

All rawdata goes into a folder rawdata_2020/
1. Run `dmost_planfile` in rawdata directory to generate .plan files for each mask
2. Conda activate pypeit
3. Run `dmost_setup` to generate directories and run pypeit setup files.
4. Examine mask.pypeit file to ensure its good.

5. `run_pypeit mask.pypeit -o`
6.  Examine spec2d files

7. Run flexure:  `dmost_flexure` which generates file
8. Run telluric: `dmost_telluric` which generates telluric spectrum
9. Run co-add, dmost_create_marz and marz
10. Read in marz and run MCMC



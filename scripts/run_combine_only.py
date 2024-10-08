#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import glob


from astropy.table import Table
from astropy.io import ascii


from dmost import *

from dmost.core.dmost_utils import write_dmost 
from dmost.core             import dmost_combine_exp


def run_dmost(maskname, rerun_chi2 = 0, rerun_emcee = 0, rerun_coadd = 0):
    '''
    Main execution script for dmost 

    '''
    DEIMOS_REDUX = os.getenv('DEIMOS_REDUX')
    data_dir     = DEIMOS_REDUX+maskname+'/'


    # CREATE OR READ DMOST OUTPUT TABLE
    slits, mask, nexp, outfile = dmost_create_maskfile.create_single_mask(data_dir, maskname)


  # COMBINE VELOCITIES ACROSS MULTIPLE EXPOSURES
    slits, mask = dmost_combine_exp.combine_exp(data_dir,slits, mask)
    write_dmost(slits,mask,outfile)

    # CALCULATE EQUIVALENT WIDTH QUANTITIES
    #slits, mask  = dmost_EW.run_coadd_EW(data_dir, slits, mask)
    #write_dmost(slits,mask,outfile)




    print()
    return slits,mask
    
    
#####################################################    
#####################################################    
#####################################################    
#####################################################    
def main(*args):

    
    for msk in sys.argv[1:]:
        print(msk)
        run_dmost(msk)
    
    
if __name__ == "__main__":
    main()

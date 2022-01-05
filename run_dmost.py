#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii

from pypeit import pypeitsetup



def run_dmost(maskname, rerun_chi2 = 0, erun_emcee = 0, rerun_coadd = 0):
    '''
    Main drive for dmost 
    '''


    # CREATE OR READ DMOST OUTPUT TABLE
    slits, mask = dmost_create_maskfile.create_single_mask(maskname)


    # RUN FLEXURE
    if ~(np.sum(mask['flag_flexure']) == nexp):
        slits,mask = dmost_flexure.run_flexure(data_dir,slits,mask)
        write_dmost(slits,mask,outfile)


    # RUN TELLURIC -- USE OLD FILES IF AVAILABLE
    tfile = glob.glob(data_dir+'/dmost/telluric_'+maskname+'*.fits')
    if ~(np.sum(mask['flag_telluric']) == nexp) | ~(np.size(tfile) == nexp):
        slits,mask  = dmost_telluric.run_telluric_mask(data_dir, slits, mask)
        write_dmost(slits,mask,outfile)


    # RUN CHI2 TEMPLATE FINDER ON COMBINED DATA
    if ~(np.sum(mask['flag_template']) == nexp) | (rerun_chi2 == 1):
        slits,mask  = dmost_chi2_template.run_chi2_templates(data_dir, slits, mask)
        write_dmost(slits,mask,outfile)


    # RUN EMCEE
    if ~(np.sum(mask['flag_emcee']) == nexp) | (rerun_emcee == 1):
        slits, mask  = dmost_emcee.run_emcee(data_dir, slits, mask,outfile)
        write_dmost(slits,mask,outfile)

    # RUN LOW SN EMCEE
    if ~(np.sum(mask['flag_emcee']) == nexp) | (rerun_coadd == 1):
        slits, mask  = dmost_coadd_emcee.run_coadd_emcee(data_dir, slits, mask,outfile)
        write_dmost(slits,mask,outfile)

    # CALCULATE EQUIVALENT WIDTH QUANTITIES
    #slits, mask  = dmost_EW.run_ew(data_dir, slits, mask,outfile)


    # COMBINE VELOCITIES ACROSS MULTIPLE EXPOSURES
    slits, mask = dmost_combine_exp.combine_exp(slits, mask)
    write_dmost(slits,mask,outfile)


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

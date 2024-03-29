#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import glob


from astropy.table import Table,Column
from astropy.io import ascii


from dmost import *

from dmost.core.dmost_utils import write_dmost 
from dmost.core             import dmost_combine_exp

######################################################
# FILL A COLUMN
def filled_column(name, fill_value, size):
    """
    Tool to allow for large strings
    """
    return Column([fill_value]*int(size), name)


def run_dmost(maskname, rerun_chi2 = 0, rerun_emcee = 0, rerun_coadd = 0):
    '''
    Main execution script for dmost 

    '''
    DEIMOS_REDUX = os.getenv('DEIMOS_REDUX')
    data_dir     = DEIMOS_REDUX+maskname+'/'


    # CREATE OR READ DMOST OUTPUT TABLE
    slits, mask, nexp, outfile = dmost_create_maskfile.create_single_mask(data_dir, maskname)


    nslits = np.size(slits) 
    skew=filled_column('emcee_skew',-99*np.ones(nexp),nslits)
    ker =filled_column('emcee_kertosis',-99*np.ones(nexp),nslits)
    egood=filled_column('emcee_good',-1*np.ones(nexp),nslits)


    
    cskew=filled_column('coadd_skew',-99.,nslits)
    cker=filled_column('coadd_kertosis',-99.,nslits)
    cgood=filled_column('coadd_good',-99,nslits)
    cflag=filled_column('coadd_flag',-99,nslits)

   
    coadd_flag = filled_column('coadd_flag',-1,nslits)

    try:
        # ADD NEW COLUMNS
        slits.add_column(skew)
        slits.add_column(ker)
        slits.add_column(egood)

        slits.add_column(cskew)
        slits.add_column(cker)
        slits.add_column(cgood)
        print('adding columns')
    except:
        print('kertosis col exist')

    try:
        slits.add_column(coadd_flag)
    except:
        print('coadd exists')


    write_dmost(slits,mask,outfile)

    # RUN FLEXURE 
    if ~(np.sum(mask['flag_flexure']) == nexp):
        slits,mask = dmost_flexure.run_flexure(data_dir,slits,mask)
        write_dmost(slits,mask,outfile)



    # RUN TELLURIC -- USE OLD FILES IF AVAILABLE
    tfile = glob.glob(data_dir+'/dmost/telluric_'+maskname+'*.fits')
    #if ~(np.sum(mask['flag_telluric']) == nexp) | ~(np.size(tfile) == nexp):
    if  not (np.size(tfile) == nexp):

        slits,mask = dmost_telluric.run_telluric_mask(data_dir, slits, mask)
        slits,mask = dmost_chip_gap.run_chip_gap(data_dir, slits, mask, clobber=0)
        write_dmost(slits,mask,outfile)



    # RUN CHI2 TEMPLATE FINDER ON COMBINED DATA
    if ~(np.sum(mask['flag_template']) == nexp) | (rerun_chi2 == 1):
        slits,mask  = dmost_chi2_template.run_chi2_templates(data_dir, slits, mask)
        write_dmost(slits,mask,outfile)


    # RUN EMCEE
    #if ~(np.sum(mask['flag_emcee']) == nexp) | (rerun_emcee == 1):
    slits, mask  = dmost_emcee.run_emcee(data_dir, slits, mask,outfile)
    write_dmost(slits,mask,outfile)

    # RUN LOW SN EMCEE ON COADDED SPECTRA
    rerun_coadd=0
    #if ~(np.sum(mask['flag_coadd']) == nexp) | (rerun_coadd == 1):
    slits, mask  = dmost_coadd_emcee.run_coadd_emcee(data_dir, slits, mask,outfile)
    write_dmost(slits,mask,outfile)

    # COMBINE VELOCITIES ACROSS MULTIPLE EXPOSURES
    slits, mask = dmost_combine_exp.combine_exp(slits, mask)
    write_dmost(slits,mask,outfile)


    # CALCULATE EQUIVALENT WIDTH QUANTITIES
    slits, mask  = dmost_EW.run_coadd_EW(data_dir, slits, mask)
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

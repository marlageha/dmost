#!/usr/bin/env python

import numpy as np
from scipy import stats
from dmost import dmost_utils



########################################################
# Combine results from multiple exposures


########################################################
# SET SHORT VELOCITY VARIABLE FLAG WITHIN MASK
def set_short_var_flag(slits,mask,sys_mult,sys_exp_flr):


    for i,obj in enumerate(slits):
    

        m=obj['emcee_good'] == 1
        slits['flag_short_var'][i]  = -99
        if np.sum(m) > 1:


            err = sys_mult * (obj['emcee_v_err84'] - obj['emcee_v_err16'])/2.
            err2 = err**2 + sys_exp_flr**2
            ivar = 1./err2
        

            v_mean = np.average(obj['emcee_v'][m],weights=ivar[m])
            chi2   = np.sum((obj['emcee_v'][m] - v_mean)**2/err[m]**2)
            pv     = 1 - stats.chi2.cdf(chi2, np.sum(m)-1)

            if (pv == 0) | (pv < 1e-14):
                pv = 1e-14

            lpv = np.log10(pv)

            slits['var_short_pval'][i]  = lpv
            slits['var_short_max_v'][i] = np.max(obj['emcee_v'][m]) - np.min(obj['emcee_v'][m])
            slits['var_short_max_t'][i] = 24*(np.max(mask['mjd'][m])-np.min(mask['mjd'][m]))
            slits['flag_short_var'][i]  = 0

            if lpv < -4:
                slits['flag_short_var'][i]  = 1

    return slits



def combine_multiple_exp(obj, mask, nexp, sys_mult, sys_flr):

    '''
    Combine velocity and velocity errors for single object 
    across multiple exposures including systematic error term

    Parameters
    ----------
    obj: table
        single slit object
    f_acc_threshold, f_acc_coadd: 
        acceptance threshold from MCMC
    sys_exp:
        systematic error btw exposures, determine in notebook
    
    Returns
    -------
    v, verr, ncomb
        combined velocity, error and number of combined exposures
    '''

    v, verr, ncomb    = [-999,-99,0]
    
    # IS THIS A GALAXY?
    if (obj['marz_flag'] > 2):
        v     = obj['marz_z'] * 3e5
        verr  = 0
        ncomb = 100
        return v,verr,ncomb

    
    # USE VELOCITY IF ANY TWO SINLGE EXPOSURES ARE GOOD
    if (np.sum(obj['emcee_good'] == 1) > 1):
        
        vt,et = [], []
        for j in np.arange(0,nexp,1):
            if obj['emcee_good'][j]  == 1:

                verr_rand = (obj['emcee_v_err84'][j]-obj['emcee_v_err16'][j])/2.

                vt   = np.append(vt,obj['emcee_v'][j])
                et   = np.append(et,verr_sys)
                ncomb=ncomb+1

        v         = np.average(vt,weights = 1./et**2)
        verr_rand = np.sqrt(1./np.sum(1./et**2))        
        verr      = np.sqrt((sys_mult * verr_rand)**2 + sys_flr**2)   # RANDOM + SYSTEMATIC


        # USE COADD IF SINGLE COMBINED ERROR IS > 10 kms
        if (obj['coadd_good'] ==  1):
            cerr_rand = (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.
            cerr      = np.sqrt((sys_mult*cerr_rand)**2 + sys_flr**2)  # RANDOM + SYSTEMATIC COADD

            if (verr > 10):
                v         = obj['coadd_v']
                verr      = cerr
                verr_rand = cerr_rand
                ncomb     = nexp + 100.


    # IF NONE, THEN USE COADD 
    else:
        if (obj['coadd_good'] ==  1):
            v          = obj['coadd_v']
            verr_rand  = (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.  
            verr       = np.sqrt((sys_mult*verr_rand)**2 + sys_flr**2)  # RANDOM + SYSTEMATIC COADD
            ncomb      = nexp + 100.


    return v, verr, verr_rand, ncomb
  

  
  
def combine_exp(data_dir,slits, mask):
    '''
    Combine exposures in a single mask, 
    either single or multiple exposures
    '''  
  

    # DETERMINED IN PAPER I
    sys_mask_mult   = 1.4  # Multiplier for final mask error
    sys_mask_flr    = 1.1  # Floor for final mask error
    sys_exp_flr     = 0.3  # Floor only when comparing internal exposures

    logfile      = data_dir + mask['maskname'][0]+'_dmost.log'
    log          = open(logfile,'a')   

    # READ MASK, SLIT FILES
    nexp = mask['nexp'][0]
    
    dmost_utils.printlog(log,'{} Combining {} exposure results'.format(mask['maskname'][0],nexp))

    
    if (nexp > 0):
        for i,obj in enumerate(slits):

            v,verr,verr_rand,ncomb = combine_multiple_exp(obj,mask, nexp, sys_mask_mult, sys_mask_flr)
            slits['dmost_v'][i]          = v
            slits['dmost_v_err'][i]      = verr
            slits['dmost_v_err_rand'][i] = verr_rand
            slits['v_nexp'][i]           = ncomb
            slits['coadd_flag'][i]       = 0
            if ncomb > 99:
                slits['coadd_flag'][i] = 1  
                slits['v_nexp'][i]     = ncomb - 100


    # SET FLAG IF EXPOSURE VELOCITIES ARE VARIABLE
    slits = set_short_var_flag(slits,mask,sys_exp_mult,sys_exp_flr)


    ngal = np.sum((slits['marz_flag'] > 2))
    dmost_utils.printlog(log,'{} Velocities measured for {} galaxies'.format(mask['maskname'][0],ngal))

    nstar  = np.sum((slits['marz_flag'] < 2)) 
    ngood  = np.sum((slits['marz_flag'] < 2) & (slits['dmost_v_err'] > 0))
    ncoadd = np.sum((slits['marz_flag'] < 2) & (slits['dmost_v_err'] > 0) & (slits['coadd_flag'] ==1))


    dmost_utils.printlog(log,'{} Stellar velocities measured for {} of {} ({} coadds)'.format(mask['maskname'][0],ngood,nstar,ncoadd))
    log.close()

            
    return slits,mask

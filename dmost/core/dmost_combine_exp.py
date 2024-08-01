#!/usr/bin/env python

import numpy as np
from scipy import stats



########################################################
# Combine results from multiple exposures


# SET BINARY FLAG WITHIN MASK
def set_mask_binary_flag(slits,mask,sys_exp_mult,sys_exp_flr):


    for i,obj in enumerate(slits):
    

        m=obj['emcee_good'] == 1
        slits['vv_short_flag'][i]  = -99
        if np.sum(m) > 1:


            err = sys_exp_mult * (obj['emcee_v_err84'] - obj['emcee_v_err16'])/2.
            err = np.sqrt(err**2 + sys_exp_flr**2)
            ivar = 1./err*2
        

            v_mean = np.average(obj['emcee_v'][m],weights=ivar[m])
            chi2   = np.sum((obj['emcee_v'][m] - v_mean)**2/err[m]**2)
            pv     = 1 - stats.chi2.cdf(chi2, np.sum(m)-1)

            if (pv == 0) | (pv < 1e-14):
                pv = 1e-14

            lpv = np.log10(pv)

            slits['vv_short_pval'][i]  = lpv
            slits['vv_short_max_v'][i] = np.max(obj['emcee_v'][m]) - np.min(obj['emcee_v'][m])
            slits['vv_short_max_t'][i] = 24*(np.max(mask['mjd'][m])-np.min(mask['mjd'][m]))
            slits['vv_short_flag'][i]  = 0

            if lpv < -4:
                slits['vv_short_flag'][i]  = 1



    return slits

def combine_multiple_exp(obj, mask, nexp, sys_exp_mult,sys_exp_flr):

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
        v    = obj['marz_z'] * 3e5
        verr = 0
        ncomb= 100
        return v,verr,ncomb

    
    # USE VELOCITY IF ANY EXPOSURE IS GOOD
    if np.any(obj['emcee_good'] == 1):
        vt,et = [], []
        for j in np.arange(0,nexp,1):
            if obj['emcee_good'][j]  == 1:
                vt   = np.append(vt,obj['emcee_v'][j])
                terr = sys_exp_mult * (obj['emcee_v_err84'][j]-obj['emcee_v_err16'][j])/2.
                et   = np.append(et,terr)
                ncomb=ncomb+1

        v    = np.average(vt,weights = 1./et**2)
        sverr = np.sqrt(1./np.sum(1./et**2))
        verr  = np.sqrt(sverr**2 + ncomb*sys_exp_flr**2)


        # USE COADD IF SINGLE ERROR IS > 10 kms
        if (obj['coadd_good'] ==  1):
            cerr    = sys_exp_mult*(obj['coadd_v_err84']-obj['coadd_v_err16'])/2.

            print('{} verr, cerr:  {:0.2f} {:0.2f} '.format(obj['objname'],verr,cerr))

            if (1.5*cerr < verr) | (verr > 10):
                v     = obj['coadd_v']
                verr  = cerr
                ncomb = nexp + 100.


    # IF NONE, THEN USE COADD 
    else:
        if (obj['coadd_good'] ==  1):
            v     = obj['coadd_v']
            verr  = sys_exp_mult * (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.  # NO FLOOR FOR COADD
            ncomb = nexp + 100.


    return v,verr,ncomb
  
    
#def combine_single_exp(obj, mask, sys_exp_mult, sys_exp_flr):

#    v, verr, ncomb    = [-1,-1,0]
    
    # IS THIS A GALAXY?
#    if (obj['marz_flag'] > 2):
#        v    = obj['marz_z'] * 3e5
#        verr = 0
#        ncomb= 100
#        return v,verr,ncomb

    
    # USE VELOCITY IF ANY EXPOSURE IS GOOD
#    if (obj['emcee_good'] == 1):
#        v     = obj['emcee_v']
#        terr  = sys_exp_mult * (obj['emcee_v_err84']-obj['emcee_v_err16'])/2.
        
        # ADDING SYS ERROR 
#        err   =  np.sqrt(terr**2 + sys_exp_flr**2)
#        ncomb = 1                     
            
#    return v,verr,ncomb    
  
  
def combine_exp(data_dir,slits, mask):
    '''
    Combine exposures in a single mask, 
    either single or multiple exposures
    '''  
  

    sys_exp_mult = 1.3
    sys_exp_flr  = 0.4

    logfile      = data_dir + mask['maskname'][0]+'_dmost.log'
    log          = open(logfile,'a')   

    # READ MASK, SLIT FILES
    nexp = mask['nexp'][0]
    
    dmost_utils.printlog(log,'{} Combining {} exposure results'.format(mask['maskname'][0],nexp))


    
    if (nexp > 0):
        for i,obj in enumerate(slits):

            v,verr,ncomb = combine_multiple_exp(obj,mask, nexp, sys_exp_mult,sys_exp_flr)
            slits['dmost_v'][i]     = v
            slits['dmost_v_err'][i] = verr
            slits['v_nexp'][i]      = ncomb
            slits['coadd_flag'][i]  = 0
            if ncomb > 99:
                slits['coadd_flag'][i] = 1                           

    slits = set_mask_binary_flag(slits,mask,sys_exp_mult,sys_exp_flr)

    ngal = np.sum((slits['marz_flag'] > 2))
    dmost_utils.printlog(log,'{} Velocities measured for {} galaxies'.format(mask['maskname'][0],ngal))

    nstar  = np.sum((slits['marz_flag'] < 3) & (slits['collate1d_SN'] > 3))
    ngood  = nstar & np.sum((slits['dmost_v_err'] > 0))
    ncoadd = nstar & np.sum((slits['dmost_v_err'] > 0)) & np.sum((slits['coadd_flag'] ==1))

    dmost_utils.printlog(log,'{} Stellar velocities measured for {} of {} ({} coadds)'.format(mask['maskname'][0],ngood,nstar,ncoadd))
    log.close()

            
    return slits,mask

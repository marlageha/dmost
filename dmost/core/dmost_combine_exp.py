#!/usr/bin/env python

import numpy as np
from scipy import stats
from dmost import dmost_utils



########################################################
# Combine results from multiple exposures


########################################################
def ew_sys_errors(slits):

    # FOR NaI and MgI, multiplier only
    m = slits['naI_err'] > 0
    slits['naI_err'][m] =  0.7*slits['naI_err'][m]


    m = slits['mgI_err'] > 0
    slits['mgI_err'][m] =  0.7*slits['mgI_err'][m]

    # CaT has two multiplers:  Guass and GL profiles
    mcat    = slits['cat_err'] > 0
    m_gauss = (slits['cat_gl'] == 1) | (slits['cat_gl'] == 3):
    m_gl    = (slits['cat_gl'] == 2) | (slits['cat_gl'] == 4):

    slits['cat_err'][mcat&m_guass] = 1.2 * slits['cat_err'][mcat&m_guass]
    slits['cat_err'][mcat&m_gl]    = 0.3 * slits['cat_err'][mcat&m_gl]


    return slits

########################################################
# SET SHORT VELOCITY VARIABLE FLAG WITHIN MASK
def set_short_var_flag(slits,mask,sys_exp_mult,sys_exp_flr):


    for i,obj in enumerate(slits):
    

        m=obj['emcee_good'] == 1
        slits['vv_short_flag'][i]  = -99
        if np.sum(m) > 1:


            err = sys_exp_mult * (obj['emcee_v_err84'] - obj['emcee_v_err16'])/2.
            err2 = err**2 + sys_exp_flr**2
            ivar = 1./err2
        

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

def combine_multiple_exp(obj, mask, nexp, sys_exp_mult,sys_exp_flr,sys_coadd_mult,sys_coadd_flr):

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

    
    # USE VELOCITY IF ANY TWO SINLGE EXPOSURES ARE GOOD
    if (np.sum(obj['emcee_good'] == 1) > 1):
        
        vt,et = [], []
        for j in np.arange(0,nexp,1):
            if obj['emcee_good'][j]  == 1:

                verr_rand = (obj['emcee_v_err84'][j]-obj['emcee_v_err16'][j])/2.
                verr_sys = np.sqrt((sys_exp_mult * verr_rand)**2 + sys_exp_flr**2)

                vt   = np.append(vt,obj['emcee_v'][j])
                et   = np.append(et,verr_sys)
                ncomb=ncomb+1

        v    = np.average(vt,weights = 1./et**2)
        sverr = np.sqrt(1./np.sum(1./et**2))
        verr  = np.sqrt(sverr**2)
        


        # USE COADD IF SINGLE COMBINED ERROR IS > 10 kms
        if (obj['coadd_good'] ==  1):
            cerr_rand = (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.
            cerr      = np.sqrt((sys_coadd_mult*cerr_rand)**2 + sys_coadd_flr**2)

            if (verr > 10):
                v     = obj['coadd_v']
                verr  = cerr
                ncomb = nexp + 100.


    # IF NONE, THEN USE COADD 
    else:
        if (obj['coadd_good'] ==  1):
            v     = obj['coadd_v']
            cerr  = (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.  
            verr  = np.sqrt((sys_coadd_mult*cerr)**2 + sys_coadd_flr**2)
            ncomb = nexp + 100.


    return v,verr,ncomb
  

  
  
def combine_exp(data_dir,slits, mask):
    '''
    Combine exposures in a single mask, 
    either single or multiple exposures
    '''  
  

    # THIS IS USED ONLY TO SET VARIABLE FLAG
    sys_exp_mult   = 1.4
    sys_exp_flr    = 0.3


    logfile      = data_dir + mask['maskname'][0]+'_dmost.log'
    log          = open(logfile,'a')   

    # READ MASK, SLIT FILES
    nexp = mask['nexp'][0]
    
    dmost_utils.printlog(log,'{} Combining {} exposure results'.format(mask['maskname'][0],nexp))


    
    if (nexp > 0):
        for i,obj in enumerate(slits):

            v,verr,ncomb = combine_multiple_exp(obj,mask, nexp)
            slits['dmost_v'][i]     = v
            slits['dmost_v_err'][i] = verr
            slits['v_nexp'][i]      = ncomb
            slits['coadd_flag'][i]  = 0
            if ncomb > 99:
                slits['coadd_flag'][i] = 1        

    # SET FLAG IF EXPOSURE VELOCITIES ARE VARIABLE
    slits = set_short_var_flag(slits,mask,sys_exp_mult,sys_exp_flr)

    # ADD EW SYSTEMATIC ERRORS HERE
    slits = ew_sys_errors(slits)


    ngal = np.sum((slits['marz_flag'] > 2))
    dmost_utils.printlog(log,'{} Velocities measured for {} galaxies'.format(mask['maskname'][0],ngal))

    nstar  = np.sum((slits['marz_flag'] < 2)) 
    ngood  = np.sum((slits['marz_flag'] < 2) & (slits['dmost_v_err'] > 0))
    ncoadd = np.sum((slits['marz_flag'] < 2) & (slits['dmost_v_err'] > 0) & (slits['coadd_flag'] ==1))


    dmost_utils.printlog(log,'{} Stellar velocities measured for {} of {} ({} coadds)'.format(mask['maskname'][0],ngood,nstar,ncoadd))
    log.close()

            
    return slits,mask

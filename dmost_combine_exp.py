import numpy as np


def combine_multiple_exp(obj, mask, f_acc_thresh = 0.69, f_acc_coadd = 0.65, sys_exp = 0.25):

    v, verr, ncomb    = [-1,-1,0]
    
    # IS THIS A GALAXY?
    if (obj['marz_flag'] > 2):
        v    = obj['marz_z'] * 3e5
        return v,verr,ncomb

    
    # USE VELOCITY IF ANY EXPOSURE IS GOOD
    if np.any(obj['emcee_f_acc'] > f_acc_thresh):
        
        for j in np.arange(0,nexp,1):
            if obj['emcee_f_acc'][j] > f_acc_thresh:
                v     = obj['emcee_v'][j]+mask['vhelio'][j]
                terr  = (obj['emcee_v_err84'][j]-obj['emcee_v_err16'][j])/2.
                err   = np.sqrt(terr**2 + sys_exp**2)
                ncomb=ncomb+1
        sum1 = np.sum(1./err**2)
        sum2 = np.sum(v/err**2)

        v    = sum2/sum1
        verr = np.sqrt(1./sum1)
        

    # IF NONE, THEN USE COADD WITH LOWER THRESHOLD
    else:
        if (obj['coadd_f_acc'] > 0.65):
            v     = obj['coadd_v']+np.mean(mask['vhelio'])
            verr  = (obj['coadd_v_err84']-obj['coadd_v_err16'])/2.
            ncomb = 1 + 100.

    return v,verr,ncomb
  
    
def combine_single_exp(obj, mask, f_acc_thresh = 0.69, sys_exp = 0.25):

    v, verr, ncomb    = [-1,-1,0]
    
    # IS THIS A GALAXY?
    if (obj['marz_flag'] > 2):
        v    = obj['marz_z'] * 3e5
        return v,verr,ncomb

    
    # USE VELOCITY IF ANY EXPOSURE IS GOOD
    if (obj['emcee_f_acc'] > 0.69):
        v     = obj['emcee_v']+mask['vhelio']
        terr  = (obj['emcee_v_err84']-obj['emcee_v_err16'])/2.
        err   =  np.sqrt(terr**2 + sys_exp**2)
        ncomb = 1                     
            
    return v,verr,ncomb    
  
  
def combine_exp():
  
  
    # READ MASK, SLIT FILES
    slits, mask = dmost_utils.read_dmost(file)
    nexp = mask['nexp'][0]
    
    
    if (nexp > 1):
        for i,obj in enumerate(slits):

            v,verr,ncomb = combine_multiple_exp(obj,mask)
            slits['dmost_v'][i]     = v
            slits['dmost_v_err'][i] = verr
            slits['v_nexp'][i]      = ncomb
            
            print(v,verr,ncomb)
            
            
    if (nexp == 1):
        for i,obj in enumerate(slits):

            v,verr,ncomb = combine_single_exp(obj,mask)
            slits['dmost_v'][i]     = v
            slits['dmost_v_err'][i] = verr
            slits['v_nexp'][i]      = ncomb
            
            
    write_dmost(slits,mask,file)

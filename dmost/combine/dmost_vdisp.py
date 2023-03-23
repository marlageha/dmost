import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii, fits


from astropy.table import vstack
from scipy.stats import norm

import glob
from dmost.core import dmost_utils


import emcee
import corner



######################################################
def lnprob_v(theta, vel, vel_err):
 
    lp = lnprior_v(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_v(theta, vel,vel_err)


######################################################
def lnprior_v(theta):
   
    #vr = theta[0],  
    #sig = theta[1]
    if (-500. < theta[0] < 500.) & (0. < theta[1] < 25.):
        return 0.0
    
    return -np.inf

######################################################
def lnlike_v(theta, vel,vel_err):

    # Gaussian with 
    term1 = -0.5*np.sum(np.log(vel_err**2 + theta[1]**2))
    term2 = -0.5*np.sum((vel - theta[0])**2/(vel_err**2 + theta[1]**2))
    term3 = -0.5*np.size(vel)* np.log(2*np.pi)

    lnl  = term1+term2+term3
    
    return lnl


######################################################
# INITIALIZE WALKERS
def initialize_walkers(vr_guess,sig_guess):

    # v, sig
    ndim, nwalkers = 2, 20
    p0 =  np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # vr 
    p0[:,0] = (p0[:,0]*50. - 25.) + vr_guess
    
    # W SHIFT
    p0[:,1] = (p0[:,1] * 2 - 1.) + sig_guess
    if sig_guess < 2:
            p0[:,1] = p0[:,1] + sig_guess


    return ndim,nwalkers,p0

######################################################
def run_sampler(sampler, p0, max_n):

    pos, prob, state = sampler.run_mcmc(p0, max_n,progress=False)
    
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100
        
        
    return sampler, convg, burnin

######################################################
def mcmc_vdisp(vel,vel_err, vr_guess, sig_guess, max_n = 5000, plot=1):


    ndim, nwalkers,p0   = initialize_walkers(vr_guess,sig_guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                          args=(vel, vel_err))

    sampler, convg, burnin = run_sampler(sampler, p0, max_n)
    #print(burnin, convg)
    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    if (plot == 1):
        # PLOT CORNER
        labels=['vr','sig']
        ndim=2
        samples = sampler.chain[:, 2*burnin:, :].reshape((-1, ndim))
        fig     = corner.corner(samples, labels=labels,show_titles=True,quantiles=[0.16, 0.5, 0.84])

    
    return sampler, theta





    

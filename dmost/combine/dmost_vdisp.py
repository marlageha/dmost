import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii, fits


from astropy.table import vstack
from scipy.stats import norm
from scipy.stats import kurtosis, skew

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

def calc_SK(sampler):

    flag_upper_limit = 0

    chain_v  = np.array(sampler.chain[:,100:, 0]).flatten()
    chain_s  = np.array(sampler.chain[:,100:, 1]).flatten()

    v_kt = kurtosis(chain_v)
    v_sw = skew(chain_v)

    s_kt = kurtosis(chain_s)
    s_sw = skew(chain_s)


    return v_kt,v_sw,s_kt,s_sw


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


######################################################
def calc_sigma(vel,vel_err, vr_guess, sig_guess,plot=1):

    sampler, theta = mcmc_vdisp(vel,vel_err, vr_guess, sig_guess, plot=plot)

    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        #print(tau,burnin, converged)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100

    if (burnin < 75):
        burnin = 75

    #print(burnin, convg)

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            vr = mcmc[1] 
            vr_err16 = mcmc[0]
            vr_err84= mcmc[2] 
            vr_err   =(vr_err84 - vr_err16)/2.   # APPROX
        if (ii==1):
            sigma = mcmc[1] 
            sigma_err16 = mcmc[0]
            sigma_err84 = mcmc[2] 

            sigma_err_m   = (sigma - sigma_err16)
            sigma_err_p   = (sigma_err84 - sigma)



    return vr, vr_err, sigma, sigma_err_p,sigma_err_m, sampler


    

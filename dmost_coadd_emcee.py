#!/usr/bin/env python

import numpy as np
import os,sys
import time
    
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits


import emcee
import corner
import glob
import numba
import h5py

import dmost_utils, dmost_create_maskfile

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit

#from multiprocessing import Pool
#os.environ["OMP_NUM_THREADS"] = "1"


DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')


###################################################
def mk_single_model(theta, wave, flux, ivar, twave,tflux, pflux,pwave, npoly, losvd_pix):
    
    # Velocity shift stellar model
    swave              = pwave * np.exp(theta[0]/2.997924e5)
    linear_shift_vflux = np.interp(twave,swave,pflux)
     
    # TRANSFORM BACK TO LINEAR, ON TELLURIC GRID
    syn_flux = tflux * linear_shift_vflux

    # SHIFT LINEAR SPECTRUM
    shift_syn_wave = twave + theta[1]*0.01


    # REBIN INTO DATA SPACE
    conv_int_spec = np.interp(wave,shift_syn_wave,syn_flux)
    

    # FIT CONTINUUM
    m  = (flux > np.percentile(flux,5)) & (flux < np.percentile(flux,99.9))
    p  = np.polyfit(wave[m],flux[m]/conv_int_spec[m],npoly,w=ivar[m])
    fit=np.poly1d(p)
    
    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    model = conv_int_spec * fit(wave)
    
       
    return model



######################################################
def lnprob_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,losvd_pix):
 
    lp = lnprior_v(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,losvd_pix)


######################################################
@numba.jit(nopython=True)
def lnprior_v(theta):
   
    #v = theta[0],  w = theta[1]
    if (-500. < theta[0] < 500.) & (-45. < theta[1] < 45.):
        return 0.0
    
    return -np.inf

######################################################
def lnlike_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,losvd_pix):

    # MAKE MODEL
    model = mk_single_model(theta, wave, flux, ivar, twave, tflux, pflux,pwave,npoly,losvd_pix)

    chi2 = ((flux - model)**2)*(ivar)
    lnl  = -0.5 * np.sum(chi2)
    
    return lnl


######################################################
# INITIALIZE WALKERS
def initialize_walkers(vguess,wguess):

    # v, w
    ndim, nwalkers = 2, 20
    p0 =  np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # v 
    p0[:,0] = (p0[:,0]*50. - 25.) + vguess
    
    # W SHIFT
    p0[:,1] = (p0[:,1] * 10 - 5) + wguess
    
    return ndim,nwalkers,p0



######################################################
def read_best_template(pfile):
    pfile    = DEIMOS_RAW + '/templates/pheonix/'+pfile
    phdu     = fits.open(pfile)
    data     = phdu[1].data
    phx_flux = np.array(data['flux']).flatten()
    phx_logwave= np.array(data['wave']).flatten()
    
    return phx_logwave, phx_flux



######################################################
def run_sampler(sampler, p0, max_n):


    index = 0
    convg = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf


    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # check convergence every 200 steps
        if sampler.iteration % 200:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            convg = 1
            break
        old_tau = tau

    
    return sampler, convg


######################################################
def run_emcee_single(data_dir, slits, mask, arg, wave, flux, ivar,\
                     telluric,sm_tell, sm_pflux,plogwave,npoly,losvd_pix):

    # INITIAL GUESSES
    vguess    = slits['chi2_v'][arg]
    wguess    = np.mean(slits['telluric_w'][arg,:])
    if np.abs(wguess) > 40:
        wguess = 0


    # INITIALIZEWALKERS
    ndim, nwalkers,p0         = initialize_walkers(vguess,wguess)
    max_n = 3000


    # BACKEND FILENAME
    filename = data_dir+'/emcee/'+mask['maskname'][0]+'_'+slits['maskdef_id'][arg]+'.h5'


    # SETUP BACKEND
    backend = emcee.backends.HDFBackend(filename,name = 'coadd',read_only=False)

    if not os.path.isfile(filename): 

        # IF SAVED FILE DOESN"T EXIST, CREATE AND RUN
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                args=(wave, flux, ivar, telluric['wave'],sm_tell, \
                                      sm_pflux,plogwave,npoly,losvd_pix),a=0.5,backend=backend)

        sampler, convg = run_sampler(sampler, p0, max_n)
        slits['coadd_converge'][arg] = convg

    else:
        if not ('coadd' in h5py.File(filename).keys()):

            # IF FILES EXISTS, BUT EXPOSURE DOESN'T, RUN SAMPLER
            backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                    args=(wave, flux, ivar, telluric['wave'],sm_tell, \
                                          sm_pflux,plogwave,npoly,losvd_pix),a=0.5,backend=backend)
            sampler, convg = run_sampler(sampler, p0, max_n)
            slits['coadd_converge'][arg] = convg

        else:
            print('Reading previous chains')
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                        args=(wave, flux, ivar, telluric['wave'],sm_tell, \
                                          sm_pflux,plogwave,npoly,losvd_pix),a=0.5,backend=backend)


    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))

    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]
    print(burnin,arg,theta)

    slits['coadd_f_acc'][arg]    = np.mean(sampler.acceptance_fraction)

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            slits['coadd_v'][arg] = mcmc[1]
            slits['coadd_v_err16'][arg] = mcmc[0]
            slits['coadd_v_err84'][arg] = mcmc[2]

        if (ii==1):
            slits['coadd_w'][arg] = mcmc[1]
            slits['coadd_w_err'][arg] = (mcmc[0]+mcmc[2])/2.


    return sampler, slits, theta, burnin


######################################################

def coadd_emcee_allslits(data_dir, slits, mask, arg, telluric,pdf):


    # READ COADDED DATA
    jhdu = fits.open(data_dir+'/collate1d/'+slits['collate1d_filename'][arg])

    jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(jhdu) 
    wave_lims = dmost_utils.vignetting_limits(slits[arg],0,jwave)

    wave = jwave[wave_lims]
    flux = jflux[wave_lims]
    ivar = jivar[wave_lims]


    # READ STELLAR TEMPLATE 
    pwave,pflux = read_best_template(slits['chi2_tfile'][arg])
    plogwave    = np.exp(pwave)

    # PARAMETERS -- USE MEAN LSF
    m         = slits['fit_los'][arg,:] > 0
    losvd_pix = np.mean(slits['fit_los'][arg,m]/ 0.01)


    # SMOOTH TEMPLATES 
    sm_pflux = scipynd.gaussian_filter1d(pflux,losvd_pix,truncate=3)
    sm_tell  = scipynd.gaussian_filter1d(telluric['flux'],losvd_pix,truncate=3)
    npoly = 5


    # RUN EMCEE!!
    ###################
    sampler, slits, theta, burnin = run_emcee_single(data_dir, slits, mask, arg, wave, flux, ivar,\
                               telluric,sm_tell, \
                               sm_pflux,plogwave,npoly,losvd_pix)
    ###################




    # MAKE BEST MODEL 
    model = mk_single_model(theta, wave, flux, ivar, telluric['wave'],sm_tell, \
                        sm_pflux,plogwave, npoly,losvd_pix)
         

    # SAVE QUALITY OF FIT INFO
    slits['coadd_lnprob'][arg] = np.sum((flux - model)**2 * ivar)/(np.size(flux)-3)



    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))


    for ii in range(20):
        ax1.plot(sampler.chain[ii,:,0], color="k",linewidth=0.5)
        ax1.axvline(burnin)
        ax1.set_title('f_acc = {:0.3f}  v = {:0.2f}'.format(np.mean(sampler.acceptance_fraction),slits['coadd_v'][arg]))


    for ii in range(20):
        ax2.plot(sampler.chain[ii,:,1], color="k",linewidth=0.5)
        ax2.axvline(burnin)
        ax2.set_title('w = {:0.2f}'.format(slits['coadd_w'][arg]))

    pdf.savefig()
    plt.close(fig)


    # PLOT SPECTRUM
    plt.figure(figsize=(20,5))
    m = (flux > np.percentile(flux,0.1)) & (flux < np.percentile(flux,99.9))
    #            plt.plot(wave,flux,linewidth=0.8)
    plt.plot(wave[m],flux[m],'k',linewidth=0.8)
    plt.plot(wave,model,'r',linewidth=0.8,alpha=0.8,label='model')
    plt.title('SN = {:0.1f}   chi2 = {:0.1f}'.format(slits['collate1d_SN'][arg],\
                                              slits['coadd_lnprob'][arg]))

    pdf.savefig()
    plt.close(fig)

    # PLOT CORNER
    labels=['v','w']
    ndim=2
    samples   = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=labels,show_titles=True,quantiles=[0.16, 0.5, 0.84])

    pdf.savefig()
    plt.close('all')



    return slits
    

######################################################

def run_coadd_emcee(data_dir, slits, mask, outfile, clobber=0):
    

    file  = data_dir+'/QA/coadd_emcee_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
 
    # READ THE AVERAGE TELLURIC
    tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][0]+'_'+mask['fname'][0]+'*.fits')
    telluric = Table.read(tfile[0])
       

    SNmax = 20
    SNmin = 2
    m = (slits['collate1d_SN'] < SNmax) & (slits['collate1d_SN'] > SNmin) & (slits['marz_flag'] < 3)

    nslits = np.sum(m)
    print('{}  Coadd emcee with {} slits w/SN < {}'.format(mask['maskname'][0],\
                                                                nslits,SNmax))
    

    # FOR EACH SLIT
    for ii,slt in enumerate(slits): 

    
        if (slt['collate1d_SN'] < SNmax) &  (slt['collate1d_SN'] > SNmin) & (slt['marz_flag'] < 3):
      

            # RUN EMCEE ON COADD
            slits = coadd_emcee_allslits(data_dir, slits, mask, ii ,telluric,pdf)
            
    pdf.close()
    plt.close('all')
#    if write_file:
#        outfile      = data_dir+'/dmost/dmost_mask_'+mask['maskname'][0]+'.fits'
    outfile = 'coadd_test.fits'
    dmost_create_maskfile.write_dmost(slits,mask,outfile)
        
    return slits, mask


    

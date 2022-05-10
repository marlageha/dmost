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
from numba import njit
import h5py

import dmost_utils, dmost_create_maskfile

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit

#from multiprocessing import Pool
#os.environ["OMP_NUM_THREADS"] = "1"


DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')

from numpy.core.multiarray import interp as compiled_interp



###################################################
def mk_single_model(theta, wave, flux, ivar, twave,tflux, pflux,pwave, npoly, pfit):
    
    # Velocity shift stellar model
    swave              = shift_v(pwave,theta[0])
    linear_shift_vflux = compiled_interp(twave,swave,pflux)

     
    # TRANSFORM BACK TO LINEAR, ON TELLURIC GRID
    syn_flux = tflux * linear_shift_vflux
    shift_syn_wave = twave + theta[1]*0.02

    # REBIN INTO DATA SPACE
    conv_int_spec = compiled_interp(wave,shift_syn_wave,syn_flux)

    # FIT CONTINUUM
    fit = faster_polyval(pfit, wave)

    
    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    model = conv_int_spec * fit

    return model


@njit
def shift_v(pwave,v):
    swave = pwave * np.e**(v/2.997924e5)
    return swave


def faster_polyval(p, x):
    y = np.zeros(x.shape, dtype=float)
    for i, v in enumerate(p):
        y *= x
        y += v
    return y


############################
def get_poly_fit(theta, wave, flux, ivar, twave,tflux, pflux,pwave, npoly):
    
    # Velocity shift stellar model
    swave              = pwave * np.exp(theta[0]/2.997924e5)
    linear_shift_vflux = np.interp(twave,swave,pflux)

    # TRANSFORM BACK TO LINEAR, ON TELLURIC GRID
    syn_flux = tflux * linear_shift_vflux

    # SHIFT LINEAR SPECTRUM
    shift_syn_wave = twave + theta[1]*0.02


    # REBIN INTO DATA SPACE
    conv_int_spec = np.interp(wave,shift_syn_wave,syn_flux)
    

    # FIT CONTINUUM
    m  = (flux > np.percentile(flux,5)) & (flux < np.percentile(flux,99.9))
    p  = np.polyfit(wave[m],flux[m]/conv_int_spec[m],npoly,w=ivar[m])

       
    return p



######################################################
def lnprob_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,pfit):
 
    lp = lnprior_v(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,pfit)


######################################################
@numba.jit(nopython=True)
def lnprior_v(theta):
   
    #v = theta[0],  w = theta[1]
    if (-600. < theta[0] < 500.) & (-40. < theta[1] < 40.):
        return 0.0
    
    return -np.inf

######################################################
def lnlike_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly, pfit):

    # MAKE MODEL
    model = mk_single_model(theta, wave, flux, ivar, twave, tflux, pflux,pwave,npoly,pfit)

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
    
    plogwave    = np.e**(phx_logwave)

    return plogwave, phx_flux

######################################################
def mk_emcee_plots(pdf, slits, nexp, arg, sampler, wave, flux, model):

    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))

    burnin=slits['emcee_burnin'][arg,nexp]

    for ii in range(20):
        ax1.plot(sampler.chain[ii,:,0], color="k",linewidth=0.5,alpha=0.8)
    ax1.set_title('f_acc = {:0.3f}  v = {:0.2f}'.format(np.mean(sampler.acceptance_fraction),slits['emcee_v'][arg,nexp]))
    ax1.axvline(burnin)

    for ii in range(20):
        ax2.plot(sampler.chain[ii,:,1], color="k",linewidth=0.5,alpha=0.8)
    ax2.set_title('w = {:0.2f}'.format(slits['emcee_w'][arg,nexp]))
    ax2.axvline(burnin)

    pdf.savefig()
    plt.close(fig)


    # PLOT SPECTRUM
    plt.figure(figsize=(20,5))
    m = (flux > np.percentile(flux,0.1)) & (flux < np.percentile(flux,99.9))
    plt.plot(wave[m],flux[m],'k',linewidth=0.8)
    plt.plot(wave,model,'r',linewidth=0.8,alpha=0.8,label='model')
    plt.title('SN = {:0.1f}   chi2 = {:0.1f}'.format(slits['SN'][arg,nexp],\
                                                  slits['emcee_lnprob'][arg,nexp]))
    plt.legend(title='det={}  xpos={}'.format(slits['DET'][arg,nexp],\
                         int(slits['SPAT_PIXPOS'][arg,nexp])),loc=1)


    pdf.savefig()
    plt.close(fig)

    # PLOT CORNER
    labels=['v','w']
    ndim=2
    samples   = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=labels,show_titles=True,quantiles=[0.16, 0.5, 0.84])

    pdf.savefig()
    plt.close('all')


    return pdf

######################################################
def run_sampler(sampler, p0, max_n):

    # POOL
    #with Pool() as pool:
    #sampler = sampler.sample(p0, iterations=max_n,progress = True)
    pos, prob, state = sampler.run_mcmc(p0, max_n,progress=False)
    
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        print(tau,burnin, converged)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100
        
        
    return sampler, convg, burnin

######################################################
def test_emcee_exists(filename, fname):
    erun = -99
    if not os.path.isfile(filename):
        erun = 1
    if os.path.isfile(filename): 
        if not (fname in h5py.File(filename).keys()):
            erun = 1
        else:
            erun = 0

    return erun

######################################################
def run_emcee_single(data_dir, slits, mask, nexp, arg, wave, flux, ivar,\
                     twave,sm_tell, sm_pflux,plogwave,npoly,pfit):

    # SET GUESSES AND INITIALIZE WALKERS
    vguess    = slits['chi2_v'][arg]
    wguess    = slits['telluric_w'][arg,nexp]
    if np.abs(wguess) > 40:
        wguess = 0
    ndim, nwalkers,p0         = initialize_walkers(vguess,wguess)
    max_n = 1000

    # BACKEND FILENAME
    filename = data_dir+'/emcee/'+mask['maskname'][0]+'_'+slits['maskdef_objname'][arg]+'.h5'


    # SETUP BACKEND
    backend = emcee.backends.HDFBackend(filename,name = mask['fname'][nexp],read_only=False)

    # SET AND RUN SAMPLER
    erun = test_emcee_exists(filename, mask['fname'][nexp])
    
    # IF SAVED FILE DOESN"T EXIST, CREATE AND RUN
    if (erun == 1): 
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                  args=(wave, flux, ivar, twave,sm_tell,sm_pflux,plogwave,npoly,pfit),\
                                  backend=backend)#,pool=pool

        sampler, convg, burnin = run_sampler(sampler, p0, max_n)

        
    # OR JUST READ IN PREVIOUS RESULTS
    if (erun == 0): 
        print('Reading previous chains')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                  args=(wave, flux, ivar, twave,sm_tell,sm_pflux,plogwave,npoly,pfit),\
                                  backend=backend)
        try:
            tau    = sampler.get_autocorr_time(tol=0)
            burnin = int(2 * np.max(tau))
            converged = np.all(tau * 100 < sampler.iteration)
            print(tau,burnin, converged)
            convg = np.sum(converged)
        except:
            convg=0
            burnin=100

    if (burnin < 75):
        burnin = 75

    slits['emcee_converge'][arg,nexp] = convg
    slits['emcee_burnin'][arg,nexp]   = burnin


    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    slits['emcee_f_acc'][arg,nexp]    = np.mean(sampler.acceptance_fraction)
    slits['emcee_nsamp'][arg,nexp]    = sampler.iteration

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            slits['emcee_v'][arg,nexp] = mcmc[1]
            slits['emcee_v_err16'][arg,nexp] = mcmc[0]
            slits['emcee_v_err84'][arg,nexp] = mcmc[2]

        if (ii==1):
            slits['emcee_w'][arg,nexp] = mcmc[1]
            slits['emcee_w_err16'][arg,nexp] = mcmc[0]
            slits['emcee_w_err84'][arg,nexp] = mcmc[2]

    return sampler, slits, theta


######################################################

def emcee_allslits(data_dir, slits, mask, nexp, hdu, telluric,SNmin):
    
    # OPEN PDF QA FILE
    file  = data_dir+'QA/emcee_'+mask['maskname'][nexp]+'_'+mask['fname'][nexp]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
   

    
    # LOOP OVER EACH SLIT
    for arg in np.arange(0,np.size(slits),1,dtype='int'):


        if (slits['SN'][arg,nexp] > SNmin) & (slits['marz_flag'][arg] < 3) & \
           (bool(slits['chi2_tfile'][arg].strip())) & (slits['flag_skip_exp'][arg,nexp] != 1):
            
            
            # READ DATA AND SET VIGNETTING LIMITS
            wave, flux, ivar, sky = dmost_utils.load_spectrum(slits[arg],nexp,hdu)
            wave_lims             = dmost_utils.vignetting_limits(slits[arg],nexp,wave)
            wave = wave[wave_lims]
            flux = flux[wave_lims]
            ivar = ivar[wave_lims]

            # CORRECT CHIP GAP
            #flux,ivar = dmost_utils.correct_chip_gap(slits['chip_gap_corr'][arg,nexp],slits['ccd_gap_b'][arg,nexp],wave,flux,ivar)


            # READ STELLAR TEMPLATE 
            plogwave,pflux = read_best_template(slits['chi2_tfile'][arg])
    
            
            # CONTINUUM POLYNOMIAL SET BY SN LIMITS
            npoly = 5
            if (slits['SN'][arg][nexp]) > 100:
                npoly=7

                
            # TRIM WAVELENGTH OF TEMPLATES TO SPEED UP COMPUTATION
            dmin = np.min(wave) - 20
            dmax = np.max(wave) + 20
            mt = (telluric['wave'] > dmin) & (telluric['wave']<dmax)
            mp = (plogwave > dmin) & (plogwave<dmax)
            

            # SMOOTH TEMPLATES 
            losvd_pix = slits['fit_lsf'][arg,nexp]/ 0.02
            sm_pflux  = scipynd.gaussian_filter1d(pflux[mp],losvd_pix,truncate=3)
            pwave     = plogwave[mp]
            
            sm_tell   = scipynd.gaussian_filter1d(telluric['flux'][mt],losvd_pix,truncate=3)
            twave=telluric['wave'][mt]


            vguess    = slits['chi2_v'][arg]
            wguess    = slits['telluric_w'][arg,nexp]
            if np.abs(wguess) > 40:
                wguess = 0
            
            print('SN = {:0.1f} det={}  xpos={}'.format(slits['SN'][arg,nexp],slits['DET'][arg,0],int(slits['SPAT_PIXPOS'][arg,0])))

            try:
                pfit = get_poly_fit([vguess, wguess], wave, flux, ivar, twave,\
                                                  sm_tell,sm_pflux,pwave, npoly)
            except:
                pfit = np.ones(npoly+1)


            # RUN EMCEE!!
            ###################
            t0 = time.time()
            sampler, slits, theta = run_emcee_single(data_dir, slits, mask, nexp, arg, wave, flux, ivar,\
                                       twave,sm_tell, sm_pflux,pwave,npoly,pfit)
            t1=time.time()
            print('Time = {:0.5f}'.format(t1-t0))
            ###################




            # MAKE BEST MODEL 
            model = mk_single_model(theta, wave, flux, ivar, twave,sm_tell, \
                                    sm_pflux,pwave, npoly,pfit)

            # SAVE QUALITY OF FIT INFO
  
            slits['emcee_lnprob'][arg,nexp]   = np.sum((flux - model)**2 * ivar)/(np.size(flux)-2)
        
                     

            # PLOT STUFF
            pdf = mk_emcee_plots(pdf, slits, nexp, arg, sampler, wave, flux, model)


    pdf.close()
    plt.close('all')

    return slits
    

######################################################

def run_emcee(data_dir, slits, mask, outfile, clobber=0):
    
       
    # FOR EACH EXPOSURE
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        # READ SPEC1D FILE
        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        nslits      = np.size(slits)
        
        # READ TELLURIC FOR THIS EXPOSURE
        tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][ii]+'_'+mask['fname'][ii]+'*.fits')
        telluric = Table.read(tfile[0])

        # WRITE TO SCREEN
        SNmin = 3.
        m = (slits['SN'][:,ii] > SNmin) & (slits['marz_flag'] < 3)
        nslits = np.sum(m)
        print('{} {} Emcee with {} slits w/SN > {}'.format(mask['maskname'][0],\
                                                                mask['fname'][ii],nslits,SNmin))

        # RUN EMCEE
        slits = emcee_allslits(data_dir, slits, mask, ii, hdu, telluric,SNmin)
        mask['flag_emcee'][ii] = 1
        
        # WRITE DMOST FILE
        dmost_create_maskfile.write_dmost(slits,mask,outfile)
        
    return slits, mask


    

#!/usr/bin/env python

import numpy as np
import os,sys
import time
    
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits


import emcee, corner

import glob
import h5py


from dmost import dmost_utils


import scipy.ndimage as scipynd
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew


#import numba
#from numba import njit
from numpy.core.multiarray import interp as compiled_interp


DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')

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
def get_poly_fit(theta, wave, flux, ivar, twave,tflux, pflux,pwave, npoly,pdf):
    
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
    p  = np.polyfit(wave[m],flux[m]/conv_int_spec[m],npoly,w=np.sqrt(ivar[m]))


    return p



######################################################
def lnprob_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,pfit):
 
    lp = lnprior_v(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_v(theta, wave, flux, ivar, twave,tflux, pflux,pwave,npoly,pfit)


######################################################
#@numba.jit(nopython=True)
def lnprior_v(theta):
   
    #v = theta[0],  w = theta[1]
    if (-700. < theta[0] < 700.) & (-60. < theta[1] < 60.):
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


    if (pfile.strip() == ''):
        pfile = 'grid1/dmost_lte_4500_5.0_-2.0_.fits'    
        print('** No template file, using default')

    pfile    = DEIMOS_RAW + '/templates/pheonix/'+pfile
    phdu     = fits.open(pfile)
    data     = phdu[1].data
    phx_flux = np.array(data['flux']).flatten()
    phx_logwave= np.array(data['wave']).flatten()
    
    plogwave    = np.e**(phx_logwave)

    return plogwave, phx_flux


######################################################
def run_sampler(sampler, p0, max_n):

       
    # POOL
    #with Pool() as pool:
    #sampler = sampler.sample(p0, iterations=max_n,progress = True)
    pos, prob, state = sampler.run_mcmc(p0, max_n,progress=False)

    
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        print(burnin, converged)
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
def run_emcee_single(data_dir, slits, mask, arg, wave, flux, ivar,\
                     twave,sm_tell, sm_pflux,plogwave,npoly,pfit):

    # SET GUESSES AND INITIALIZE WALKERS
    vguess    = slits['chi2_v'][arg]
    wguess    = np.mean(slits['telluric_w'][arg,:])
    if np.abs(wguess) > 40:
        wguess = 0
    ndim, nwalkers,p0         = initialize_walkers(vguess,wguess)
    max_n = 3000

    # BACKEND FILENAME
    filename = data_dir+'/emcee/'+mask['maskname'][0]+'_'+slits['objid'][arg]+'.h5'
    if slits['objname'][arg] == 'SERENDIP':
        filename = data_dir+'/emcee/'+mask['maskname'][0]+'_'+slits['objid'][arg]+'_SERENDIP'+str(slits['serendip'][arg])+'.h5'


    # SETUP BACKEND
    backend = emcee.backends.HDFBackend(filename,name = 'coadd',read_only=False)

    # SET AND RUN SAMPLER
    erun = test_emcee_exists(filename, 'coadd')
    
    # IF SAVED FILE DOESN"T EXIST, CREATE AND RUN
    if (erun == 1): 
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                  args=(wave, flux, ivar, twave,sm_tell,sm_pflux,plogwave,npoly,pfit),\
                                  backend=backend)#,pool=pool

        sampler, convg, burnin = run_sampler(sampler, p0, max_n)
         
    # OR JUST READ IN PREVIOUS RESULTS
    if (erun == 0): 
        print('Coadd:  Reading previous chains')
        print(filename)
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


    slits['coadd_converge'][arg] = convg
    slits['coadd_burnin'][arg]   = burnin
    slits['coadd_f_acc'][arg]    = np.mean(sampler.acceptance_fraction)
     
    theta  = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    print(theta)


    vhelio_avg = np.mean(mask['vhelio'])
    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            slits['coadd_v'][arg]       = mcmc[1] + vhelio_avg
            slits['coadd_v_err16'][arg] = mcmc[0] + vhelio_avg
            slits['coadd_v_err84'][arg] = mcmc[2] + vhelio_avg
            slits['coadd_v_err'][arg]   = (mcmc[2] - mcmc[0])/2.

        if (ii==1):
            slits['coadd_w'][arg]     = mcmc[1] 
            slits['coadd_w_err'][arg] = (mcmc[2] - mcmc[0])/2.

    # CALCULATE SKEW/KERTOSIS
    chain  = np.array(sampler.chain[:,burnin:, 0]).flatten()
    slits['coadd_kertosis'][arg] = kurtosis(chain)
    slits['coadd_skew'][arg]     = skew(chain)


    # DETERMINE IF MCMC WAS SUCCESSFUL
    slits['coadd_good'][arg] = 0
    if (np.abs(slits['coadd_kertosis'][arg]) < 1) & (np.abs(slits['coadd_skew'][arg])<1) & (slits['coadd_f_acc'][arg] > 0.69):
        slits['coadd_good'][arg]  = 1

    return sampler, slits, theta, burnin,vhelio_avg 


######################################################

def coadd_emcee_allslits(data_dir, slits, mask, arg, telluric,pdf):


    # READ COADDED DATA
    jhdu = fits.open(data_dir+'/collate1d_flex/'+slits['collate1d_filename'][arg])

    jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(slits[arg],jhdu) 
    wave_lims = dmost_utils.vignetting_limits(slits[arg],0,jwave)

    wave = jwave[wave_lims]
    flux = jflux[wave_lims]
    ivar = jivar[wave_lims]

    # CORRECT CHIP GAP
    fcorr = slits['chip_gap_corr_collate1d'][arg]
    bwave_gap = np.median(slits['chip_gap_b'][arg,:])
    flux,ivar = dmost_utils.correct_chip_gap(fcorr,bwave_gap,wave,flux,ivar)


    # READ STELLAR TEMPLATE 
    pwave,pflux = read_best_template(slits['chi2_tfile'][arg])


    # CONTINUUM POLYNOMIAL SET BY SN LIMITS
    npoly = 5
 
    # PARAMETERS -- USE MEAN LSF
    m         = slits['fit_lsf'][arg,:] > 0
    losvd_pix = np.mean(slits['fit_lsf'][arg,m])/ 0.02


    # TRIM WAVELENGTH OF TEMPLATES TO SPEED UP COMPUTATION
    dmin = np.min(wave) - 20
    dmax = np.max(wave) + 20
    mt = (telluric['wave'] > dmin) & (telluric['wave']<dmax)
    mp = (pwave > dmin) & (pwave<dmax)


    # SMOOTH TEMPLATES
    sm_pflux  = scipynd.gaussian_filter1d(pflux[mp],losvd_pix,truncate=3)
    pwave     = pwave[mp]

    sm_tell   = scipynd.gaussian_filter1d(telluric['flux'][mt],losvd_pix,truncate=3)
    twave=telluric['wave'][mt]


    vguess    = slits['chi2_v'][arg]
    wguess    = np.mean(slits['telluric_w'][arg,:])
    if np.abs(wguess) > 40:
        wguess = 0

    pfit = get_poly_fit([vguess, wguess], wave, flux, ivar, twave,\
                                          sm_tell,sm_pflux,pwave, npoly,pdf)


    # RUN EMCEE!!
    ###################
    t0 = time.time()
    sampler, slits, theta, burnin,vhelio_avg  = run_emcee_single(data_dir, slits, mask, arg, wave, flux, ivar,\
                               twave,sm_tell, sm_pflux,pwave,npoly,pfit)
    t1=time.time()
    print('Time = {:0.5f}'.format(t1-t0))
    ###################



    # MAKE BEST MODEL 
    model = mk_single_model(theta, wave, flux, ivar, twave,sm_tell, \
                        sm_pflux,pwave, npoly,pfit)
         

    # SAVE QUALITY OF FIT INFO
    slits['coadd_lnprob'][arg] = np.sum((flux - model)**2 * ivar)/(np.size(flux)-3)



    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))


    for ii in range(20):
        ax1.plot(sampler.chain[ii,:,0]+vhelio_avg, color="k",linewidth=0.5)
    ax1.axvline(burnin,label='burnin')
    ax1.set_title('f_acc = {:0.3f}  v = {:0.2f}'.format(np.mean(sampler.acceptance_fraction),slits['coadd_v'][arg]))

    for ii in range(20):
        ax2.plot(sampler.chain[ii,:,1], color="k",linewidth=0.5)
    ax2.axvline(burnin)
    ax2.set_title('w = {:0.2f}  converge = {}'.format(slits['coadd_w'][arg],slits['coadd_converge'][arg]))

    err = (slits['emcee_v_err84'][arg,:] - slits['emcee_v_err16'][arg,:])/2.
    str1 = ['{:0.1f}'.format(x) for x in slits['emcee_v'][arg,:]]
    str2 = ['{:0.1f}'.format(x) for x in err]
    str3 = ['{:0.2f}'.format(x) for x in slits['emcee_f_acc'][arg,:]]
    ax1.legend(title='v = '+', '.join(str1)+'\nverr ='+', '.join(str2)+'\nfacc ='+', '.join(str3), loc=3,fontsize=11)


    

    pdf.savefig()
    plt.close(fig)


    # PLOT SPECTRUM
    plt.figure(figsize=(20,5))
    plt.rcParams.update({'font.size': 14})
    m = (flux > np.percentile(flux,0.1)) & (flux < np.percentile(flux,99.9))
    plt.plot(wave,flux,linewidth=0.8)
    plt.plot(wave[m],flux[m],'k',linewidth=0.8)
    plt.plot(wave,model,'r',linewidth=0.8,alpha=0.8,label='model')
    plt.ylim(np.min(flux[m]),1.02*np.max(flux[m]))
    plt.title('{}   SN = {:0.1f}   chi2 = {:0.1f}'.format(slits['objname'][arg],slits['collate1d_SN'][arg],\
                                              slits['coadd_lnprob'][arg]))

    plt.legend(title='det={}  xpos={}\n chip gap = {:0.2f}'.format(slits['det'][arg,0],\
                         int(slits['spat_pixpos'][arg,0]),slits['chip_gap_corr_collate1d'][arg]),loc=1)

    pdf.savefig()
    plt.close(fig)

    # PLOT CORNER
    labels=['v','w']
    ndim=2
    sampler.chain[:,:,0] = sampler.chain[:,:,0]+vhelio_avg
    samples   = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=labels,show_titles=True,quantiles=[0.16, 0.5, 0.84])

    pdf.savefig()
    plt.close('all')



    return slits
    

def coadd_threshold(nexp, slt):
    '''
    Do coadd if less then 50% of exposures are good
    Good exposure has facc > 0.7 and error < 10 kms
    '''
    do_coadd = 0

    # SKIP IF EXPOSURE HAS 2 OR MORE GOOD SINGLE VALUES
    ngood = 0.
    for exp in np.arange(0,nexp,1):
        err   = (slt['emcee_v_err84'][exp] - slt['emcee_v_err16'][exp])/2.
        if (slt['emcee_good'][exp] == 1) & (err < 10.):
            ngood = ngood +1.

    single_good   = ngood/nexp
    coadd_thresh  = 0.5  

    if single_good < coadd_thresh:
        do_coadd = 1

    return do_coadd

######################################################

def run_coadd_emcee(data_dir, slits, mask, outfile, clobber=0):
    '''
    Run the coadd when single exposure fails
    '''    

    logfile      = data_dir + mask['maskname'][0]+'_dmost.log'
    log          = open(logfile,'a')   


    file  = data_dir+'/QA/coadd_emcee_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
 
    # READ THE AVERAGE TELLURIC
    tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][0]+'_'+mask['fname'][0]+'*.fits')
    telluric = Table.read(tfile[0])
       

    SNmax = 30
    SNmin = 2.
    nexp = mask['nexp'][0]


    m = (slits['collate1d_SN'] < SNmax) & (slits['collate1d_SN'] > SNmin) & (slits['marz_flag'] < 3)

    nslits = np.sum(m)
    print()
    dmost_utils.printlog(log,'{}  Coadd emcee with {} slits {} < SN < {} and poor emcee result'.format(mask['maskname'][0],\
                                                                nslits,SNmin,SNmax))
    

    # FOR EACH SLIT
    for ii,slt in enumerate(slits): 


        # THRESHOLD TO RUN COADD
        do_coadd = coadd_threshold(nexp, slt)

        is_good_slit = dmost_utils.is_good_slit(slt,remove_galaxies=1)

        if (slt['collate1d_SN'] < SNmax) &  (slt['collate1d_SN'] > SNmin) & (is_good_slit):# &  (do_coadd == 1):

            # RUN EMCEE ON COADD
            slits = coadd_emcee_allslits(data_dir, slits, mask, ii ,telluric,pdf)
            
            
    log.close()
    pdf.close()
    plt.close('all')
    mask['flag_coadd'][:] = 1

        
    return slits, mask


    

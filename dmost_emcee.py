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

def emcee_allslits(data_dir, slits, mask, nexp, hdu, telluric):
    
    SNmin = 10

    file  = data_dir+'QA/emcee_'+mask['maskname'][nexp]+'_'+mask['fname'][nexp]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
   

    m = (slits['collate1d_SN'] > SNmin) & (slits['marz_flag'] < 3)
    nslits = np.sum(m)
    print('{} {} Emcee with {} slits w/SN > {}'.format(mask['maskname'][0],\
                                                                mask['fname'][nexp],nslits,SNmin))
    
    # LOOP OVER EACH SLIT
    for arg in np.arange(0,nslits,1,dtype='int'):

        if (slits['collate1d_SN'][arg] > SNmin) & (slits['marz_flag'][arg] < 3) & \
           (bool(slits['chi2_tfile'][arg].strip())) & (slits['reduce_flag'][arg,nexp] == 1):
            
            # READ STELLAR TEMPLATE 
            pwave,pflux = read_best_template(slits['chi2_tfile'][arg])
            plogwave    = np.exp(pwave)
            
            # PARAMETERS
            losvd_pix = slits['fit_los'][arg,nexp]/ 0.01
            vguess    = slits['chi2_v'][arg]
            wguess    = slits['telluric_w'][arg,nexp]
            if np.abs(wguess) > 40:
                wguess = 0
            

            # READ DATA AND SET VIGNETTING LIMITS
            wave, flux, ivar,sky = dmost_utils.load_spectrum(slits[arg],nexp,hdu)
            wave_lims = dmost_utils.vignetting_limits(slits[arg],nexp,wave)
            wave = wave[wave_lims]
            flux = flux[wave_lims]
            ivar = ivar[wave_lims]
            
            # CONTINUUM POLYNOMIAL SET BY SN LIMITS
            npoly = 5
            if (slits['rSN'][arg][nexp]) > 100:
                npoly=7


            # INITIALIZEWALKERS
            ndim, nwalkers,p0         = initialize_walkers(vguess,wguess)

            # SMOOTH TEMPLATES 
            sm_pflux = scipynd.gaussian_filter1d(pflux,losvd_pix,truncate=3)
            sm_tell  = scipynd.gaussian_filter1d(telluric['flux'],losvd_pix,truncate=3)

            
            #with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v,\
                                        args=(wave, flux, ivar, telluric['wave'],sm_tell, \
                                              sm_pflux,plogwave,npoly,losvd_pix),a=0.5)

            # FIRST BURN-IN AND RUN EMCEE
            t0=time.time()

            pos, prob, state = sampler.run_mcmc(p0, 100)
            sampler.reset()  
            sampler.run_mcmc(p0, 1800)

            t1=time.time()
            print('mcmc run = {:0.3f}'.format(t1-t0))
            
            # Samples to burn
            d = 100
            
            #tau = sampler.get_autocorr_time()
            #print('tau = ',tau)
            theta = [np.mean(sampler.chain[:, d:, i])  for i in [0,1]]
            print(arg,theta)
            
            for ii in [0,1]:
                mcmc = np.percentile(sampler.chain[:,d:, ii], [16, 50, 84])
                if (ii==0):
                    slits['emcee_v'][arg,nexp] = mcmc[1]
                    slits['emcee_v_err16'][arg,nexp] = mcmc[0]
                    slits['emcee_v_err84'][arg,nexp] = mcmc[2]

                if (ii==1):
                    slits['emcee_w'][arg,nexp] = mcmc[1]
                    slits['emcee_w_err16'][arg,nexp] = mcmc[0]
                    slits['emcee_w_err84'][arg,nexp] = mcmc[2]


            # MAKE BEST MODEL 
            model = mk_single_model(theta, wave, flux, ivar, telluric['wave'],sm_tell, \
                                    sm_pflux,plogwave, npoly,losvd_pix)
                     

            # SAVE QUALITY OF FIT INFO
            slits['emcee_f_acc'][arg,nexp]  = np.mean(sampler.acceptance_fraction)
            slits['emcee_lnprob'][arg,nexp] = np.sum((flux - model)**2 * ivar)/(np.size(flux)-3)
        

            
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))

            
            for ii in range(20):
                ax1.plot(sampler.chain[ii,:,0], color="k",linewidth=0.5)
                ax1.axvline(d)
            ax1.set_title('f_acc = {:0.3f}  v = {:0.2f}'.format(np.mean(sampler.acceptance_fraction),slits['emcee_v'][arg,nexp]))


            for ii in range(20):
                ax2.plot(sampler.chain[ii,:,1], color="k",linewidth=0.5)
                ax2.axvline(d)
            ax2.set_title('w = {:0.2f}'.format(slits['emcee_w'][arg,nexp]))

            pdf.savefig()
            plt.close(fig)


            # PLOT SPECTRUM
            plt.figure(figsize=(20,5))
            m = (flux > np.percentile(flux,0.1)) & (flux < np.percentile(flux,99.9))
#            plt.plot(wave,flux,linewidth=0.8)
            plt.plot(wave[m],flux[m],'k',linewidth=0.8)
            plt.plot(wave,model,'r',linewidth=0.8,alpha=0.8,label='model')
            plt.title('SN = {:0.1f}   chi2 = {:0.1f}'.format(slits['rSN'][arg,nexp],\
                                                          slits['emcee_lnprob'][arg,nexp]))
            plt.legend(title='det={}  xpos={}'.format(slits['rdet'][arg,nexp],int(slits['rspat'][arg,nexp])))

            pdf.savefig()
            plt.close(fig)

            # PLOT CORNER
            labels=['v','w']
            samples   = sampler.chain[:, d:, :].reshape((-1, ndim))
            fig = corner.corner(samples, labels=labels,show_titles=True,quantiles=[0.16, 0.5, 0.84])

            pdf.savefig()
            plt.close('all')

    pdf.close()
    plt.close('all')

    return slits
    

######################################################

def run_emcee(data_dir, slits, mask, outfile, clobber=0):
    
       
    # FOR EACH EXPOSURE
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        nslits      = np.size(slits)
        
        # READ TELLURIC
        tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][ii]+'_'+mask['fname'][ii]+'*.fits')
        telluric = Table.read(tfile[0])


        # RUN EMCEE
        slits = emcee_allslits(data_dir, slits, mask, ii, hdu  ,telluric)
        mask['flag_emcee'][ii] = 1
        
#    if write_file:
#        outfile      = data_dir+'/dmost/dmost_mask_'+mask['maskname'][0]+'.fits'
        dmost_create_maskfile.write_dmost(slits,mask,outfile)
        
    return slits, mask


    

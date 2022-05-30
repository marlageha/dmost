#!/usr/bin/env python

import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit

from dmost import dmost_utils




########################################
def fit_syn_continuum_mask(data_wave,data_flux,data_ivar,cmask,synth_flux):

    
    # FIT CONTINUUM -- for weights use 1/sigma
    ivar = data_ivar/synth_flux**2
    p = np.polyfit(data_wave[cmask],data_flux[cmask]/synth_flux[cmask],5,w=np.sqrt(ivar[cmask]))
    fit=np.poly1d(p)
   
    d = data_flux/fit(data_wave)
    cmask2 = (d > np.percentile(d,15)) & (d < np.percentile(d,99))
    p = np.polyfit(data_wave[cmask2],data_flux[cmask2]/synth_flux[cmask2],5,w=np.sqrt(ivar[cmask2]))
    fit=np.poly1d(p)

    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    continuum_syn_flux = synth_flux * fit(data_wave)
    
    return continuum_syn_flux

########################################
def create_masks(data_wave):
    
        
    # USE THIS FOR CONTINUUM FITTING
    cmask1 = (data_wave > 6555) & (data_wave < 6567) 
    cmask2 = (data_wave > 7590) & (data_wave < 7680) 
    cmask3 = (data_wave > 8470) & (data_wave < 8660) 

    cmaski = cmask1 | cmask2 | cmask3
    continuum_mask = np.invert(cmaski)

    return continuum_mask

######################################################
def find_chip_gap_factor(data_wave,data_flux,data_ivar,wave_gap_b,tflux, det,SN, name):

    # SEARCH CORRECTION FACTORS MSC4 gain flucuates more than other chips
    frange = np.arange(0.95,1.05,0.01)
    if (det[0] == 'MSC04'):
        frange = np.arange(0.85,1.15,0.01)


    fbest  = -1. 

    m     = data_wave < wave_gap_b
    cmask = create_masks(data_wave)

    chi2 = []
    for f in frange:
        scale_flux = data_flux.copy()
        scale_ivar = data_ivar.copy()

        scale_flux[m] = f*data_flux[m]
        scale_ivar[m] = data_ivar[m]/f**2


        model = fit_syn_continuum_mask(data_wave,scale_flux, scale_ivar,cmask,tflux)
        tchi2  = np.sum((scale_flux - model)**2 * scale_ivar)/(np.size(scale_flux))

        chi2 = np.append(chi2,tchi2)


    n         = np.argmin(chi2)
    fbest     = frange[n]
    chi2_best = chi2[n]

    # ASSUME BAD FIT IF AT EDGE OF RANGE
    if (fbest == np.min(frange)) |  (fbest == np.max(frange)):
        fbest = 1.

    # ASSUME BAD FIT IF CHI2 HIGH
    #print(chi2_best,fbest)
    if (chi2_best > 15) & (SN < 75):
        fbest = 1.
    if (chi2_best > 250): 
        fbest = 1.



    plot=0
    if (plot == 1):
        plt.figure(figsize=(18,5))
        plt.title('chi2 = {:0.2f}, sn ={:0.2f}, f= {:0.2f}'.format(chi2[n],  SN, fbest))

        plt.plot(frange,chi2)

        plt.figure(figsize=(18,5))
        scale_flux = data_flux.copy()
        scale_ivar = data_ivar.copy()

        m = data_wave < wave_gap_b
        scale_flux[m] = fbest*data_flux[m]
        scale_ivar[m] = data_ivar[m]/f**2

        t = 'corrected f={:0.2f}'.format(fbest)
        plt.plot(data_wave,data_flux,lw=0.5)
        plt.plot(data_wave,scale_flux,lw=0.5,label=t)
        plt.title(name)
        plt.legend()

    return fbest, chi2_best


######################################################

def chip_gap_single_slit(slits, mask, hdu, nexp,telluric,SNmin):
    
    
    # LOOP OVER EACH SLIT
    for arg in np.arange(0,np.size(slits),1,dtype='int'):


        if (slits['collate1d_SN'][arg] > SNmin) & (slits['marz_flag'][arg] < 3)  & (slits['flag_skip_exp'][arg,nexp] != 1):
            
            
            # READ DATA AND SET VIGNETTING LIMITS
            wave, flux, ivar, sky = dmost_utils.load_spectrum(slits[arg],nexp,hdu)
            wave_lims             = dmost_utils.vignetting_limits(slits[arg],nexp,wave)
            wave = wave[wave_lims]
            flux = flux[wave_lims]
            ivar = ivar[wave_lims]

                
            # TRIM WAVELENGTH OF TEMPLATES TO SPEED UP COMPUTATION
            dmin = np.min(wave) - 20
            dmax = np.max(wave) + 20
            mt = (telluric['wave'] > dmin) & (telluric['wave']<dmax)
            

            # SMOOTH TEMPLATES 
            losvd_pix = slits['fit_lsf_corr'][arg,nexp]/ 0.02            
            sm_tell   = scipynd.gaussian_filter1d(telluric['flux'][mt],losvd_pix,truncate=3)
            twave     = telluric['wave'][mt]
            tflux     = np.interp(wave,twave,sm_tell)


            wave_gap_b  = slits['chip_gap_b'][arg,nexp]

            fbest, chi2 = find_chip_gap_factor(wave, flux, ivar,wave_gap_b,tflux,slits['det'][arg],slits['SN'][arg,nexp],slits['slitname'][arg,nexp])

            slits['chip_gap_corr'][arg,nexp] = fbest

    return slits

def chip_gap_single_collate1d(data_dir, slits, mask, telluric,SNmin):


    for ii,obj in enumerate(slits): 

        # FIND TEMPLATES FOR GOOD NON-GALAXY SLITS
        if (obj['marz_flag'] < 3) & (obj['collate1d_SN'] > SNmin):

            jhdu = fits.open(data_dir+'collate1d/'+obj['collate1d_filename'])

            jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(obj,jhdu,chip_gap =0) 
            vexp = 0
            if (obj['flag_skip_exp'][0] == 0):
                m=obj['flag_skip_exp'] != 0
                vexp=m[0]
            wave_lims = dmost_utils.vignetting_limits(obj,vexp,jwave)

            wave = jwave[wave_lims]
            flux = jflux[wave_lims]
            ivar = jivar[wave_lims]


           # TRIM WAVELENGTH OF TEMPLATES TO SPEED UP COMPUTATION
            dmin = np.min(wave) - 20
            dmax = np.max(wave) + 20
            mt = (telluric['wave'] > dmin) & (telluric['wave']<dmax)
            

            # SMOOTH TEMPLATES 
            losvd_pix = np.mean(slits['fit_lsf'][ii,:]/ 0.02)           
            sm_tell   = scipynd.gaussian_filter1d(telluric['flux'][mt],losvd_pix,truncate=3)
            twave     = telluric['wave'][mt]
            tflux     = np.interp(wave,twave,sm_tell)


            wave_gap_b   = np.mean(slits['chip_gap_b'][ii,:])
            fbest, chi2  = find_chip_gap_factor(wave, flux, ivar,wave_gap_b,tflux,obj['det'],obj['collate1d_SN'],obj['slitname'][0])

            slits['chip_gap_corr_collate1d'][ii] = fbest

    return slits
    

######################################################

def run_chip_gap(data_dir, slits, mask, clobber=0):
    '''
    Determine a factor to multiply blue chip to best match red chip flux
    '''   

    SNmin = 8
       
    print('{} Calculate chip gap factor for stellar slits w/SN > {}'.format(mask['maskname'][0],SNmin))
   
    # DETERMINE CHIP GAP FOR INDIVIDUAL EXPOSURES FIRST
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        # READ SPEC1D FILE
        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        nslits      = np.size(slits)
        
        # READ TELLURIC FOR THIS EXPOSURE
        tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][ii]+'_'+mask['fname'][ii]+'*.fits')
        telluric = Table.read(tfile[0])

        # DETERMINE CHIP GAPS
        slits = chip_gap_single_slit(slits, mask, hdu,ii,telluric,SNmin)
        


    # NEXT RUN ON COLLATE1D
    slits = chip_gap_single_collate1d(data_dir,slits, mask, telluric,SNmin)


    mm = (slits['chip_gap_corr'] != 0.0) & (slits['chip_gap_corr'] != 1.0)
    if np.sum(mm) > 0:
        print('{} Chip gap factor median is {:0.2f}'.format(mask['maskname'][0],np.median(slits['chip_gap_corr'][mm])))

        
    return slits, mask


    

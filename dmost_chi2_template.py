#!/usr/bin/env python

import numpy as np
import os, sys

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table
from astropy.io import ascii,fits
import glob
import time

from scipy.optimize import curve_fit
import scipy.ndimage as scipynd

import dmost_utils
DEIMOS_RAW     = os.getenv('DEIMOS_RAW')


########################################################
# Determine best stellar template using coadd1d spectrum
# Use this template on indivudal exposures
# Prepare templates using dmost_pheonix_templates


#######################################################
# Create masks for chi2 evaluation and continuum fitting
# 
def create_chi2_masks(data_wave):
    
    
    # DETERMINE CHI2 NEAR STRONG LINES
    cmask1 = (data_wave > 6200) & (data_wave < 6750) 
    cmask2 = (data_wave > 6950) & (data_wave < 7140) 
    cmask3 = (data_wave > 7350) & (data_wave < 7550) 
    cmask4 = (data_wave > 7800) & (data_wave < 8125) 
    cmask5 = (data_wave > 8170) & (data_wave < 8210) 
    cmask6 = (data_wave > 8350) & (data_wave < 8875) 
    chi2_mask = cmask1 | cmask2 | cmask3 | cmask4 | cmask5 | cmask6


    # USE THIS FOR CONTINUUM FITTING
     # EXCLUDE FOR CONTINUUM FITTING
    cmask1 = (data_wave > 6554) & (data_wave < 6567) 
    cmask2 = (data_wave > 6855) & (data_wave < 6912)
    cmask3 = (data_wave > 7167) & (data_wave < 7320)
    cmask4 = (data_wave > 7590) & (data_wave < 7680) 
    cmask5 = (data_wave > 8160) & (data_wave < 8300)
    cmask6 = (data_wave > 8925) & (data_wave < 9120)

    
    cmaski = cmask1 | cmask2 | cmask3 | cmask4 | cmask5 | cmask6
    continuum_mask = np.invert(cmaski)


    return continuum_mask, chi2_mask


########################################
def fit_continuum(data_wave,data_flux,data_ivar,cmask,synth_flux,npoly):

    
    # FIT CONTINUUM -- for weights use 1/sigma
    ivar = data_ivar/synth_flux**2
    p    = np.polyfit(data_wave[cmask],data_flux[cmask]/synth_flux[cmask],npoly,w=np.sqrt(ivar[cmask]))

    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    model_flux = synth_flux * faster_polyval(p, data_wave)

    return p, model_flux


###############################################
def calc_chi2(data_flux,data_wave,data_ivar,final_fit,nparam=3):

    chi2 = np.sum((data_flux - final_fit)**2 * data_ivar)/(np.size(data_flux)-nparam)
    return chi2


###############################################
def single_stellar_template(file,data_wave,data_flux,data_ivar,losvd_pix,vbest,npoly):
    
    hdu  = fits.open(file)
    data = hdu[1].data
    
    phx_flux    = np.array(data['flux']).flatten()
    phx_logwave = np.array(data['wave']).flatten()
    
    # CREATE MODEL 
    conv_spec = scipynd.gaussian_filter1d(phx_flux,losvd_pix,truncate=3)
    
    # MASK TELLURIC REGIONS
    cmask, chi2_mask = create_chi2_masks(data_wave)

    # Velocity shift star
    shifted_logwave = phx_logwave + vbest/2.997924e5       
    conv_int_flux   = np.interp(data_wave,np.exp(shifted_logwave),conv_spec)


    # FIT CONTINUUM
    p, final_model = fit_continuum(data_wave,data_flux,data_ivar,cmask,conv_int_flux,npoly)

    return final_model


########################################
def faster_polyval(p, x):
    y = np.zeros(x.shape, dtype=float)
    for i, v in enumerate(p):
        y *= x
        y += v
    return y

###############################################
def chi2_single_stellar_template(phx_flux,pwave,data_wave,data_flux,data_ivar,losvd_pix,vrange,npoly):
    

    # CREATE MODEL 
    conv_spec = scipynd.gaussian_filter1d(phx_flux,losvd_pix,truncate=3)

    # MASK TELLURIC REGIONS
    cmask, chi2_mask = create_chi2_masks(data_wave)

    # FIT CONTINUUM OUTSIDE LOOP TO SAVE TIME
    tmp_flux      = np.interp(data_wave,pwave,conv_spec)
    cont_p, tmp = fit_continuum(data_wave,data_flux,data_ivar,cmask,tmp_flux,npoly)

    
    vchi2=[]
    for v in vrange:

        # Velocity shift star
        shifted_logwave = pwave * np.e**(v/2.997924e5)        
        conv_int_flux   = np.interp(data_wave,shifted_logwave,conv_spec)
    

        # FIT CONTINUUM
        final_fit_tmp = conv_int_flux * faster_polyval(cont_p,data_wave)

        # CALCUALTE CHI2
        chi2 = calc_chi2(data_flux[chi2_mask],data_wave[chi2_mask],\
                         data_ivar[chi2_mask], final_fit_tmp[chi2_mask])

        vchi2 = np.append(vchi2,chi2)

        
    # NOW CALCULATE BEST CHI2
    n        = np.argmin(vchi2)
    min_v    = vrange[n]
    min_chi2 = vchi2[n]

    
    return min_v,min_chi2


###############################################
def projected_chi_plot(x,y,z):
    
    unq_a = np.unique(x)
    unq_b = np.unique(y)
    aa,bb,cc = [],[],[]
    
    
    for a in unq_a:
        for b in unq_b:
            m=(x == a) & (y == b)
            if np.sum(m) > 0:
                cc = np.append(cc,np.min(z[m]))
                aa = np.append(aa,a)
                bb = np.append(bb,b)

    return aa,bb,cc

###############################################
def get_stellar_template_files(SN):
    
    if (SN >= 25):
        grid = 'grid1'
  
    if (10 < SN < 25):
        grid = 'grid2'
  
    if (SN <= 10):
        grid = 'grid3'
        
    
    templ = DEIMOS_RAW + '/templates/pheonix/'+grid+'/dmost*'
    pfiles = glob.glob(templ)
    
    
    return pfiles, grid

###############################################
def chi2_best_template(f,data_wave,data_flux,data_ivar,vrange,pdf,plot=0):
    
    best_chi, best_v, best_t       = [], [], []
    best_feh, best_teff, best_logg = [], [], []

    
    # GRAB TEMPLATES FOR SPECIFIC SN
    phx_files, grid = get_stellar_template_files(f['collate1d_SN'])
    
    
    # USE MEAN LINE SPREAD FUNCTION
    losvd_pix =    np.mean(f['fit_lsf'][f['fit_lsf']>0])/ 0.02


    # CONTINUUM POLYNOMIAL SET BY SN LIMITS
    npoly = 5
    if (f['collate1d_SN']) > 100:
        npoly=7

    # LOOP THROUGH ALL TEMPLATES
    for phx_file in phx_files:

        # READ SINGLE SYNTHEIC TEMPLATE
        hdu         = fits.open(phx_file)
        data        = hdu[1].data
        phx_flux    = np.array(data['flux']).flatten()
        phx_logwave = np.array(data['wave']).flatten()


        # TRIM WAVELENGTH OF TEMPLATES TO SPEED UP COMPUTATION
        dmin = np.min(data_wave) - 20
        dmax = np.max(data_wave) + 20
        pwave = np.e**(phx_logwave)
        mp = (pwave > dmin) & (pwave<dmax)

        # RUN ONE STELLAR TEMPLATE
        min_v,min_chi2 = chi2_single_stellar_template(phx_flux[mp],pwave[mp],data_wave,data_flux,\
                                           data_ivar,losvd_pix,vrange,npoly)

        final_file = ''
        if min_chi2 > 0.1:
            best_chi = np.append(best_chi,min_chi2)
            best_v   = np.append(best_v,min_v)
            best_t   = np.append(best_t,phx_file)
            best_feh = np.append(best_feh,data['feh'])
            best_logg= np.append(best_logg,data['logg'])
            best_teff= np.append(best_teff,data['teff'])
 
            n=np.argmin(best_chi)

            # SPLIT FILENAME
            final_file = best_t[n]
            tfile = final_file.split('pheonix/')

            f['chi2_tfile'] = tfile[1]
            f['chi2_tgrid'] = grid
            f['chi2_tchi2'] = best_chi[n]
            f['chi2_v']     = best_v[n]
            f['chi2_teff']  = best_teff[n]
            f['chi2_logg']  = best_logg[n]
            f['chi2_feh']   = best_feh[n]

            
    # PLOT RESULTS
    if (plot==1):
        vmn = np.log(np.min(best_chi))
        vmx = np.log(vmn + np.percentile(best_chi,25))

        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(20,5))
        
        aa,bb,cc = projected_chi_plot(best_teff,best_feh,best_chi)
        ax1.scatter(aa,bb,c=np.log(cc),vmin=vmn,vmax=vmx,cmap='cool')
        ax1.plot(best_teff[n],best_feh[n],'o',mfc='none',mec='k',ms=15)
        ax1.set_xlabel('Teff')
        ax1.set_ylabel('[Fe/H]')
        ax1.set_title('Teff = {:0.1f}'.format(f['chi2_teff']))
        ax1.set_xlim(2400,8100)
        ax1.set_ylim(-4.2,0.2)

        aa,bb,cc = projected_chi_plot(best_teff,best_logg,best_chi)
        ax2.scatter(aa,bb,c=np.log(cc),vmin=vmn,vmax=vmx,cmap='cool')
        ax2.plot(best_teff[n],best_logg[n],'o',mfc='none',mec='k',ms=15)
        ax2.set_xlabel('Teff')
        ax2.set_ylabel('Logg')
        ax2.set_title('Logg = {:0.1f}'.format(f['chi2_logg']))
        ax2.set_xlim(2400,8100)
        ax2.set_ylim(0.5,5.5)

        
        aa,bb,cc = projected_chi_plot(best_feh,best_logg,best_chi)
        ax3.scatter(aa,bb,c=np.log(cc),vmin=vmn,vmax=vmx,cmap='cool')
        ax3.plot(best_feh[n],best_logg[n],'o',mfc='none',mec='k',ms=15)
        ax3.set_xlabel('[Fe/H]')
        ax3.set_ylabel('Logg')
        ax3.set_title('[Fe/H] = {:0.1f}'.format(f['chi2_feh']))
        ax3.set_ylim(-4.2,0.2)
        ax3.set_ylim(0.5,5.5)

        
        # MAKE COLORBAR
        v1 = np.linspace(vmn,vmx, 8, endpoint=True)
        cax, _    = matplotlib.colorbar.make_axes(ax3,ticks=v1)
        normalize = matplotlib.colors.Normalize(vmin = vmn,vmax=vmx)
        cbar      = matplotlib.colorbar.ColorbarBase(cax,norm=normalize,cmap=matplotlib.cm.cool)
        cbar.ax.set_yticklabels(["{:4.1f}".format(np.exp(i)) for i in v1])

        pdf.savefig()
        plt.close(fig)
        

        
        fig,ax = plt.subplots(figsize=(20,5))
        cmask, chi2_mask = create_chi2_masks(data_wave)
        plt.plot(data_wave,data_flux,'k',label='full spectrum',linewidth=0.9)

        plt.plot(data_wave[chi2_mask],data_flux[chi2_mask],'grey',label='fitted region',linewidth=0.8)
        model = single_stellar_template(final_file,data_wave,data_flux,data_ivar,losvd_pix,f['chi2_v'],npoly)
        plt.plot(data_wave,model,'r',label='Model',linewidth=0.8,alpha=0.8)
        plt.title('SN = {:0.1f}   chi2 = {:0.1f}   v = {:0.1f}'.format(f['collate1d_SN'],f['chi2_tchi2'],f['chi2_v']))
        plt.legend(title='det={}  xpos={}'.format(f['rdet'][0],int(f['rspat'][0])))

        pdf.savefig()
        plt.close(fig)

    return final_file, f, pdf



###############################################
###############################################
def run_chi2_templates(data_dir, slits, mask, clobber=0):
    
    
    # QA FILENAME
    file  = data_dir+'QA/chi2_collate1d_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
    
    # BROADCAST
    m = (slits['marz_flag'] < 3) & (slits['collate1d_SN'] > 1.0) 
    print('{} Finding chi2 templates for {} stellar slits w/SN > 1.0'.format(mask['maskname'][0],np.sum(m)))
     
    # V RANGE FOR TEMPLATE FINDER
    vrange = np.arange(-500,500,3)

    for ii,obj in enumerate(slits): 

        # FIND TEMPLATES FOR GOOD NON-GALAXY SLITS
        if (obj['marz_flag'] < 3) & (obj['collate1d_SN'] > 1.0) & (bool(obj['collate1d_filename'].strip())):

            jhdu = fits.open(data_dir+'collate1d/'+obj['collate1d_filename'])

            jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(jhdu) 
            vexp = 0
            if (obj['reduce_flag'][0] == 0):
                m=obj['reduce_flag'] != 0
                vexp=m[0]
            wave_lims = dmost_utils.vignetting_limits(obj,vexp,jwave)

            data_wave = jwave[wave_lims]
            data_flux = jflux[wave_lims]
            data_ivar = jivar[wave_lims]

            tfile,f,pdf = chi2_best_template(obj,data_wave,data_flux,data_ivar,vrange,pdf,plot=1)

            slits['chi2_tfile'][ii] = f['chi2_tfile']
            slits['chi2_tgrid'][ii] = f['chi2_tgrid']
            slits['chi2_tchi2'][ii] = f['chi2_tchi2']
            slits['chi2_v'][ii]     = f['chi2_v']
            slits['chi2_teff'][ii]  = f['chi2_teff']
            slits['chi2_logg'][ii]  = f['chi2_logg']
            slits['chi2_feh'][ii]   = f['chi2_feh']

    pdf.close()
  
    mask['flag_template'][:] = 1
    return slits, mask




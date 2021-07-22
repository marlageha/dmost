#!/usr/bin/env python
import numpy as np
import os,sys

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table
from astropy.io import ascii,fits
import glob

import scipy.ndimage as scipynd
from scipy import ndimage

from scipy.optimize import curve_fit
from astropy import  convolution
from astropy.convolution import Gaussian1DKernel, convolve

import dmost_utils

DEIMOS_RAW     = os.getenv('DEIMOS_RAW')


########################################################
# Determine best synthetic telluric spectrum per exposure
#
#   1.  Chi2 fitting for grid of telluric spectra per slit
#   2.  Determine H2O and O2 values for each slit
#   3.  Determine best values for mask
#


########################################
def fit_syn_continuum_telluric(data_wave,data_flux,data_ivar,cmask,synth_flux):

    
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
def fit_syn_continuum_telluric2(data_wave,data_flux,data_ivar,cmask,synth_flux):

    
    # FIT CONTINUUM -- for weights use 1/sigma
    ivar = data_ivar/synth_flux**2
    p = np.polyfit(data_wave[cmask],data_flux[cmask]/synth_flux[cmask],5,w=np.sqrt(ivar[cmask]))
    fit=np.poly1d(p)
   
    d = data_flux/fit(data_wave)
    cmask2 = (d > np.percentile(d,15)) & (d < np.percentile(d,99))
    p = np.polyfit(data_wave[cmask2],data_flux[cmask2]/synth_flux[cmask2],5,w=np.sqrt(ivar[cmask2]))
    fit=np.poly1d(p)

    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    #continuum_syn_flux = synth_flux * fit(data_wave)

    return fit

########################################
# Create masks for chi2 evaluation and 
# telluric continuum fitting
def create_tell_masks(data_wave):
    
    b = [6855,7167,7580,8160,8925]
    r = [6912,7320,7690,8300,9120]

    data_mask1 = (data_wave > b[0]) & (data_wave < r[0]) 
    data_mask2 = (data_wave > b[1]) & (data_wave < r[1]) 
    data_mask3 = (data_wave > b[2]) & (data_wave < r[2]) 
    data_mask4 = (data_wave > b[3]) & (data_wave < r[3]) 
    data_mask5 = (data_wave > b[4]) & (data_wave < r[4]) 
    chi2_mask = data_mask1 | data_mask2 | data_mask3 | data_mask4 | data_mask5
    
    
    # USE THIS FOR CONTINUUM FITTING
    cmask1 = (data_wave > 6555) & (data_wave < 6567) 
    cmask2 = (data_wave > 7590) & (data_wave < 7680) 
    cmask3 = (data_wave > 8470) & (data_wave < 8660) 

    cmaski = cmask1 | cmask2 | cmask3
    continuum_mask = np.invert(cmaski)

    return continuum_mask, chi2_mask


########################################
def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)

def chi_interp_1d(chi,x,y):
    
    mn       = np.argmin(chi)
    best_chi = chi[mn]
    n1       = y == y[mn]
    
    # FIND X MIN in a 2D array
    
    c = chi[n1]
    yy = y[n1]
    xx = x[n1]
    argchi = np.argsort(c)
    arg = argchi[0:3]
    p_x = np.polyfit(xx[arg],c[arg],2)
    r_x = np.roots(p_x)
    err_x = -99
    r=0
    if (best_chi < 15) & (best_chi > 0):
        err = solve_for_y(p_x, best_chi+0.1)
        err_x = (err[0]-err[1])/2.
        if err_x < 5:
            err_x=10
        r = r_x.real[0]
    
    return r,err_x


def find_chi2_roots(values, chi2):

    z = np.polyfit(values,chi2,2)
    p = np.poly1d(z)

    p2 = np.roots(p)
    min_w    = p2[0].real
    min_chi2 = p(min_w)
    
    return min_w,min_chi2


########################################
# READ TELLURIC MODEL FOR MASK
def get_telluric_model(file):
        
    hdu  = fits.open(file)
    data =  hdu[1].data
    tel_wave = np.array(data['wave']).flatten()
    tel_flux = np.array(data['flux']).flatten()

    return tel_wave, tel_flux


########################################
def generate_single_telluric(tfile,w,data_wave,data_flux,data_ivar,cont_mask,lsp):

    losvd_pix   =  lsp/0.01
    twave,tflux =  get_telluric_model(tfile)

    conv_tell = scipynd.gaussian_filter1d(tflux,losvd_pix)

    shift_wave      = twave + w*0.01
    conv_shift_tell = np.interp(data_wave,shift_wave,conv_tell)

    model = fit_syn_continuum_telluric(data_wave,data_flux,   \
                                   data_ivar,cont_mask,conv_shift_tell)
    
    return model


########################################
def parse_tfile(tfile):

    spl = tfile.split('_')
    h2o = np.float(spl[3])
    o2  = np.float(spl[5])
    return o2,h2o



########################################
def run_single_telluric(twave,conv_tell,data_wave,data_flux,data_ivar,chi_mask,cfit,wshift):

    shift_wave      = twave + wshift*0.01
    conv_shift_tell = np.interp(data_wave,shift_wave,conv_tell)
   
    # FIT CONTINUUM
    final_fit = conv_shift_tell * cfit(data_wave)
 
    # CALCUALTE CHI2
    nparam = 3
    chi2 = np.sum((data_flux[chi_mask] - final_fit[chi_mask])**2 * \
                  data_ivar[chi_mask])/(np.size(data_flux[chi_mask])-nparam)

    return chi2



########################################
def telluric_marginalize_w(file,data_wave,data_flux,\
                           data_ivar,cont_mask,chi_mask,losvd_pix):
    
    # READ SINGLE SYNTHEIC TELLURIC
    twave,tflux = get_telluric_model(file)
    conv_tell   = scipynd.gaussian_filter1d(tflux,losvd_pix,truncate=3)
    
    
    
    
    # SEARCH OVER TELLURIC SHIFT RANGE
    wrange = np.arange(-0.5,0.5,0.025)/0.01
    

    # FIT CONTINUUM OUTSIDE LOOP TO SAVE TIME
    tmp_flux = np.interp(data_wave,twave,conv_tell)
    cont_fit = fit_syn_continuum_telluric2(data_wave,data_flux,data_ivar,cont_mask,tmp_flux)
        
        
    # LOOP OVER TELLURIC SHIFTS
    chi_wshift = [run_single_telluric(twave,conv_tell,\
                            data_wave,data_flux,data_ivar,\
                            chi_mask,cont_fit,wshift) for wshift in wrange]
    
    
    min_w, min_w_chi2 = find_chi2_roots(wrange, chi_wshift)
    
    
    return min_w,min_w_chi2

########################################
# DETERMINE MINIMUM SN REQUIREMENT TO GET 50 SLITS FOR TELLURIC
def telluric_min_SN(good_slits_SN):

    m=(good_slits_SN > 3) & (good_slits_SN < 150)
    ngood = np.sum(m)
    if (ngood < 52):
        min_SN = 3
    else:
        sorted_SN = np.sort(good_slits_SN[m])
        min_SN    = sorted_SN[-50]
        ngood     = np.sum(good_slits_SN > min_SN)
    return min_SN, ngood


########################################
# MAIN PROGRAM 
def run_telluric_allslits(data_dir, slits, mask, nexp, hdu):

    # GET SYNTHETIC TELLURIC SPECTRA
    DEIMOS_RAW = os.getenv('DEIMOS_RAW')
    templ      = DEIMOS_RAW + '/templates/tellurics/telluric*fits'
    tfiles     = glob.glob(templ)

    file  = data_dir+'QA/telluric_slits_'+mask['maskname'][nexp]+'_'+mask['fname'][nexp]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)

    nslits      = np.size(slits)

    # DETERMINE MIN SN TO GET 50 SLITS
    min_SN, ngood = telluric_min_SN(slits['rSN'][:,nexp])

    print('{} {} Telluric with {} slits w/SN > {:0.1f}'.format(mask['maskname'][0],\
                                                                mask['fname'][nexp],ngood,min_SN))
    
    #ncount = 0
    for arg in np.arange(0,nslits,1,dtype='int'):
        if (slits['rSN'][arg,nexp] > min_SN):

            losvd_pix =  slits['fit_los'][arg,nexp]/ 0.01
            wave,flux,ivar,sky = dmost_utils.load_spectrum(slits[arg],nexp,hdu)

            wave_lims = dmost_utils.vignetting_limits(slits[arg],nexp,wave)
            data_wave = wave[wave_lims]
            data_flux = flux[wave_lims]
            data_ivar = ivar[wave_lims]

            # CREATE DATA MASKS
            continuum_mask, chi2_mask = create_tell_masks(data_wave)


            tmp_chi=np.zeros(len(tfiles))
            tmp_h2o=np.zeros(len(tfiles))
            tmp_o2 =np.zeros(len(tfiles))
            tmp_w  =np.zeros(len(tfiles))

            for j,tfile in enumerate(tfiles):

                o2,h2o = parse_tfile(tfile)

                min_w,min_chi2 = telluric_marginalize_w(tfile,data_wave,data_flux,\
                                            data_ivar,continuum_mask, chi2_mask,losvd_pix)

                tmp_chi[j] = min_chi2
                tmp_h2o[j] = h2o
                tmp_o2[j]  = o2
                tmp_w[j]   = min_w

            n=np.argmin(tmp_chi)

            bo2,berr_o2   = chi_interp_1d(tmp_chi,tmp_o2,tmp_h2o)
            bh2o,berr_h2o = chi_interp_1d(tmp_chi,tmp_h2o,tmp_o2)


            # NEEDED TO CATCH EDGE CASES
            if (bh2o > 130) | (bh2o < 1):
                slits['telluric_chi2'][arg,nexp] = 999


            slits['telluric_o2'][arg,nexp]      = bo2
            slits['telluric_h2o'][arg,nexp]     = bh2o
            slits['telluric_o2_err'][arg,nexp]  = berr_o2
            slits['telluric_h2o_err'][arg,nexp] = berr_h2o
            slits['telluric_w'][arg,nexp]       = tmp_w[n]
            slits['telluric_chi2'][arg,nexp]    = tmp_chi[n]


            # GENERATE THE FINAL MODEL
            dir = DEIMOS_RAW + '/templates/tellurics/telluric_0.01A_h2o_'
            final_file = dir+'{:0.0f}_o2_{:0.2f}_.fits'.format(tmp_h2o[n],tmp_o2[n])

            model      = generate_single_telluric(final_file,tmp_w[n], data_wave,data_flux,\
                                      data_ivar,continuum_mask,slits['fit_los'][arg,nexp])

            plt.figure(figsize=(16,5))
            plt.plot(data_wave,data_flux)
            plt.plot(data_wave,model,'k',label='model')
            plt.plot(data_wave[chi2_mask],model[chi2_mask],'r',label='telluric fitted region')

            plt.title('{}   {}  SN={:0.1f}'.format(arg,slits['rspat'][arg,nexp],slits['rSN'][arg,nexp]))
            plt.legend(title='det={}  xpos={}'.format(slits['rdet'][arg,nexp],int(slits['rspat'][arg,nexp])))

            pdf.savefig()


            fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4,figsize=(20,5))
            ax1.plot(tmp_h2o,tmp_chi,'.')
            ax2.plot(tmp_o2,tmp_chi,'.')
            ax3.plot(tmp_w,tmp_chi,'.')

            cmin=np.min(tmp_chi)
            cmax=np.min(tmp_chi)+np.percentile(tmp_chi,25.)

            ax1.set_ylabel('Chi2')
            ax1.set_xlabel('h2o')

            ax2.set_xlabel('o2')
            ax3.set_xlabel('wshift')
            ax1.set_title('{:0.2f}'.format(slits['rSN'][arg,nexp]))

            ax2.set_title('{:0.3e}'.format(bo2))
            ax1.set_title('{:0.3f}'.format(bh2o))


            ax1.axvline(slits['telluric_h2o'][arg,nexp],color='b')
            ax2.axvline(slits['telluric_o2'][arg,nexp],color='b')

            pdf.savefig()
            plt.close('all')
            #ncount = ncount + 1

    pdf.close()
    plt.close('all') 
    return slits

###############################################
# DETERMINE FINAL MASK VALUE AND GENERATE PLOTS
def final_telluric_values(data_dir, slits, mask, nexp, hdu):

    file = data_dir+'QA/telluric_mask_'+mask['maskname'][nexp]+'_'+mask['fname'][nexp]+'.pdf'
    pdf2 = matplotlib.backends.backend_pdf.PdfPages(file)
    fig,((ax1,ax3),(ax2,ax4)) = plt.subplots(2, 2,figsize=(16,9))


    fslits2=slits
    # MASK OUT BAD DATA
    # *****************
    m1= (fslits2['telluric_chi2'][:,nexp] < 25) & (fslits2['telluric_chi2'][:,nexp] > 0)& \
         (fslits2['telluric_h2o'][:,nexp] > 1) & (fslits2['telluric_h2o'][:,nexp] < 120) & \
         (fslits2['telluric_o2'][:,nexp] > 0.6) & (fslits2['telluric_o2'][:,nexp] < 2.5) & \
         (fslits2['telluric_h2o_err'][:,nexp] > 0.01) & (fslits2['telluric_o2_err'][:,nexp] > 0.01) & \
         (np.abs(fslits2['telluric_w'][:,nexp]) < 50) 
    
    # REJECT OUTLIERS (MUST BE A BETTER WAY TO DO THIS!)
    std = np.std(fslits2['telluric_h2o'][m1,nexp])
    md  = np.median(fslits2['telluric_h2o'][m1,nexp])
    std2= np.std(fslits2['telluric_o2'][m1,nexp])
    md2 = np.median(fslits2['telluric_o2'][m1,nexp])

    
    m= (fslits2['telluric_chi2'][:,nexp] < 25) & (fslits2['telluric_chi2'][:,nexp] > 0)& \
         (fslits2['telluric_h2o'][:,nexp] > 1) & (fslits2['telluric_h2o'][:,nexp] < 120) & \
         (fslits2['telluric_o2'][:,nexp] > 0.6) & (fslits2['telluric_o2'][:,nexp] < 2.5) & \
         (fslits2['telluric_h2o_err'][:,nexp] > 0.01) & (fslits2['telluric_o2_err'][:,nexp] > 0.01) & \
         (np.abs(fslits2['telluric_w'][:,nexp]) < 50) &\
         (fslits2['telluric_h2o'][:,nexp] > md-3*std) & (fslits2['telluric_h2o'][:,nexp] < md+3*std) &\
         (fslits2['telluric_o2'][:,nexp] > md2-3*std2) & (fslits2['telluric_o2'][:,nexp] < md2+3*std2)
        
    # FIX PROBLEM WITH SMALL H2O VALUES
    mh20 = fslits2['telluric_h2o'][:,nexp] < 10
    fslits2['telluric_h2o_err'][mh20,nexp] = fslits2['telluric_h2o_err'][mh20,nexp]+10
        
    # DETERMINE FINAL VALUES BASED ON WEIGHTED MEANS
    if np.sum(m) > 1:
        final_h2o = np.average(fslits2['telluric_h2o'][m,nexp],\
                               weights = 1./(fslits2['telluric_h2o_err'][m,nexp])**2)
        final_o2 = np.average(fslits2['telluric_o2'][m,nexp],\
                               weights = 1./(fslits2['telluric_o2_err'][m,nexp])**2)

    else:
        # RELAX WHEN FEW POINTS
        m= (fslits2['telluric_chi2'][:,nexp] > 0)& \
        (fslits2['telluric_h2o'][:,nexp] > 1) & (fslits2['telluric_h2o'][:,nexp] < 120) & \
        (fslits2['telluric_o2'][:,nexp] > 0.6) & (fslits2['telluric_o2'][:,nexp] < 2.5) 
        print('Fewer tellurics2:',np.sum(m))

        if np.sum(m) > 1:
            final_h2o = np.nanmean(fslits2['telluric_h2o'][m,nexp])
            final_o2  = np.nanmean(fslits2['telluric_o2'][m,nexp])
        else:
            print('FAKING IT!')
            final_h2o = 100
            final_o2  = 1.3

    # PLOT H20-- zoom in and all data
    ax1.plot(fslits2['rSN'][:,nexp],fslits2['telluric_h2o'][:,nexp],'.')
    ax1.plot(fslits2['rSN'][m,nexp],fslits2['telluric_h2o'][m,nexp],'r.')
    ax1.errorbar(fslits2['rSN'][m,nexp],fslits2['telluric_h2o'][m,nexp],\
                 yerr=np.abs(fslits2['telluric_h2o_err'][m,nexp]),fmt='.r',ecolor='grey')

    ax1.axhline(final_h2o)

    ax1.set_title('H2O = {:0.3f}'.format(final_h2o))
    ax1.set_ylim(0,115)

    ax3.plot(fslits2['rSN'][:,nexp],fslits2['telluric_h2o'][:,nexp],'.')
    ax3.plot(fslits2['rSN'][m,nexp],fslits2['telluric_h2o'][m,nexp],'r.')
    ax3.errorbar(fslits2['rSN'][m,nexp],fslits2['telluric_h2o'][m,nexp],\
                 yerr=np.abs(fslits2['telluric_h2o_err'][m,nexp]),fmt='.r',ecolor='grey')

    ax3.axhline(final_h2o)
    ax3.set_title('Full data range: H2O = {:0.3f}'.format(final_h2o))


    # PLOT OXYGEN-- zoom in and all data
    ax2.plot(fslits2['rSN'][:,nexp],fslits2['telluric_o2'][:,nexp],'.')
    ax2.errorbar(fslits2['rSN'][m,nexp],fslits2['telluric_o2'][m,nexp],\
                 yerr=np.abs(fslits2['telluric_o2_err'][m,nexp]),fmt='.r',ecolor='grey')

    ax2.set_title('O2 = {:0.3f}'.format(final_o2))
    ax2.axhline(final_o2)
    ax2.set_ylim(0.6,2.4)


    ax4.plot(fslits2['rSN'][:,nexp],fslits2['telluric_o2'][:,nexp],'.')
    ax4.errorbar(fslits2['rSN'][m,nexp],fslits2['telluric_o2'][m,nexp],\
              yerr=np.abs(fslits2['telluric_o2_err'][m,nexp]),fmt='.r',ecolor='grey')

    ax4.set_title('Full data range: O2 = {:0.3f}'.format(final_o2))
    ax4.axhline(final_o2)
    


    ax1.set_xlabel('SN')
    ax2.set_xlabel('SN')
    ax1.set_ylabel('H2O')
    ax2.set_ylabel('O2')

    pdf2.savefig()
    pdf2.close()
    plt.close('all')

    if final_h2o > 100:
        final_h2o = 100
    
    ########################################
    # ROUND TO FINEST GRID 
    round_h2o = 2. * round(final_h2o/2)
    round_o2  = 0.05*round(final_o2/0.05) 
    
    print('{} {}          H2O = {:0.0f}, O2 = {:2.2f}'.format(mask['maskname'][0],\
                                                        mask['fname'][nexp],round_h2o,round_o2))
    
    str = '_h{:0.0f}_o{:2.2f}'.format(round_h2o, round_o2)
    tfile = data_dir+'/dmost/telluric_'+mask['maskname'][nexp]+'_'+mask['fname'][nexp]+str+'.fits'
    tfine = DEIMOS_RAW + '/templates/fine_tellurics/telluric_0.01A_h2o_{}_o2_{:2.2f}_.fits'.format(int(round_h2o),round_o2)
        
    # COPY FINE GRAIN TELLURIC TO DATA DIRECTORY
    os.system('cp '+tfine+' '+tfile)
    
    # UPDATE MASK!
    mask['telluric_h2o'][nexp] = round_h2o
    mask['telluric_o2'][nexp]  = round_o2


    return final_h2o, final_o2, mask


########################################    
########################################
def run_telluric_mask(data_dir, slits, mask, clobber=0):
    
       
    # FOR EACH EXPOSURE
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        nslits      = np.size(slits)

        # PARSE NAMES
        tfile = glob.glob(data_dir+'/dmost/telluric_'+mask['maskname'][ii]+'_'+mask['fname'][ii]+'*.fits')

        # CHECK IF THIS HAS ALREADY BEEN DONE
        if (np.size(tfile) > 0) & (clobber == 0):
            thdu  = fits.open(tfile[0])
            thdr  =  thdu[1].header
            mask['telluric_h2o'][ii] = thdr['h2o']
            mask['telluric_o2'][ii]  = thdr['o2']/1e5
            print('{} {} Telluric already done, adding to header {} {}'.format(mask['maskname'][ii],mask['fname'][ii],\
                                                                               thdr['h2o'],thdr['o2']))


        if (np.size(tfile) == 0) | (clobber == 1):

            # RUN ALL SLITS 
            slits  = run_telluric_allslits(data_dir, slits, mask, ii, hdu)

            # CALCULATE FINAL VALUES AND QA PLOTS
            final_h2o, final_o2, mask = final_telluric_values(data_dir, slits, mask, ii, hdu)

        mask['flag_telluric'][ii] = 1
        
    return slits, mask


    

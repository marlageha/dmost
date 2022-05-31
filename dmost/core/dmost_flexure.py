#!/usr/bin/env python

import numpy as np
import os,sys

from astropy.table import Table,Column
from astropy.io import ascii,fits
from astropy.modeling import models, fitting

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from scipy.optimize import curve_fit
from dmost import dmost_utils

import warnings

########################################################
# Determine linear flexure fits for each slit
#   1.  Determine center+width of sky lines in all slits
#   2.  Fit linear function to residual: mx+b 
#   3.  Fit 2D surface across mask in m, b and LSF
#   4.  Update individual slits using surface
#


#####################################################
# CALCULATE SKY EMISSION LINE 
# DETERMINE GAUSSIAN CENTER+WIDTHS OF ISOLATED LINES
def sky_em_residuals(wave,flux,ivar,sky,plot=0):

     
    dwave, diff, diff_err, los, los_err= [], [], [], [], []
    for line in sky['Wave']:
        wline = [line-5.,line+5.] 
        mw    = (wave > wline[0]) & (wave < wline[1]) & (flux > 0)

        p=[0,0,0,0]
        if np.sum(mw) > 25:
            
            p0 = gauss_guess(wave[mw],flux[mw])
            try:
                p, pcov = curve_fit(gaussian,wave[mw],flux[mw], sigma = 1./np.sqrt(ivar[mw]), p0=p0)
                perr = np.sqrt(np.diag(pcov))
            except:
                p=p0
                p[2] = -99
                perr=p0
            gfit = gaussian(wave[mw],*p)
            d = p[2] - line

            plot=0
            if (plot==1):
                plt.figure(figsize=(8,3)) 
                plt.plot(wave[mw],gfit,'g')
                plt.plot(wave[mw],flux[mw])
                plt.title('{} {:0.2f} diff= {:0.3f}'.format(line,p[3],d))

            if ~np.isfinite(perr[2]):
                perr[2] = 1000.
            dwave    = np.append(dwave,line)
            diff     = np.append(diff,d)
            diff_err = np.append(diff_err,perr[2])
            los      = np.append(los,p[3])
            los_err  = np.append(los_err,perr[3])
            
    m=(diff_err < 0.1) & (diff_err > 0.0)

    return dwave[m],diff[m],diff_err[m],los[m],los_err[m]


#######################################################
# CREATE QUALITY PLOTS
def qa_flexure_plots(plot_dir, nslits, slits, nexp,sky, hdu,mask,fit_slope, fit_b, fit_los, x, y):

  
    pdf2 = matplotlib.backends.backend_pdf.PdfPages(plot_dir+'QA/flex_slits_'+\
                                                    mask['maskname'][nexp]+'_'+mask['fname'][nexp]+'.pdf')
    plt.rcParams.update({'figure.max_open_warning': 0})
    for arg in np.arange(0,nslits,1,dtype='int'):

        if (slits['flag_skip_exp'][arg,nexp] != 1):

            pn = slits['slitname'][arg,nexp]

            # SKY LINES FIRST
            sky_lines, sky_diff,sky_ediff,sky_los,sky_elos = sky_em_residuals(hdu[pn].data['OPT_WAVE'], \
                                                    hdu[pn].data['OPT_COUNTS_SKY'],\
                                                    hdu[pn].data['OPT_COUNTS_IVAR'],sky)
            fitted_line = fit_sky_linear(sky_lines,sky_diff,sky_ediff)


            fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(20,4))
            ax1.plot(sky_lines,sky_diff,'ko',alpha=0.8,label='Sky Emission')
            ax1.errorbar(sky_lines,sky_diff,yerr=sky_ediff,fmt='none',ecolor='k',alpha=0.5)
            #ax1.text(6320,0,'{}'.format(pn),fontsize=11)
            ax1.set_ylim(-0.45,0.45)

            xx=np.arange(6000,9200,1)
            l1 = slits['fit_slope'][arg,nexp]*xx + slits['fit_b'][arg,nexp]
            l2 = fitted_line[1]*xx + fitted_line[0]
            ax1.plot(xx,l1,'-',label='mask fit')
            ax1.plot(xx,l2,'--',label='slit fit')
            ax1.axhline(linewidth=1, color='grey',alpha=0.5)
            ax1.set_ylabel('Wavelength offset (AA)')
            ax1.set_xlabel('Wavelength (AA)')
            ax1.set_xlim(6300,9100)
            ax1.legend(fontsize=12)
            t = 'Sky RMS = {:0.3f} AA, Arc RMS = {:0.3f} AA'.format(slits['rms_sky'][arg,nexp],0.32*slits['rms_arc'][arg])
            ax1.set_title(t)



            ax2.plot(sky_lines,sky_los,'ko',alpha=0.8,label='Sky Emission')
            ax2.errorbar(sky_lines,sky_los,yerr=sky_elos,fmt='none',ecolor='k',alpha=0.5)
            ax2.axhline(slits['fit_lsf'][arg,nexp] ,'-',label='mask value')
            ax2.axhline(np.median(sky_los),'--',label='slit value')

            #lsf_fit = create_lsf_parabola(sky_lines,slits['fit_lsf_p0'][arg,nexp],\
            #                        slits['fit_lsf_p1'][arg,nexp],slits['fit_lsf_p2'][arg,nexp])
            #ax2.plot(sky_lines,lsf_fit)
            ax2.legend(fontsize=12)

            ax2.set_title('Line widths: {}'.format(pn))
            ax2.set_xlabel('Wavelength (AA)')
            ax2.set_ylim(0.3,0.8)
            ax2.set_xlim(6300,9100)

            pdf2.savefig()
    pdf2.close()
    plt.close('all')

    #########################################################################
    # CREATE FULL MASK FITS
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_dir+'QA/flex_mask_'+mask['maskname'][nexp]+\
                                                   '_'+mask['fname'][nexp]+'.pdf')
 
    # SET SIGMA FOR PLOTTING
    t=1.5
    tf=2

    mu  =  np.median(slits['fit_slope'][:,nexp])
    sd  =  np.std(slits['fit_slope'][:,nexp])
    mu2 =  np.median(slits['fit_b'][:,nexp])
    sd2 =  np.std(slits['fit_b'][:,nexp])
    mu3 =  np.median(slits['fit_lsf'][:,nexp])#/mask['lsf_correction'][nexp]
    sd3 =  np.std(slits['fit_lsf'][:,nexp])
    mu4 =  np.median(slits['fit_lsf'][:,nexp])

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(22,5))
 
    ax1.scatter(x,y,c=fit_slope,cmap="cool",vmin = mu-t*sd,vmax=mu+t*sd)
    ax1.set_ylabel('Dec [deg]')
    ax1.set_xlabel('RA [deg]')
    ax1.set_title('Wave MEASURE: line slope')

    ax2.scatter(x,y,c=fit_b,cmap="summer",vmin = mu2-t*sd2,vmax=mu2+t*sd2)
    ax2.set_ylabel('Dec [deg]')
    ax2.set_xlabel('RA [deg]')
    ax2.set_title('Wave MEASURE: line intercept')

    ax3.scatter(x,y,c=fit_los,cmap="cool",vmin = mu3-t*sd3,vmax=mu3+t*sd3)
    ax3.set_ylabel('Dec [deg]')
    ax3.set_xlabel('RA [deg]')
    ax3.set_title('Wave MEASURE: line width')
    cax, _    = matplotlib.colorbar.make_axes(ax3)
    normalize = matplotlib.colors.Normalize(vmin = mu3-t*sd3,vmax=mu3+t*sd3)
    cbar      = matplotlib.colorbar.ColorbarBase(cax,norm=normalize,cmap=matplotlib.cm.cool)

    pdf.savefig()
    
    #######################
    # PLOT MEASURED VALUES
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(22,5))
    xslit = slits['RA']
    yslit = slits['DEC']

    ax1.scatter(xslit,yslit,c=slits['fit_slope'][:,nexp],cmap="cool",vmin = mu-t*sd,vmax=mu+t*sd)

    ax1.set_ylabel('Dec [deg]')
    ax1.set_xlabel('RA [deg]')
    ax1.set_title('Wave fit: line slope')

    ax2.scatter(xslit,yslit,c=slits['fit_b'][:,nexp],cmap="summer",vmin = mu2-t*sd2,vmax=mu2+t*sd2)
    ax2.set_ylabel('Dec [deg]')
    ax2.set_xlabel('RA [deg]')
    ax2.set_title('Wave fit: line intercept')

    ax3.scatter(xslit,yslit,c=slits['fit_lsf'][:,nexp],cmap="cool",vmin = mu4-t*sd3,vmax=mu4+t*sd3)
    ax3.set_ylabel('Dec [deg]')
    ax3.set_xlabel('RA [deg]')
    ax3.set_title('Wave fit: line width  w/seeing corr: {:0.2f}'.format(mask['lsf_correction'][nexp]))
    cax, _ = matplotlib.colorbar.make_axes(ax3)
    normalize = matplotlib.colors.Normalize(vmin = mu4-t*sd3,vmax=mu4+t*sd3)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=matplotlib.cm.cool,norm=normalize)
    
    
    pdf.savefig()
    pdf.close()
    plt.close('all')


########################################
def gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]+p[1]*np.exp(-1.*(x-p[2])**2/(2.*p[3]**2))


########################################
def gauss_guess(x,y):
    norm = np.median(np.percentile(y,50))
    w=np.mean(x)
    N_guess   = np.max(y) - np.min(y)
    sig_guess = 0.5
    p0 = [norm,N_guess,w,sig_guess]

    return p0


#######################################################
def fit_sky_linear(wlines,wdiff,wdiff_err):
    
    z = np.polyfit(wlines,wdiff,w = 1./wdiff_err,deg= 1,cov=False)
    p = np.poly1d(z)

    return p

#######################################################
def fit_lsf_parabola(wlines,wlsf,wlsf_err):
    
    p = np.polyfit(wlines,wlsf,w = 1./wlsf_err,deg= 2,cov=False)
    
    return p

#######################################################
def create_lsf_parabola(wave,p0,p1,p2):
    
    p = [p0,p1,p2]
    pfit = np.poly1d(p)
    lsf_fit = pfit(wave)

    return lsf_fit


#######################################################
def fit_mask_surfaces(fslope, fb, flos, x, y):

    mu  =  np.median(fslope)
    sd  =  np.std(fslope)
    mu2 =  np.median(fb)
    sd2 =  np.std(fb)

    mgood=(np.abs(fslope-mu) < 2.*sd)  & (np.abs(fb-mu2) < 2.*sd2)
    
    # FIT ALL SURFACES WITH 3D POLYNOMIAL
    if (np.sum(mgood) > 10):
        p_init = models.Polynomial2D(degree=3)
        fit_p = fitting.LevMarLSQFitter()
    else:
        p_init = models.Polynomial2D(degree=2)
        fit_p = fitting.LevMarLSQFitter()
        
    # FIT FOR SLOPES, INTERCEPTS, LOS
    pmodel_m   = fit_p(p_init, x[mgood], y[mgood], fslope[mgood])
    pmodel_b   = fit_p(p_init, x[mgood], y[mgood], fb[mgood])
    pmodel_los = fit_p(p_init, x[mgood], y[mgood], flos[mgood])

    
    return pmodel_m,pmodel_b,pmodel_los,np.sum(mgood)


#######################################################
def update_flexure_fit(slits, mask,ii, nslits, hdu, pmodel_m,pmodel_b,pmodel_los,sky):
    '''
    Update Flexure fit and apply:
        -- Determine chip gaps in flexure/air wavelength frame
        -- Apply correction to LSF
    '''

    # UPDATE FITS
    for arg in np.arange(0,nslits,1,dtype='int'):

        slits['fit_slope'][arg,ii] = pmodel_m(slits['RA'][arg],slits['DEC'][arg])
        slits['fit_b'][arg,ii]     = pmodel_b(slits['RA'][arg],slits['DEC'][arg])
        slits['fit_lsf'][arg,ii]   = pmodel_los(slits['RA'][arg],slits['DEC'][arg])


        # APPLY CORRECTION FACTOR DETERMINED FROM ABSORPTION LINE FIT TO SEETING
        # 
        slits['fit_lsf_corr'][arg,ii] = slits['fit_lsf'][arg,ii] * mask['lsf_correction'][ii]



        # CALCULATE RESIDUALS FROM FIT        
        if (slits['serendip'][arg] < 1) & (slits['flag_skip_exp'][arg,ii] != 1) :
            all_wave,all_flux,all_ivar,all_sky = dmost_utils.load_spectrum(slits[arg],ii,hdu,vacuum = 1)
            dwave,diff,diff_err,los,elos       = sky_em_residuals(all_wave,all_sky,all_ivar,sky,plot=0)
            
            m=np.isfinite(diff)
            if np.sum(m) > 0:
                sky_mean = np.average(np.abs(diff[m]), weights = 1./diff_err[m]**2)
                slits['rms_sky'][arg,ii] = sky_mean
                slits['SN'][arg,ii]     = np.median(all_flux*np.sqrt(all_ivar))

    return slits


#############################################
def measure_sky_lines(slits, ii, nslits, hdu,sky):

    fslope, fb, flos = [], [], []
    x,y = [],[]
    
    for arg in np.arange(0,nslits,1,dtype='int'):

    
        if slits['flag_skip_exp'][arg,ii] == 0:
            pn = slits['slitname'][arg,ii]

            try:
                # MEASURE SKY LINE DIFFERENCES
                sky_lines, sky_diff,sky_ediff,sky_los,sky_elos = sky_em_residuals(hdu[pn].data['OPT_WAVE'], \
                                                        hdu[pn].data['OPT_COUNTS_SKY'],\
                                                        hdu[pn].data['OPT_COUNTS_IVAR'],sky)

                # FIT SINGLE SLIT SKY LINES WITH A LINE           
                fitted_line = fit_sky_linear(sky_lines,sky_diff,sky_ediff)


                fslope = np.append(fslope,fitted_line[1])
                fb     = np.append(fb,fitted_line[0])
                flos   = np.append(flos,np.median(sky_los))
                x      = np.append(x,slits['RA'][arg])
                y      = np.append(y,slits['DEC'][arg])
                
                
                # FIT LSF WITH PARABOLA
                lsf_p = fit_lsf_parabola(sky_lines,sky_los,sky_elos)
                slits['fit_lsf_p0'][arg,ii] = lsf_p[0]
                slits['fit_lsf_p1'][arg,ii] = lsf_p[1]
                slits['fit_lsf_p2'][arg,ii] = lsf_p[2]

            except:
                print('  Skipping slit {}'.format(pn))
                slits['flag_skip_exp'][arg] = 1
            
    return slits,fslope, fb, flos, x, y


#######################################################
def run_flexure(data_dir,slits,mask):
    
    # READ SKY LINES -- THESE ARE VACUUM WAVELENGTHS
    DEIMOS_RAW = os.getenv('DEIMOS_RAW')
    sky_file   = DEIMOS_RAW+'Other_data/sky_single_mg.dat'
    sky        = ascii.read(sky_file)

    warnings.simplefilter('ignore')

    
    # FOR EACH EXPOSURE
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        header      = hdu[0].header
        nslits      = np.size(slits)
        
        # INITIAL SKY LINE STUFF
        slits, fslope, fb, flos, x, y = measure_sky_lines(slits, ii,nslits,hdu,sky)
             
            
        # FIT SURFACES
        pmodel_m, pmodel_b,pmodel_los, ngood = fit_mask_surfaces(fslope, fb, flos, x, y)  
        print('{} {} Flexure with {} slits'.format(mask['maskname'][0],\
                                                                mask['fname'][ii],ngood))
        
        # ADD TO TABLE
        slits = update_flexure_fit(slits, mask, ii, nslits, hdu, pmodel_m, pmodel_b,pmodel_los,sky)

  
        # REFIT FOR QA PLOTS
        qa_flexure_plots(data_dir,nslits,slits,ii,sky,hdu,mask,fslope, fb, flos, x, y)

        mask['flag_flexure'][ii] = 1
        
    return slits,mask
  
#####################################################        
#####################################################    
def main(*args,clobber=0):


    slits = sys.argv[1]
    mask = sys.argv[2]
    
    DEIMOS_REDUX     = os.getenv('DEIMOS_REDUX')
    data_dir = DEIMOS_REDUX+'/'+mask+'/'
    
    print('Running flexure on {}'.format(mask))
    fslits = run_flexure(data_dir,slits,mask)
    
    
    
if __name__ == "__main__":
    main()

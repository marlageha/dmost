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

import dmost_utils, dmost_create_maskfile

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit



DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')


######################################################
def mk_EW_plots(pdf, this_slit, nwave,nspec, cat_fit, mg_fit, na_fit):

    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(22,5))

    ax1.plot(nwave,nspec)
    ax1.set_xlim(8484, 8560)
    ax1.plot(nwave,cat_fit,'r')
    ax1.set_title('SN= {:0.2f}'.format(this_slit['collate1d_SN']))


    ax2.set_xlim(8630,8680)
    ax2.plot(nwave,cat_fit,'r',label='_nolegend_')
    ax2.set_title('CaT EW= {:0.2f}  err={:0.2f}'.format(this_slit['cat'],this_slit['cat_err']))


    ax3.plot(nwave,nspec)
    ax3.plot(nwave,mg_fit,'r')
    ax3.set_title(' MgI EW = {:0.2f}  err={:0.2f}'.format(this_slit['mgI'],this_slit['mgI_err']))
    ax3.set_xlim(8802,8810)

    ax4.plot(nwave,nspec)

#    ax4.plot(nawave,naspec)
    ax4.set_xlim(8150,8220)
    ax4.plot(nwave,na_fit,'r')
    ax4.set_title('Na1 EW={:0.2f} err={:0.2f}'.format(this_slit['naI'],this_slit['naI_err']))


    ymax = 1.2
    if this_slit['collate1d_SN'] < 10:
        ymax = 1.5

    ax1.set_ylim(0,ymax)
    ax2.set_ylim(0,ymax)
    ax3.set_ylim(0,ymax)
    ax4.set_ylim(0,ymax)

    pdf.savefig()
    plt.close('all')


    return pdf


    ######################################
def NaI_double_gauss(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    # 8183.3, 8194.8
    return 1.0-p[0]*np.exp(-1.*(x-p[1])**2/(2.*p[2]**2)) \
               -p[3]*np.exp(-1.*(x-(p[1]+11.5))**2/(2.*p[2]**2))

########################################
def NaI_guess(x,y):

    N_guess   = np.max(y) - np.min(y)
    wv_guess  = 8183.3
    sig_guess = 0.8
    p0 = [N_guess,wv_guess,sig_guess,N_guess]

    return p0

########################################
def NaI_normalize(wave,spec,ivar):
    
    # 21AA window centered on 8190AA
    wred  = [8203., 8228.]
    wblue = [8155., 8175.]
    waver = (wred[0] + wred[1])/2.
    waveb = (wblue[0] + wblue[1])/2.

    mred = (wave > wred[0]) & (wave < wred[1])
    mblue = (wave > wblue[0]) & (wave < wblue[1])

    # DETERMINE WEIGHTED MEAN OF BLUE/RED PSEUDO-CONTINUUM BAND
    # DON"T CALCULATE IF DATA DOESN"T EXIST
    
    if (np.sum(mblue) != 0) & (np.sum(mred) != 0): 
         
        m = (spec > np.percentile(spec,20)) & (spec < np.percentile(spec,95))
        sum1 = np.sum(spec[mblue&m] * ivar[mblue&m]**2 )
        sum2 = np.sum( ivar[mblue&m]**2 )
        bcont = sum1 / sum2

        sum1 = np.sum(spec[mred&m] * ivar[mred&m]**2 )
        sum2 = np.sum( ivar[mred&m]**2 )
        rcont = sum1 / sum2


        # DEFINE CONTINUUM LINE BETWEEN RED/BLUE PASSBAND (y=mx+b)
        mline = (rcont - bcont) / (waver - waveb)
        bline = rcont - (mline * waver)
        fit   = (mline * wave) + bline

        
    # NORMALIZE SPECTRUM
    nwave = wave
    nspec = spec/fit
    nivar = ivar*fit**2

    return nwave,nspec,nivar

########################################
def NaI_fit_EW(wvl,spec,ivar):
    
    wline = [8172., 8210.5]

    Na1_EW,Na1_EW_err     = -99, -99
    gfit       = -99

    mw  = (wvl > wline[0]) & (wvl < wline[1]) 
    mzero = spec[mw] == 0
    
    p3=0
    if (np.sum(mzero) < 10):
    
        # GAUSSIAN FIT
        p0 = NaI_guess(wvl[mw],spec[mw])
        errors = 1./np.sqrt(ivar[mw])
        
        try:
            p, pcov = curve_fit(NaI_double_gauss,wvl[mw],spec[mw],sigma = errors,p0=p0)                  
            perr = np.sqrt(np.diag(pcov))
        except:
            p, pcov = p0, None
            perr = p0

        # INTEGRATE PROFILE
        Na1_EW1 = (p[0])*(p[2]*np.sqrt(2.*np.pi))
        Na1_EW2 = (p[3])*(p[2]*np.sqrt(2.*np.pi))
        Na1_EW  = Na1_EW1+Na1_EW2
        
        # CALCUALTE ERROR
        tmp1 = p[0] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[3] * perr[2]* np.sqrt(2*np.pi)
        Na1_EW_err = np.sqrt(tmp1**2 + tmp2**2)

        # CREATE FIT FOR PLOTTING
        gfit = NaI_double_gauss(wvl,*p)

        if (p[2] > 3) | (Na1_EW >10):
            Na1_EW=-99
            Na1_EW_err = -99

        
    return Na1_EW,Na1_EW_err,gfit

########################################
def MgI_gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]-p[1]*np.exp(-1.*(x-8806.8)**2/(2.*p[2]**2))

########################################
def MgI_guess(x,y):
    N_guess   = np.max(y) - np.min(y)
    sig_guess = 0.8
    p0 = [1.,N_guess,sig_guess]

    return p0


########################################
def mgI_EW_fit(wvl,spec,ivar):
    
    # CALCULATE in +/- 5A of MgI line
    # there is a line at 8804.6 (need to deal with this?)
    mgI_line = 8806.8
    wline = [mgI_line-5.,mgI_line+5.] 
    mw    = (wvl > wline[0]) & (wvl < wline[1]) 

    mg1_EW, mg1_EW_err, p3     = -99, -99,  -99
    gfit       = -99*wvl
       
    if np.median(spec[mw]) > 0:


        # GAUSSIAN FIT
        p0 = MgI_guess(wvl[mw],spec[mw])
        errors = 1./np.sqrt(ivar[mw])
        try:
            p, pcov = curve_fit(MgI_gaussian,wvl[mw],spec[mw],sigma = errors,p0=p0)                  
            perr = np.sqrt(np.diag(pcov))
         
            # INTEGRATE PROFILE
            mg1_EW = (p[1])*(p[2]*np.sqrt(2.*np.pi))

            # CALCUALTE ERROR
            tmp1 = p[1] * perr[2]* np.sqrt(2*np.pi)
            tmp2 = p[2] * perr[1]* np.sqrt(2*np.pi)
            mg1_EW_err = np.sqrt(tmp1**2 + tmp2**2)

            # CREATE FIT FOR PLOTTING
            gfit = MgI_gaussian(wvl,*p)
            p3=p[2]

            if (p[2] > 2.0) | (np.abs(mg1_EW) > 10) | (mg1_EW_err == 0.):
                mg1_EW=-99
                mg1_EW_err = -99
        except:
            p, pcov = p0, None
      
    return mg1_EW,mg1_EW_err,gfit, p3


######################################
# DIRECT INTEGRATE MgI LINE OVER 6 AA
# WITH REJECTION
def mgI_EW_direct(wvl,spec,ivar):
    
    # CALCULATE in +/- 3A of MgI line
    mgI_line = 8806.8
    wline = [mgI_line-3.,mgI_line+3.]
    mg1_EW, mg1_EW_err   = -99, -99

    if wvl[0] > 0:

        # CALCULATE DELTA LAMBAs, ASSUME NON-LINEAR BINNING
        dlambda = np.zeros(np.size(wvl))
        for i,item in enumerate(wvl):
            if (i != np.size(wvl)-1):
                dlambda[i] = wvl[i+1]-wvl[i]    
                
        # SET WAVELENGTH RANGE
        mw  = (wvl > wline[0]) & (wvl < wline[1]) 
        dlambda = dlambda[mw]
        spec=spec[mw]
        ivar=ivar[mw]
                
        # SET REJECTION
        mrej = abs(ivar - np.mean(ivar)) < 2.5 * np.std(ivar)

        # DIRECT SUM OF EW 
        mg1_EW = np.sum((1. - spec[mrej]) * dlambda[mrej])
        mg1_EW_ivar = np.sum(ivar[mrej]*dlambda[mrej])
        mg1_EW_err = np.sqrt(1./mg1_EW_ivar)
          
    return mg1_EW,mg1_EW_err


###########################################
def CaT_gauss_plus_lorentzian(x,*p):
#P[0] = CONTINUUM LEVEL
#P[1] = LINE POSITION
#P[2] = GAUSSIAN WIDTH
#P[3] = LORENTZIAN WIDTH
#P[4] = GAUSSIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
#P[5] = LORENTZIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
#P[6] = GAUSSIAN HEIGHT/DEPTH FOR 8498 CAT LINE
#P[7] = GAUSSIAN HEIGHT/DEPTH FOR 8662 CAT LINE
#P[8] = LORENTZIAN HEIGHT/DEPTH FOR 8498 CAT LINE
#P[9] = LORENTZIAN HEIGHT/DEPTH FOR 8662 CAT LINE
 
    gauss = -1.*p[4]*np.exp(-0.5*( (x-p[1])/p[2] )**2) - \
            p[5]*np.exp(-0.5*( (x-p[1]*0.994841)/p[2])**2) - \
            p[6]*np.exp(-0.5*( (x-p[1]*1.01405)/p[2])**2)
    
    lorentz = -1.*p[7]*p[3]/( (x-p[1])**2 + (p[3]/2.)**2 ) - \
              p[8]*p[3]/( (x-p[1]*0.994841)**2 + (p[3]/2.)**2 ) - \
              p[9]*p[3]/( (x-p[1]*1.01405)**2 + (p[3]/2.)**2 )


    return p[0] + gauss + lorentz


########################################
def CaT_GL_guess(x,y):

    Ng, sg  = 0.2, 1.5
    p0 = [1.,8542.09,sg,0.8*sg, Ng,Ng,Ng,Ng,Ng,Ng]

    return p0


########################################
def CaII_EW_fit_all(wvl,spec,ivar):
    
    # CALCULATE IN CENARRO (2001) DEFINED WINDOWS, TABLE 4
    # lines  = [8498.02,8542.09,8662.14]
    wline1 = [8484, 8513]
    wline2 = [8522, 8562]
    wline3 = [8642, 8682]

    CaT, CaT_err, p   = -99, -99, 0
    gfit    = -99*wvl

    # FIT SIMULTANOUSLY IN THE THREE WINDOWS
    mw1  = (wvl > wline1[0]) & (wvl < wline1[1]) 
    mw2  = (wvl > wline2[0]) & (wvl < wline2[1]) 
    mw3  = (wvl > wline3[0]) & (wvl < wline3[1]) 
    mw = mw1 | mw2 | mw3
        

    # GAUSSIAN +  LORENTZ FIT
    p0 = CaT_GL_guess(wvl[mw],spec[mw])
    errors = 1./np.sqrt(ivar[mw])
    try:
        p, pcov = curve_fit(CaT_gauss_plus_lorentzian,wvl[mw],spec[mw],sigma = errors,p0=p0)
    
        perr = np.sqrt(np.diag(pcov))

        # INTEGRATE PROFILE -- GAUSSIAN FIRST
        gint2 = p[4] * p[2] * np.sqrt(2.*np.pi)
        gint1 = p[5] * p[2] * np.sqrt(2.*np.pi)
        gint3 = p[6] * p[2] * np.sqrt(2.*np.pi)
        
         # CALCUALTE GAUSSIAN ERROR
        tmp1 = p[4] * perr[2]* np.sqrt(2.*np.pi)
        tmp2 = p[2] * perr[4]* np.sqrt(2.*np.pi)
        gerr2 = np.sqrt(tmp1**2 + tmp2**2)
        tmp1 = p[5] * perr[2]* np.sqrt(2.*np.pi)
        tmp2 = p[2] * perr[5]* np.sqrt(2.*np.pi)
        gerr1 = np.sqrt(tmp1**2 + tmp2**2)
        tmp1 = p[6] * perr[2]* np.sqrt(2.*np.pi)
        tmp2 = p[2] * perr[6]* np.sqrt(2.*np.pi)
        gerr3 = np.sqrt(tmp1**2 + tmp2**2)

        # INTEGRATE LORENTIAN
        lint1 = 2.*np.pi*p[7]
        lint2 = 2.*np.pi*p[8]
        lint3 = 2.*np.pi*p[9]


        lerr1 = 2.*np.pi*perr[7]
        lerr2 = 2.*np.pi*perr[8]
        lerr3 = 2.*np.pi*perr[9]

        # PUT IT ALL TOGETHER
        #print('CA1 = ',gint1+lint1)
        #print('CA2 = ',gint2+lint2)
        #print('CA3 = ',gint3+lint3)

        CaT = gint1 + gint2 + gint3 + lint1 + lint2 + lint3
        CaT_err = np.sqrt(gerr1**2 + gerr2**2 + gerr3**2 + \
                          lerr1**2 + lerr2**2 + lerr3**2)

        # CREATE FIT FOR PLOTTING
        gfit = CaT_gauss_plus_lorentzian(wvl,*p)
        if (CaT > 14.0) | (p[2] > 6.) | ~(np.isfinite(CaT_err)):
            CaT, CaT_err   = -99, -99
        if (CaT_err > 1.0) & (p[2] > 4.):
            CaT, CaT_err   = -99, -99
        p2=p[2]
    except:
        p, pcov = p0, None

         
        # OMG, WHY 0.85??
    return 0.85*CaT, CaT_err, gfit


########################################
def CaII_normalize(wave,spec,ivar):

    # Cenarro (2001) Table 4
    cont1 = [8474.0,8484.0]
    cont2 = [8563.0,8577.0]
    cont3 = [8619.0,8642.0]
    cont4 = [8700.0,8725.0]
    cont5 = [8776.0,8792.0]
    
    m1 = (wave > cont1[0]) & (wave < cont1[1])
    m2 = (wave > cont2[0]) & (wave < cont2[1])
    m3 = (wave > cont3[0]) & (wave < cont3[1])
    m4 = (wave > cont4[0]) & (wave < cont4[1])
    m5 = (wave > cont5[0]) & (wave < cont5[1])
    m = m1 | m2 | m3 | m4 | m5
    
    fwave = wave[m]
    fspec = spec[m]
    fivar = ivar[m]
    z = np.polyfit(fwave,fspec,1)
    p = np.poly1d(z)
    fit = p(wave)
    
    # NORMALIZE SPECTRUM
    nwave = wave
    nspec = spec/fit
    nivar = ivar*fit**2

    
    return nwave,nspec,nivar



######################################################

def calc_all_EW(data_dir, slits, mask, arg, pdf):


    # READ COADDED DATA
    jhdu = fits.open(data_dir+'/collate1d/'+slits['collate1d_filename'][arg])

    jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(jhdu) 
    wave_lims = dmost_utils.vignetting_limits(slits[arg],0,jwave)

    wvl  = jwave[wave_lims]
    flux = jflux[wave_lims]
    ivar = jivar[wave_lims]

    redshift = slits['dmost_v'][arg] / 299792.
    wave = wvl / (1.0+redshift)  

    #####################             
    # CALCULATE Ca II LINES CONTINUUM
    nwave,nspec,nivar           = CaII_normalize(wave,flux,ivar)
    CaT_EW, CaT_EW_err, CaT_fit = CaII_EW_fit_all(nwave,nspec,nivar)

    slits['cat'][arg]     = CaT_EW
    slits['cat_err'][arg] = CaT_EW_err
    print(CaT_EW)

    ##########################
    # CALCULATE MgI LINES
    MgI_EW_dir,MgI_EW_dir_err          = mgI_EW_direct(nwave,nspec,nivar)        
    MgI_EW,MgI_EW_err, MgI_fit,mg1_p3  = mgI_EW_fit(nwave,nspec,nivar)

    # REPLACE BAD MgI FIT WITH DIRECT
    if (MgI_EW > 1.0) | (mg1_p3 > 1.5):
        if (MgI_EW_dir < 0.3):
            MgI_EW = MgI_EW_dir
            MgI_EW_err = MgI_EW_dir_err*3.

    slits['mgI'][arg]     = MgI_EW
    slits['mgI_err'][arg] = MgI_EW_err
    print(MgI_EW)

    #############################
    # NaI LINES
    nawave,naspec,naivar       = NaI_normalize(wave,flux,ivar)
    NaI_EW,NaI_EW_err, NaI_fit = NaI_fit_EW(nawave,naspec,naivar)
    slits['naI'][arg]     = NaI_EW
    slits['naI_err'][arg] = NaI_EW_err
    print(NaI_EW)

    mk_EW_plots(pdf, slits[arg], nwave, nspec, CaT_fit, MgI_fit, NaI_fit)


    return slits
    

######################################################

def run_coadd_EW(data_dir, slits, mask):
    '''
    CALCUALTE EW USING COADDED SPECTRA
    '''    


    file  = data_dir+'/QA/ew_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)
 

    m = (slits['dmost_v_err'] > 0) & (slits['marz_flag'] < 3)
    print('{} EW estimates for {} slits '.format(mask['maskname'][0],np.sum(m)))
    

    # FOR EACH SLIT
    for ii,slt in enumerate(slits): 


        if (slt['dmost_v_err'] > 0) & (slt['marz_flag'] < 3):
      

            # RUN EMCEE ON COADD
            slits = calc_all_EW(data_dir, slits, mask, ii, pdf)
            

    pdf.close()
    plt.close('all')

        
    return slits, mask


    
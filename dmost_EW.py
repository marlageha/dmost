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
import warnings


import dmost_utils_old, dmost_create_maskfile

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit



DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')


######################################################
def mk_EW_plots(pdf, this_slit, nwave,nspec, nawave, naspec, cat_fit, mg_fit, na_fit,p_na,p_mg):

    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(22,5))

    ax1.plot(nwave,nspec,label = 'chi2 = {:0.2f}'.format(this_slit['cat_chi2']))
    ax1.set_xlim(8484, 8560)
    ax1.plot(nwave,cat_fit,'r')
    ax1.set_title('SN= {:0.1f} v = {:0.1f}'.format(this_slit['collate1d_SN'], this_slit['dmost_v']))
    ax1.legend(loc=3)

    ax2.plot(nwave,nspec)
    ax2.set_xlim(8630,8680)
    tt = 'teff = {}  feh = {}'.format(this_slit['chi2_teff'],this_slit['chi2_feh'])
    ax2.plot(nwave,cat_fit,'r',label=tt)
    ax2.set_title('CaT EW= {:0.2f}  err={:0.2f}'.format(this_slit['cat'],this_slit['cat_err']))
    ax2.legend(loc=3)


    mg_label = 'sig = {:0.2f}, cen = {:0.1f}'.format(p_mg[2],p_mg[3])
    ax3.plot(nwave,nspec,label=mg_label)
    ax3.plot(nwave,mg_fit,'r')
    ax3.set_title(' MgI EW = {:0.2f}  err={:0.2f}'.format(this_slit['mgI'],this_slit['mgI_err']))
    ax3.set_xlim(8802,8811)
    ax3.legend(loc=3)


    na_label = 'sig = {:0.2f}, ratio= {:0.1f}, cen = {:0.1f}'.format(p_na[2],p_na[3],p_na[1])
    ax4.plot(nawave,naspec,label=na_label)
    ax4.set_xlim(8150,8220)
    ax4.plot(nwave,na_fit,'r')
    ax4.set_title('Na1 EW={:0.2f} err={:0.2f}'.format(this_slit['naI'],this_slit['naI_err']))
    ax4.legend(loc=3)

    ymax = 1.2
    if this_slit['collate1d_SN'] < 20:
        ymax = 1.5

    ax1.set_ylim(0,ymax)
    ax2.set_ylim(0,ymax)
    ax3.set_ylim(0,ymax)
    ax4.set_ylim(0,ymax)

    pdf.savefig()
    plt.close('all')


    return pdf

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
    
    fit = 0
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


######################################
def NaI_double_gauss(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Central value                : p[1]
    #   Standard deviatio            : p[2]
    #   Relative height              : p[3]
    # 8183.3, 8194.8
    return p[4]-p[0]*np.exp(-1.*(x-p[1])**2/(2.*p[2]**2)) \
               -p[3]*p[0]*np.exp(-1.*(x-(p[1]+11.54))**2/(2.*p[2]**2))

########################################
def NaI_guess(x,y):

    N_guess   = np.max(y) - np.min(y)
    wv_guess  = 8183.256
    sig_guess = 0.6
    p0 = [N_guess,wv_guess,sig_guess,1.,1.]

    return p0


########################################
def NaI_fit_EW(wvl,spec,ivar,SN):
    
    wline = [8172., 8210.5]

    Na1_EW,Na1_EW_err     = -99, -99
    gfit       = -99*wvl
    p0 = [-99,-99,-99,-99,-99]


    mw  = (wvl > wline[0]) & (wvl < wline[1]) 
    mzero = spec[mw] == 0
    
    if (np.sum(mzero) < 10):
    
        # GAUSSIAN FIT
        p0 = NaI_guess(wvl[mw],spec[mw])
        errors = 1./np.sqrt(ivar[mw])
        
        try:
            p, pcov = curve_fit(NaI_double_gauss,wvl[mw],spec[mw],sigma = errors,p0=p0,\
                   bounds=((0, 8182, 0.4, 1.0,0.95), (2, 8185, 1., 1.6,1.05)))

            perr = np.sqrt(np.diag(pcov))
       
            # INTEGRATE PROFILE
            Na1_EW1 = (p[0])*(p[2]*np.sqrt(2.*np.pi))
            Na1_EW2 = (p[3])*p[0]*(p[2]*np.sqrt(2.*np.pi))
            Na1_EW  = Na1_EW1+Na1_EW2
            
            # CALCUALTE ERROR
            tmp1 = p[0] * perr[2]* np.sqrt(2*np.pi)
            tmp2 = p[2] * perr[0]* np.sqrt(2*np.pi)


            tmp3 = p[3] *p[0] * perr[2]* np.sqrt(2*np.pi)
            tmp4 = p[0] *p[2] * perr[3]* np.sqrt(2*np.pi)
            tmp5 = p[3] *p[2] * perr[0]* np.sqrt(2*np.pi)


            Na1_EW_err = np.sqrt(tmp1**2 + tmp2**2 + tmp3**2 + tmp4**2 + tmp5**2)

            # CREATE FIT FOR PLOTTING
            gfit = NaI_double_gauss(wvl,*p)

            if (Na1_EW > 5) | (Na1_EW_err > 5) | (Na1_EW_err == 0):
                Na1_EW     = -99
                Na1_EW_err = -99
                p=p0
        except:
            p, pcov = p0, None
            perr = p0

        
    return Na1_EW,Na1_EW_err,gfit,p

########################################
def MgI_gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Standard deviation           : p[2]
    return p[0]-p[1]*np.exp(-1.*(x-p[3])**2/(2.*p[2]**2))

########################################
def MgI_guess(x,y):
    N_guess   = np.max(y) - np.min(y)
    sig_guess = 0.6
    p0 = [1.,N_guess,sig_guess, 8806.8]

    return p0


########################################
def mgI_EW_fit(wvl,spec,ivar,SN):
    
    # CALCULATE in +/- 5A of MgI line
    # there is a line at 8804.6 (need to deal with this?)
    mgI_line = 8806.8
    wline = [mgI_line-5.,mgI_line+5.] 
    mw    = (wvl > wline[0]) & (wvl < wline[1]) 

    mg1_EW, mg1_EW_err    = -99, -99
    mgfit       = -99*wvl
    p0 =  [-99,-99,-99,-99]
       

    # GAUSSIAN FIT
    try:
        p0 = MgI_guess(wvl[mw],spec[mw])

        errors = 1./np.sqrt(ivar[mw])
        if SN > 5:
            p, pcov = curve_fit(MgI_gaussian,wvl[mw],spec[mw],sigma = errors,p0=p0, \
                            bounds=((0.5, 0.0, 0.45, 8805.8), (2, 2, 0.9,8807.8)))
        if SN < 5:
            p, pcov = curve_fit(MgI_gaussian,wvl[mw],spec[mw],sigma = errors,p0=p0, \
                            bounds=((0.5, 0.0, 0.45, 8806.2), (2, 2, 0.8,8807.4)))
        perr = np.sqrt(np.diag(pcov))
     
        # INTEGRATE PROFILE
        mg1_EW = (p[1])*(p[2]*np.sqrt(2.*np.pi))

        # CALCUALTE ERROR
        tmp1 = p[1] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[2] * perr[1]* np.sqrt(2*np.pi)
        mg1_EW_err = np.sqrt(tmp1**2 + tmp2**2)

        # CREATE FIT FOR PLOTTING
        mgfit = MgI_gaussian(wvl,*p)
        p3=p[2]

        if (np.abs(mg1_EW) > 10) | (mg1_EW_err == 0.) | (mg1_EW_err > 10.):
            mg1_EW     = -99
            mg1_EW_err = -99
    except:
        p=p0

      
    return mg1_EW,mg1_EW_err,mgfit, p



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
 

    norm  = 1./ (np.sqrt(2*np.pi) * p[2])

    gauss = p[4]*norm*np.exp(-0.5*( (x-p[1])/p[2] )**2) + \
            p[5]*norm*np.exp(-0.5*( (x-p[1]*0.994841)/p[2])**2) + \
            p[6]*norm*np.exp(-0.5*( (x-p[1]*1.01405)/p[2])**2)
    
    norm2   = p[3] / (2.*np.pi)
    lorentz = (p[7]*norm2/( (x-p[1])**2 + (p[3]/2.)**2 )) + \
              (p[8]*norm2/( (x-p[1]*0.994841)**2 + (p[3]/2.)**2 )) + \
              (p[9]*norm2/( (x-p[1]*1.01405)**2 + (p[3]/2.)**2 ))


    return p[0] - gauss - lorentz


########################################
def CaII_EW_fit_GL(wvl,spec,ivar):
    
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
        p, pcov = curve_fit(CaT_gauss_plus_lorentzian,wvl[mw],spec[mw],sigma = errors,p0=p0,\
                                bounds=((0.9, 8540., 0.5,0.5, 0,0,0, 0,0,0), (1.5, 8543.5, 5,3, 3,3,3, 3,3,3)))


        perr = np.sqrt(np.diag(pcov))
        print(p)

        # INTEGRATE PROFILE -- GAUSSIAN FIRST
        gint1 = p[4] 
        gint2 = p[5] 
        gint3 = p[6] 
            
        gerr1 = perr[4]
        gerr2 = perr[5]
        gerr3 = perr[6]


        # INTEGRATE LORENTIAN
        lint1 = p[7] 
        lint2 = p[8] 
        lint3 = p[9] 


        lerr1 = perr[7] 
        lerr2 = perr[8] 
        lerr3 = perr[9] 
        print(gint1,gint2, gint3 ,lint1 ,lint2 ,lint3)
        print(gerr1,gerr1,gerr3,lerr1,lerr2,lerr3)
        print()

        CaT = gint1 + gint2 + gint3 + lint1 + lint2 + lint3

        CaT_err = np.sqrt(gerr1**2 + gerr2**2 + gerr3**2 + \
                          lerr1**2 + lerr2**2 + lerr3**2)

        # CREATE FIT FOR PLOTTING
        gfit = CaT_gauss_plus_lorentzian(wvl,*p)
        chi2 = calc_chi2_ew(wvl,spec,ivar,mw, gfit)

        if (CaT > 14.0) | ~(np.isfinite(CaT_err)):
            CaT, CaT_err   = -99, -99


    except:
        p, pcov = p0, None
        chi2    = -99
         
        # OMG, WHY 0.85??
    return CaT, 0.4*CaT_err, gfit, chi2


###########################################
def CaT_gauss(x,*p):
# USE ON LOW SN SPECTRA
 
    gauss = -1.*p[3]*np.exp(-0.5*( (x-p[1])/p[2] )**2) - \
            p[4]*np.exp(-0.5*( (x-p[1]*0.994841)/p[2])**2) - \
            p[5]*np.exp(-0.5*( (x-p[1]*1.01405)/p[2])**2)
    

    return p[0] + gauss


########################################
def CaII_EW_fit_gauss(wvl,spec,ivar):
    
    # CALCULATE IN CENARRO (2001) DEFINED WINDOWS, TABLE 4
    # lines  = [8498.02,8542.09,8662.14]
    wline1 = [8484, 8513]
    wline2 = [8522, 8562]
    wline3 = [8642, 8682]


    CaT, CaT_err, p, chi2   = -99, -99, 0, -99
    gfit    = -99*wvl

    if np.mean(spec) > 0:

        # FIT SIMULTANOUSLY IN THE THREE WINDOWS
        mw1  = (wvl > wline1[0]) & (wvl < wline1[1]) 
        mw2  = (wvl > wline2[0]) & (wvl < wline2[1]) 
        mw3  = (wvl > wline3[0]) & (wvl < wline3[1]) 
        mw = mw1 | mw2 | mw3
        

        # GAUSSIAN GUESS
        sg=0.3
        p0 = [1.,8542.09,0.8,sg,sg,0.8*sg]

        errors = 1./np.sqrt(ivar[mw])
        

        try:
            p, pcov = curve_fit(CaT_gauss,wvl[mw],spec[mw],sigma = errors,p0=p0,\
                            bounds=((0.5, 8540., 0.5, 0,0,0), (2, 8543.5, 1.5,2,2,2)))
        except:
            p, pcov = p0, None
            return CaT, CaT_err, gfit, chi2

        perr = np.sqrt(np.diag(pcov))

          
        # INTEGRATE PROFILE -- GAUSSIAN FIRST
        gint1 = p[3] * p[2] * np.sqrt(2.*np.pi)
        gint2 = p[4] * p[2] * np.sqrt(2.*np.pi)
        gint3 = p[5] * p[2] * np.sqrt(2.*np.pi)

         # CALCUALTE GAUSSIAN ERROR
        tmp1 = p[4] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[2] * perr[4]* np.sqrt(2*np.pi)
        gerr1 = np.sqrt(tmp1**2 + tmp2**2)
        tmp1 = p[5] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[2] * perr[5]* np.sqrt(2*np.pi)
        gerr2 = np.sqrt(tmp1**2 + tmp2**2)
        tmp1 = p[3] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[2] * perr[3]* np.sqrt(2*np.pi)
        gerr3 = np.sqrt(tmp1**2 + tmp2**2)

        # PUT IT ALL TOGETHER
        CaT = gint1 + gint2 + gint3
        CaT_err = np.sqrt(gerr1**2 + gerr2**2 + gerr3**2)

        # CREATE FIT FOR PLOTTING
        gfit = CaT_gauss(wvl,*p)
        chi2 = calc_chi2_ew(wvl,spec,ivar,mw, gfit)
        

    return CaT, CaT_err, gfit, chi2


########################################
def CaT_GL_guess(x,y):

    Ng, sg  = 0.2, 1.5
    p0 = [1.,8542.09,sg,0.8*sg, Ng,Ng,Ng,Ng,Ng,Ng]

    return p0
def calc_chi2_ew(wave,spec,ivar,mwindow, fit):

    model = fit[mwindow]
    data  = spec[mwindow]
    ivar  = ivar[mwindow]

    chi2 = np.sum((data - model)**2 * ivar)/np.size(data)

    return chi2



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

    warnings.simplefilter('ignore')#, OptimizeWarning)


    # READ COADDED DATA
    jhdu = fits.open(data_dir+'/collate1d/'+slits['collate1d_filename'][arg])

    jwave,jflux,jivar, SN = dmost_utils_old.load_coadd_collate1d(slits[arg],jhdu) 
    wave_lims = dmost_utils_old.vignetting_limits(slits[arg],0,jwave)

    wvl  = jwave[wave_lims]
    flux = jflux[wave_lims]
    ivar = jivar[wave_lims]

    redshift = slits['dmost_v'][arg] / 299792.
    wave     = wvl / (1.0+redshift)  

    wlims = (wvl > 8100) & (wvl < 8700)
    if (np.sum(flux[wlims] > 0) > 1200):

        
        #####################             
        # CALCULATE Ca II LINES CONTINUUM
        nwave,nspec,nivar                   = CaII_normalize(wave,flux,ivar)
        CaT_EW, CaT_EW_err, CaT_fit, CaT_chi2 = CaII_EW_fit_GL(nwave,nspec,nivar)

        if (CaT_EW_err == -99) | (slits['collate1d_SN'][arg] < 20) |  (CaT_EW < 0):
            CaT_EW, CaT_EW_err, CaT_fit, CaT_chi2 = CaII_EW_fit_gauss(nwave,nspec,nivar)


        slits['cat'][arg]      = CaT_EW
        slits['cat_err'][arg]  = CaT_EW_err
        slits['cat_chi2'][arg] = CaT_chi2

        ##########################
        # CALCULATE MgI LINES
        MgI_EW,MgI_EW_err, MgI_fit,p_mg  = mgI_EW_fit(nwave,nspec,nivar,SN)

        slits['mgI'][arg]     = MgI_EW
        slits['mgI_err'][arg] = MgI_EW_err

        #############################
        # NaI LINES
        nawave,naspec,naivar            = NaI_normalize(wave,flux,ivar)
        NaI_EW,NaI_EW_err, NaI_fit,p_na = NaI_fit_EW(nawave,naspec,naivar,SN)
        slits['naI'][arg]     = NaI_EW
        slits['naI_err'][arg] = NaI_EW_err

        mk_EW_plots(pdf, slits[arg], nwave, nspec, nawave, naspec, CaT_fit, MgI_fit, NaI_fit,p_na,p_mg)


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
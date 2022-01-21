import numpy as np
import os

from astropy.table import Table
from astropy.io import ascii,fits

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astroplan import  Observer


import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt



######################################################
# HELIOCENTRIC CORRECTION FOR KECK
# ADD THIS CORRECTION TO MEASURED VELOCITY
######################################################
def deimos_helio(mjd,ra,dec):
    '''
    Heliocentric velocity correction for Keck 
    
    Parameters
    ----------
    mjd: float
        mjd date of observatiom
    ra: float
        right ascension of observation
    dec: float
        declination of observation
    
    Returns
    -------
    vhelio
        heliocentric velocity correction
        This should be ADDED to the measured velocity
    '''
    i=0
    t = Time(mjd,format='mjd')
    r = np.median(ra)
    d = np.median(dec)
    sc = SkyCoord(r,d, unit=(u.deg,u.deg))

    keck = Observer.at_site('keck')
    keck = EarthLocation.of_site('Keck Observatory')
    heliocorr = sc.radial_velocity_correction('heliocentric',obstime=t, location=keck)

    vhelio = heliocorr.to(u.km/u.s) * (u.s/u.km)

    return vhelio


#######################################################
# CALC SN
#######################################################
def calc_rb_SN(r1,b1, hdu):

    # TRY SPECTRUM AND CALCULATE SN
    try:
        rSN = np.median(hdu[r1].data['OPT_COUNTS'] * np.sqrt(hdu[r1].data['OPT_COUNTS_IVAR']))
        bSN = np.median(hdu[b1].data['OPT_COUNTS'] * np.sqrt(hdu[b1].data['OPT_COUNTS_IVAR']))
        aa=hdu[b1].data['OPT_WAVE'] * hdu[r1].data['OPT_WAVE']        
    except:
        rSN=0
        bSN=0

    return rSN, bSN                




#############################################
# NAME TO READ SPEC1d FILES
def get_slit_name(slits,i):
    
    rname = 'SPAT{:04d}-SLIT{:04d}-DET{:02d}'.format(slits['rspat'][i],slits['rslit'][i],slits['rdet'][i])
    bname = 'SPAT{:04d}-SLIT{:04d}-DET{:02d}'.format(slits['bspat'][i],slits['bslit'][i],slits['bdet'][i])
    
    return rname, bname

##################################################
# GET CHIP GAP WAVELENGTHS
def get_chip_gaps(slits,arg,nexp,hdu,r,b):
    

    try:
        ccd_b = hdu[b].data['OPT_WAVE'][-1]
        ccd_r = hdu[r].data['OPT_WAVE'][0]
    except:
        ccd_b=-1
        ccd_r=-1
        
    slits['ccd_gap_b'][arg,nexp] = ccd_b
    slits['ccd_gap_r'][arg,nexp] = ccd_r

    return slits



#######################################################
#  LOAD A SPECTRUM
######################################################
def load_spectrum(slits,nexp,hdu,vacuum=0,vignetted = 0):


    r,b = get_slit_name(slits,nexp)

    try:
        # READ IN DATA FROM SPEC1D, TRIM INNER ENDS
        tmp_wave = np.concatenate((hdu[b].data['OPT_WAVE'][:-3],hdu[r].data['OPT_WAVE'][3:]),axis=None)
        all_flux = np.concatenate((hdu[b].data['OPT_COUNTS'][:-3],hdu[r].data['OPT_COUNTS'][3:]),axis=None)
        all_sky = np.concatenate((hdu[b].data['OPT_COUNTS_SKY'][:-3],hdu[r].data['OPT_COUNTS_SKY'][3:]),axis=None)
        all_ivar = np.concatenate((hdu[b].data['OPT_COUNTS_IVAR'][:-3],hdu[r].data['OPT_COUNTS_IVAR'][3:]),axis=None)

        fitwave  = slits['fit_slope'][nexp]*tmp_wave + slits['fit_b'][nexp]
        vwave = tmp_wave - fitwave

        # CONVERT PYPEIT OUTPUT WAVELENGTHS FROM VACUUM TO AIR
        all_wave = vwave
        if (vacuum == 0):
            all_wave = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)


        # TRIM ENDS
        all_wave=all_wave[5:-15]
        all_flux=all_flux[5:-15]
        all_ivar=all_ivar[5:-15]
        all_sky=all_sky[5:-15]

        # REMOVE CRAZY  VALUES
        sn = all_flux*np.sqrt(all_ivar)
        cmask = (sn > np.percentile(sn,0.1)) & (sn < np.percentile(sn,99.9))

        m=np.median(sn[cmask])
        s=np.std(sn[cmask])
        mm = (sn > 15.*s + m) | (sn < m-20.*s)
        all_flux[mm] = np.median(all_flux)
        all_ivar[mm] = 1e-6
        if (np.sum(mm) > 50):
            print('  Removing more than 50 pixels of data')
        
        
       
    except:
        print('  No data!')
        n=8172
        all_wave=np.zeros(n)
        all_flux=np.zeros(n)
        all_ivar=np.zeros(n)
        all_sky=np.zeros(n)

    return all_wave,all_flux,all_ivar,all_sky

##################################################
# LOAD COLLATE1D SPECTRUM
def load_coadd_collate1d(hdu,vacuum=0,vignetted = 0):

    SN = 0
    data = hdu[1].data
    try:

        tmp_wave = data['wave']
        all_flux = data['flux']
        all_ivar = data['ivar']

        # NO FLEXURE CORRECTION HAPPENING FOR COADDS, USE WITH CAUTION
        vwave = tmp_wave #- fitwave

        # CONVERT PYPEIT OUTPUT WAVELENGTHS FROM VACUUM TO AIR
        all_wave = vwave
        if (vacuum == 0):
            all_wave = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)


        # TRIM ENDS
        all_wave=all_wave[5:-15]
        all_flux=all_flux[5:-15]
        all_ivar=all_ivar[5:-15]

        # REMOVE CRAZY  VALUES
        sn = all_flux*np.sqrt(all_ivar)
        cmask = (sn > np.percentile(sn,0.1)) & (sn < np.percentile(sn,99.9))

        m=np.median(sn[cmask])
        s=np.std(sn[cmask])
        mm = (sn > 25.*s + m) | (sn < m-20.*s)
        all_flux[mm] = np.median(all_flux)
        all_ivar[mm] = 1e-6
        if (np.sum(mm) > 50):
            print('  Removing more than 50 pixels of data')
        
        SN = np.median(all_flux*np.sqrt(all_ivar))


    except:
        print('no data!')
        n=8172
        all_wave=np.zeros(n)
        all_flux=np.zeros(n)
        all_ivar=np.zeros(n)
        all_sky=np.zeros(n)

    return all_wave,all_flux,all_ivar, SN


def correct_chip_gap(fcorr,bwave_gap,wave,flux,ivar):

    if (fcorr != -1):
        m = wave < bwave_gap
        flux[m] = fcorr * flux[m]
        ivar[m] = ivar[m]/fcorr**2

    return flux, ivar


####################################################
def vignetting_limits(slits,nexp,wave):
    
    # DEFAULT SET TO FULL WAVELENGTH RANGE
    vwave_min = np.min(wave)
    vwave_max = np.max(wave)


    
    # FOR DETECTOR 1+5, APPLY VIGNETTING FIT
    xpos = slits['rspat'][nexp]
    if (slits['rdet'][nexp] == 5) & (xpos < 1350):
        p5 = [ 1.05254130e-04, -4.26541379e-01 , 4.20569201e+02] 
        pfit5 = np.poly1d(p5)
        wlim5 = pfit5(xpos) 
        fdg = 50  # fudge for blue side, needed more
        if (xpos < 500):
            fdg=100
        vwave_min = np.min(wave) + wlim5  + fdg 
        vwave_max = np.max(wave) - wlim5

        vwave_min,vwave_max = arc_rms_limits(slits, nexp, vwave_min,vwave_max, wlim5)



    # FOR DETECTOR 4+8, APPLY VIGNETTING FIT
    if (slits['rdet'][nexp] == 8) & (xpos > 700):
        p8=[ 1.37190021e-04, -3.30255017e-02, -6.38328134e+00]  
        pfit8 = np.poly1d(p8)
        wlim8 = pfit8(xpos)
    
        vwave_min = np.min(wave) + wlim8
        vwave_max = np.max(wave) - wlim8
            
        vwave_min,vwave_max = arc_rms_limits(slits, nexp, vwave_min,vwave_max,wlim8)


    # RETURN NON-VIGNETTED WAVELENGTH REGION
    wave_lims = (wave > vwave_min) & (wave < vwave_max)
    

    return wave_lims


def arc_rms_limits(slits,nexp,vwmin,vwmax,wlim):


    # IF RED ARC SOLUTION BAD, REMOVE RED CHIP
    if (slits['rms_arc_r'][nexp] > 0.2):
        vwmax = slits['ccd_gap_b'][nexp] 
        if (vwmin < vwmax - 4096*0.32):
            vwmin = vwmax - 4096*0.32  + wlim


   # IF BLUE ARC SOLUTION BAD, REMOVE BLUE CHIP
    if (slits['rms_arc_b'][nexp] > 0.175):
        vwmin = slits['ccd_gap_r'][nexp] 
        if (vwmax > vwmin + 4096*0.31):  
            vwmax = vwmin + 4096*0.31

    return vwmin,vwmax

####################################################
def read_dmost(outfile):
    
    hdu   = fits.open(outfile)
    mask  = Table(hdu[1].data)
    slits = Table(hdu[2].data)
    hdu.close()
    
    return slits, mask


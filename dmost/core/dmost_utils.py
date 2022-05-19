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
def calc_SN(pn, hdu):

    # TRY SPECTRUM AND CALCULATE SN
    try:
        SN = np.median(hdu[pn].data['OPT_COUNTS'] * np.sqrt(hdu[pn].data['OPT_COUNTS_IVAR']))
    except:
        SN=0
        
    return SN

#######################################################
def find_chip_gap(flux):

    mflux =0
    pix_min,pix_max = int(flux.size/2), int(flux.size/2)
    while mflux == 0:
        pix_min = pix_min-1
        mflux   = flux[pix_min]
    last_zero_pix_min = pix_min +1

    mflux =0
    while mflux == 0:
        pix_max = pix_max+1
        mflux   = flux[pix_max]
    last_zero_pix_max = pix_max -1

    pix_min_final = last_zero_pix_min -4
    pix_max_final = last_zero_pix_max +4

    return pix_min_final,pix_max_final


#######################################################
def correct_chip_gap(fcorr,bwave_gap,wave,flux,ivar):

    # CORRECTION IS APPLIED TO BLUE CHIP
    if (fcorr != -1):
        m = wave < bwave_gap
        flux[m] = fcorr * flux[m]
        ivar[m] = ivar[m]/fcorr**2

    return flux, ivar


#######################################################
#  LOAD A SPECTRUM
######################################################
def load_spectrum(slit,nexp,hdu,vacuum=0,vignetted = 0,fix_flux = 1):


    pn = slit['slitname'][nexp]

    try:
        # READ IN DATA FROM SPEC1D, TRIM INNER ENDS
        tmp_wave = hdu[pn].data['OPT_WAVE'][10:-10]
        all_flux = hdu[pn].data['OPT_COUNTS'][10:-10]
        all_sky  = hdu[pn].data['OPT_COUNTS_SKY'][10:-10]
        all_ivar = hdu[pn].data['OPT_COUNTS_IVAR'][10:-10]

        fitwave  = slit['fit_slope'][nexp]*tmp_wave + slit['fit_b'][nexp]
        vwave = tmp_wave - fitwave

        # CONVERT PYPEIT OUTPUT WAVELENGTHS FROM VACUUM TO AIR
        all_wave = vwave
        if (vacuum == 0):
            all_wave = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)



        # FIND CHIP GAP
        pix_min, pix_max = find_chip_gap(all_flux)        



        # REMOVE CRAZY  VALUES
        sn = all_flux * np.sqrt(all_ivar)
        m1 = (all_wave > np.min(all_wave)+100) & (all_wave < np.max(all_wave)-100)
        m2 = (all_wave > 7580) & (all_wave < 7700)
        m3 = all_ivar != 0.0
        m4 = (sn > np.percentile(sn,0.1)) & (sn < np.percentile(sn,99.9))


        m=np.median(sn[m1&~m2&m3&m4])
        s=np.std(sn[m1&~m2&m3&m4])
        mm = (sn > 10.*s + m) | (sn < m-15.*s)

        mask = mm | (sn < -10)

        if (fix_flux == 1):
            all_flux[mask] = np.nanmedian(all_flux)
            all_ivar[mask] = 0


          # FIX CHIP GAP
            med1 = np.nanmedian(all_flux[pix_min-50:pix_min])
            med2 = np.nanmedian(all_flux[pix_max:pix_max+50])
            all_flux[pix_min:pix_max] = np.nanmedian(all_flux)
            all_ivar[pix_min:pix_max] = 0.

        if (np.sum(mm) > 50):
            print('  Removing more than 50 pixels of data: {} {}'.format(nexp,pn))
            
        
       
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
def load_coadd_collate1d(slit,hdu,vacuum=0,vignetted = 0,flexure=1,chip_gap =1):

    SN = 0
    data = hdu[1].data
    try:

        tmp_wave = data['wave']
        all_flux = data['flux']
        all_ivar = data['ivar']

        # AVERAGE FLEXURE -- USE WITH CAUTION
        vwave = tmp_wave 
        if (flexure == 1):
            mfit = np.median(slit['fit_slope'])
            bfit = np.median(slit['fit_b'])
            fitwave  = mfit*tmp_wave + bfit
            vwave    = tmp_wave - fitwave



        # CONVERT PYPEIT OUTPUT WAVELENGTHS FROM VACUUM TO AIR
        all_wave = vwave
        if (vacuum == 0):
            all_wave = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)


        # TRIM CHIP GAPS (NEEDS FLEXURE TO RUN)
        if (chip_gap == 1):
            rpix = np.median(slit['chip_gap_r'])
            bpix = np.median(slit['chip_gap_b'])
            mr = np.abs(all_wave-rpix) < 1
            all_wave = np.delete(all_wave,mr)
            all_flux = np.delete(all_flux,mr) 
            all_ivar = np.delete(all_ivar,mr) 

            mb = np.abs(all_wave-bpix) < 1
            all_wave = np.delete(all_wave,mb)
            all_flux = np.delete(all_flux,mb) 
            all_ivar = np.delete(all_ivar,mb) 


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
        #all_flux[mm] = np.median(all_flux)
        #all_ivar[mm] = 1e-6
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



####################################################
def vignetting_limits(slit,nexp,wave):
    
    # DEFAULT SET TO FULL WAVELENGTH RANGE
    vwave_min = np.min(wave)
    vwave_max = np.max(wave)

    xpos = slit['spat_pixpos'][nexp]
    det  = slit['det'][nexp]
    
    # FOR DETECTOR 1+5, APPLY VIGNETTING FIT
    if (det == 'MSC01') & (xpos < 1350):
        p5 = [ 1.05254130e-04, -4.26541379e-01 , 4.20569201e+02] 
        pfit5 = np.poly1d(p5)
        wlim5 = pfit5(xpos) 
        fdg = 50  # fudge for blue side, needed more
        if (xpos < 500):
            fdg=100
        vwave_min = np.min(wave) + wlim5  + fdg 
        vwave_max = np.max(wave) - wlim5



    # FOR DETECTOR 4+8, APPLY VIGNETTING FIT
    if (det == 'MSC04') & (xpos > 700):
        p8=[ 1.37190021e-04, -3.30255017e-02, -6.38328134e+00]  
        pfit8 = np.poly1d(p8)
        wlim8 = pfit8(xpos)
    
        vwave_min = np.min(wave) + wlim8
        vwave_max = np.max(wave) - wlim8
            


    # RETURN NON-VIGNETTED WAVELENGTH REGION
    wave_lims = (wave > vwave_min) & (wave < vwave_max)
    

    return wave_lims

####################################################
def read_dmost(outfile):
    
    hdu   = fits.open(outfile)
    mask  = Table(hdu[1].data)
    slits = Table(hdu[2].data)
    hdu.close()
    
    return slits, mask


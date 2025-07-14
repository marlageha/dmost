#!/usr/bin/env python

import numpy as np
import os,sys
    
from astropy.coordinates import SkyCoord


from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits

import sfdmap
from scipy.stats import truncnorm


###########################################
def get_ebv(allspec):

    DEIMOS_RAW  = os.getenv('DEIMOS_RAW')

    sdf_ext = sfdmap.SFDMap(DEIMOS_RAW+'SFDmaps/')
    EBV     = sdf_ext.ebv(allspec['RA'],allspec['DEC'])

    allspec['EBV'] = EBV

    #https://www.legacysurvey.org/dr10/catalogs/#galactic-extinction-coefficients
    # table 6:   https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103/pdf

    # THESE ARE WHAT DELVE AND DES ARE USING
    #   Ar = 2.140 * allspec['EBV']
    #   Ag = 3.185 * allspec['EBV']
    #
    # OLD SDSS VALUES
    #Ar = 2.751 * allspec['EBV']
    #Ag = 3.793 * allspec['EBV']

    # THESE ARE USED in ls_dr10 and decaps, but its already baked in
    Ar = 2.165 * allspec['EBV']
    Ag = 3.214 * allspec['EBV']




    return allspec, Ar, Ag
    

###########################################
def calc_MV_star(allspec,obj):
    
    # DISTANCE MODULUS AND REDDENING
    dmod = 5.*np.log10(obj['Dist_kpc']*1e3) - 5.


    # SDSS Jordi 2006
    # V = allspec['gmag_o'] - 0.565*gr_o -0.016

    # Abbott et al 
    # https://iopscience.iop.org/article/10.3847/1538-4365/ac00b3
    # Appendix A  - piecewise transformation

    gr_o =  allspec['gmag_o'] - allspec['rmag_o'] 
    V    = np.zeros(np.size(gr_o))

    m1    = gr_o <= 0.2
    V[m1] =  allspec['gmag_o'][m1] - 0.465*gr_o[m1] -0.02

    m2    = (gr_o > 0.2) & (gr_o <= 0.7)
    V[m2] =  allspec['gmag_o'][m2] - 0.496*gr_o[m2] -0.015

    m3    = gr_o > 0.7
    V[m3] = allspec['gmag_o'][m3] - 0.445*gr_o[m3] -0.062

    # Only update stars with matched photometry
    m = allspec['rmag_o'] > 0
    allspec['MV_o'][m] = V[m] - dmod
    
    return allspec

###########################################
def calc_rproj(allspec,obj):

    sc_gal = SkyCoord(obj['RA'],obj['Dec'], unit=(u.deg, u.deg),distance = obj['Dist_kpc']*u.kpc)


    # CALCULATE STAR RADIUS FROM OBJECT CENTER
    sc_all = SkyCoord(allspec['RA'],allspec['DEC'], unit=(u.deg, u.deg),distance = obj['Dist_kpc']*u.kpc)

    sep = sc_all.separation(sc_gal)
    allspec['rproj_arcm'] = sep.arcmin

    sep3d = sc_all.separation_3d(sc_gal)
    allspec['rproj_kpc'] = sep3d.kpc 


    return allspec

###########################################
def truncated_normal(loc, scale, size, lowerbound = 0.1):
    a = (lowerbound - loc) / scale
    b = np.inf
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)


def calculate_FeH(V0, V0err, ew_cat, ew_cat_err, use_truncnorm = True):

    # MAG AND DISTANCE ERROR-- could improve this
    V0err     = np.sqrt(V0err**2 + 0.1**2)  # add dmod error
    Vmag0_abs = np.random.normal(loc=V0, scale=V0err, size=5000)
    
    
    # Distribute equivalent widths as Truncated Normal (cutoff at zero)
    ew_cat_distrib = truncated_normal(loc=ew_cat, scale=ew_cat_err, lowerbound = 0., size=5000)
    

    # #######Carrera 2013##########
    # [value, error]  using M_V
    #a = [-3.45,0.04]
    #b = [0.16,0.01]
    #c = [0.41,0.004]
    #d = [-0.53,0.11]
    #e = [0.019, 0.002]
    #FeH = a[0] + b[0]* mag + c[0]*CaT + d[0]*CaT**(-1.5) + e[0]*CaT*mag

    a = np.random.normal(loc=-3.45, scale=0.04, size=5000)
    b = np.random.normal(loc=0.16,  scale=0.01, size=5000)
    c = np.random.normal(loc=0.41,  scale=0.004,size=5000)
    d = np.random.normal(loc=-0.53, scale=0.11, size=5000)
    e = np.random.normal(loc=0.019, scale=0.002,size=5000)

    # Calculate [Fe/H]
    FeH = a + (b * Vmag0_abs) + (c * ew_cat_distrib) + (d * ew_cat_distrib**(-1.5)) + \
                                (e * ew_cat_distrib * Vmag0_abs)
    
    masked_FeH = FeH.copy()
    masked_FeH[masked_FeH < -6] = np.nan
       
    feh_medians = np.nanpercentile(masked_FeH, [16,50,84])
    feh           = feh_medians[1]
    feh_err       = (feh_medians[2] - feh_medians[0])/2.
    feh_err_upper = feh_medians[2] - feh_medians[1]
    feh_err_lower = feh_medians[1] - feh_medians[0]

    # 0.1 dex systematic error due to scatter of the Carrera reln itself
    feh_err = np.sqrt(feh_err**2 + (0.1)**2)

    return feh,feh_err, feh_err_upper,feh_err_lower


# CALCULATE FeH FROM CaT EWs
def CaT_to_FeH(alldata):
    

    for ii,slt in enumerate(alldata): 

        alldata['ew_feh'][ii]      = -999.
        alldata['ew_feh_err'][ii]  = -999.
        if (slt['MV_o'] > -99) & (slt['ew_cat'] > 0.) & (slt['MV_o'] < 3):

            # MAGNITUDE ERRORS
            mag     = slt['MV_o']
            magerr  = slt['rmag_err']
            if magerr < 0: 
                magerr = 0.1

            # EW ERRORS    
            CaT     = slt['ew_cat']
            CaTerr  = slt['ew_cat_err']

            FeH, FeH_err, ferru,ferrl = calculate_FeH(mag, magerr, CaT, CaTerr)

            alldata['ew_feh'][ii]      = FeH
            alldata['ew_feh_err'][ii]  = FeH_err

    return alldata



###########################################
def match_gaia(obj,allspec):

    DEIMOS_RAW  = os.getenv('DEIMOS_RAW')
    gaia_file   = DEIMOS_RAW + '/Gaia_DR3/gaia_dr3_'+obj['Name2']+'.csv'

    
    if not os.path.isfile(gaia_file):
        print('NO GAIA FILE',gaia_file)

        
    if os.path.isfile(gaia_file):

        gaia = Table.read(gaia_file)
  

        cgaia   = SkyCoord(ra=gaia['ra']*u.degree, dec=gaia['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cgaia)  
        foo = np.arange(0,np.size(idx),1)

        mt  = foo[d2d < 1.25*u.arcsec]
        mt2 = idx[d2d < 1.25*u.arcsec]
        allspec['gaia_source_id'][mt]     = gaia['source_id'][mt2]
        allspec['gaia_pmra'][mt]          = gaia['pmra'][mt2] 
        allspec['gaia_pmra_err'][mt]      = gaia['pmra_error'][mt2]
        allspec['gaia_pmdec'][mt]         = gaia['pmdec'][mt2] 
        allspec['gaia_pmdec_err'][mt]     = gaia['pmdec_error'][mt2]
        allspec['gaia_pmra_pmdec_corr'][mt]  = gaia['pmra_pmdec_corr'][mt2]

        allspec['gaia_parallax'][mt]      = gaia['parallax'][mt2] 
        allspec['gaia_parallax_err'][mt]  = gaia['parallax_error'][mt2]
        allspec['gaia_phot_variable_flag'][mt] = gaia['phot_variable_flag'][mt2]
        allspec['gaia_rv'][mt]            = gaia['radial_velocity'][mt2] 
        allspec['gaia_rv_err'][mt]        = gaia['radial_velocity_error'][mt2] 
        allspec['gaia_grvs_mag'][mt]      = gaia['grvs_mag'][mt2] 

        allspec['gaia_aen'][mt]           = gaia['astrometric_excess_noise'][mt2] 
        allspec['gaia_aen_sig'][mt]       = gaia['astrometric_excess_noise_sig'][mt2]


        # SET NON_DETECTED BACK TO DEFAULT
        # GAIA DEFAULTS ARE ZERO (CONFUSING!)
        m = allspec['gaia_pmra_err']  == -999.
        allspec['gaia_pmra'][m]       = -999.
        m = allspec['gaia_pmdec_err'] == -999.
        allspec['gaia_pmdec'][m]      = -999.
        m = allspec['gaia_parallax_err'] == -999.
        allspec['gaia_parallax'][m]   = -999.
        mrv = allspec['gaia_rv']      == -999.
        allspec['gaia_rv'][mrv]       = -999.
        allspec['gaia_rv_err'][mrv]   = -999.

        nrv = np.sum(allspec['gaia_rv'] > -999.)

        # SET GAIA FLAG 
        m = allspec['gaia_pmra'] > -999
        allspec['flag_gaia'][m] = 1

        print('GAIA: Matched {} stars and {} Gaia RVS'.format(np.size(mt),nrv))


    return allspec

###########################################
###########################################
# CALCULATE MAGNITUDE ERRORS FROM LEGACY FILES
def legacy_mag_err(flux, flux_ivar):

    # MINIMUM MAG ERROR
    mag_err    = np.zeros(np.size(flux))
    m          = (np.isfinite(flux_ivar)) & (flux_ivar >0)

    flux_err   = 1./np.sqrt(flux_ivar[m])
    mag_err[m] = (2.5/np.log(10.)) * (flux_err/flux[m])

    mag_err = np.sqrt(mag_err**2 + 0.02**2)

    return mag_err

def transform_sdss2decals(g_sdss, r_sdss):

    # TRANSFORMATION FROM Dey+ 2019, Appendix B
    gi = g_sdss - r_sdss +0.25

    g_decals = g_sdss + 0.0244 - 0.1183*gi + 0.0322*gi**2 - 0.0066*gi**3
    r_decals = r_sdss - 0.0005 - 0.0868*gi + 0.0287*gi**2 - 0.0092*gi**3

    return g_decals, r_decals


def transform_ps12decals(g_ps1, r_ps1):

    # TRANSFORMATION FROM Dey+ 2019, Eq 1+2
    gi = g_ps1 - r_ps1 +0.25

    g_decals = g_ps1 + 0.00062 + 0.03604*gi + 0.01028*gi**2 - 0.00613*gi**3
    r_decals = r_ps1 + 0.00495 - 0.08435*gi + 0.03222*gi**2 - 0.01140*gi**3

    return g_decals, r_decals


###########################################
###########################################
def match_photometry(obj,allspec):
    
    DEIMOS_RAW      = os.getenv('DEIMOS_RAW')

    # POPULATE EBV
    nall            = np.size(allspec)
    nobj            = np.sum(allspec['v_err'] >= 0)
    nstar           = np.sum(allspec['v_err'] > 0)

    # DEFINE MATCHING LENGTH
    dm = 1.25
    dm_serendip = 2.

  #####################
    ### PRIMARY SOURCE:   LEGACY DR10
    #   Using dereddened AB magnitudes in DECAM system
    if obj['Phot'] == 'ls_dr10':
        file = DEIMOS_RAW + '/Photometry/legacy_DR10/dr10_'+obj['Name2']+'.csv'
        ls_dr10 = ascii.read(file)
        
        ls_dr10.rename_column('dered_mag_g', 'gmag')
        ls_dr10.rename_column('dered_mag_r', 'rmag')

        # REPLACE INF VALUES
        m = np.isfinite(ls_dr10['gmag'])
        ls_dr10['gmag'][~m] = -999
        m = np.isfinite(ls_dr10['rmag'])
        ls_dr10['rmag'][~m] = -999


        # Hack, update
        if obj['Name2'] == 'Eri4':
            ls_dr10['rmag'] = ls_dr10['dered_mag_i']+0.1 
            ls_dr10['flux_r'] = ls_dr10['flux_i'] 
            ls_dr10['flux_ivar_r'] = ls_dr10['flux_ivar_i'] 


        # NO DEIMOS SOURCES THIS FAINT, CUT TO REDUCE MIS-MATCHING
        ls_dr10 = ls_dr10[ls_dr10['rmag'] < 25]


        # CORRECT BASS MAGNITUDES NORTHERN MAGNITUDES
        if np.median(ls_dr10['dec'] > 34.):
            ls_dr10['rmag'] =  -0.0382 * (ls_dr10['gmag'] - ls_dr10['rmag']) + 0.0108 + ls_dr10['rmag']

        
        cls_dr10 = SkyCoord(ra=ls_dr10['ra']*u.degree, dec=ls_dr10['dec']*u.degree) 
        cdeimos  = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < dm*u.arcsec]
        print(np.size(mt))

        allspec['rmag_o'][mt] = ls_dr10['rmag'][idx[d2d < dm*u.arcsec]] 
        allspec['gmag_o'][mt] = ls_dr10['gmag'][idx[d2d < dm*u.arcsec]] 

        allspec['rmag_err'][mt] = legacy_mag_err(ls_dr10['flux_r'][idx[d2d < dm*u.arcsec]] , ls_dr10['flux_ivar_r'][idx[d2d < dm*u.arcsec]] )
        allspec['gmag_err'][mt] = legacy_mag_err(ls_dr10['flux_g'][idx[d2d < dm*u.arcsec]] , ls_dr10['flux_ivar_g'][idx[d2d < dm*u.arcsec]] )

        allspec['phot_source'][mt] = 'ls_dr10'
        allspec['phot_type'][mt] = ls_dr10['type'][idx[d2d < dm*u.arcsec]] 



        # INCREASE FOR SERENDIPS W/O Match
        sm  = (d2d < dm_serendip*u.arcsec) & (allspec['serendip'] == 1) & (allspec['rmag_o'] < 0)
        

        if np.sum(sm) > 0:
            mts = foo[sm]
            #print(allspec['rmag_o'][mts])

            allspec['rmag_o'][mts] = ls_dr10['rmag'][idx[sm]] 
            allspec['gmag_o'][mts] = ls_dr10['gmag'][idx[sm]] 

            allspec['rmag_err'][mts] = legacy_mag_err(ls_dr10['flux_r'][idx[sm]] , ls_dr10['flux_ivar_r'][idx[sm]] )
            allspec['gmag_err'][mts] = legacy_mag_err(ls_dr10['flux_g'][idx[sm]] , ls_dr10['flux_ivar_g'][idx[sm]] )

            allspec['phot_source'][mts] = 'ls_dr10'
            allspec['phot_type'][mts] = ls_dr10['type'][idx[sm]] 

            #for ob in allspec[mts]:
            #    print(ob['rmag_o'],ob['RA'],ob['DEC'],ob['SN'],ob['marz_flag'])


    #####################
    ### MUNOZ -- COMPLETE UNPUBLISHED CATALOGS
    if obj['Phot'] == 'munozf':
        
        file = DEIMOS_RAW + '/Photometry/munoz_full/final_'+obj['Name2']+'.phot'

        munozf = ascii.read(file)
        munozf.rename_column('col2', 'RA')
        munozf.rename_column('col3', 'DEC')
        munozf.rename_column('col4', 'g')
        munozf.rename_column('col5', 'gerr')
        munozf.rename_column('col6', 'r')
        munozf.rename_column('col7', 'rerr')

        
        if (obj['Name2'] == 'Eri') | (obj['Name2'] == 'K2')| (obj['Name2'] == 'Leo2')| \
           (obj['Name2'] == 'Seg2') | (obj['Name2'] == 'N2419')| (obj['Name2'] == 'Pal2'):
            munozf['RA'] *= 15.
        
        munozf  = munozf[munozf['r'] < 25.0]
        cmunf   = SkyCoord(ra=munozf['RA']*u.degree, dec=munozf['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)  
        foo = np.arange(0,np.size(idx),1)

    
        # TRANSFORM SDSS TO DECALS
        r_sdss = munozf['r']
        g_sdss = munozf['g']
        r_decals,g_decals = transform_sdss2decals(r_sdss,g_sdss)

        # Get Ar/Ag
        allspec, Ar, Ag = get_ebv(allspec)
        mt = foo[d2d < dm*u.arcsec]

        allspec['rmag_o'][mt]   = r_decals[idx[d2d < dm*u.arcsec]]  - Ar[mt]
        allspec['gmag_o'][mt]   = g_decals[idx[d2d < dm*u.arcsec]]  - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < dm*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < dm*u.arcsec]]

        allspec['phot_source'][mt] = 'munozf'
                  

       # INCREASE FOR SERENDIPS W/O Match
        sm  = (d2d < dm_serendip*u.arcsec) & (allspec['serendip'] == 1) & (allspec['rmag_o'] < 0)
        
        if np.sum(sm) > 0:
            mts = foo[sm]
            #print(allspec['rmag_o'][mts])

            allspec['rmag_o'][mts]   = r_decals[idx[sm]]  - Ar[mts]
            allspec['gmag_o'][mts]   = g_decals[idx[sm]]  - Ag[mts]
            allspec['rmag_err'][mts] = munozf['rerr'][idx[sm]]
            allspec['gmag_err'][mts] = munozf['gerr'][idx[sm]]
            allspec['phot_source'][mts] = 'munozf' 
            #for ob in allspec[mts]:
            #    print(ob['rmag_o'],ob['RA'],ob['DEC'],ob['SN'],ob['marz_flag'])


     
    if obj['Phot'] == 'munoz18_2':
        
        file = DEIMOS_RAW + '/Photometry/munoz18/munoz18_secondary.txt'
        # Get Ar/Ag
        allspec, Ar, Ag = get_ebv(allspec)

        munozf = ascii.read(file)

        cmunf   = SkyCoord(ra=munozf['ra']*u.degree, dec=munozf['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)  
        foo = np.arange(0,np.size(idx),1)
        mt  = foo[d2d < dm*u.arcsec]
        
        r_sdss = munozf['r'][idx[d2d < dm*u.arcsec]] 
        g_sdss = munozf['g'][idx[d2d < dm*u.arcsec]] 
        r_decals,g_decals = transform_sdss2decals(r_sdss,g_sdss)

        allspec['rmag_o'][mt]   = r_decals - Ar[mt]
        allspec['gmag_o'][mt]   = g_decals - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < dm*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < dm*u.arcsec]]
        
        allspec['phot_source'][mt] = 'munoz18_2'


    #####################
    ### GC SDSS PHOTOMETRY
    # http://classic.sdss.org/dr7/products/value_added/anjohnson08_clusterphotometry.html
    if obj['Phot'] == 'sdss_gc':

        file = DEIMOS_RAW + '/Photometry/sdss_gc/sdss_gc_'+obj['Name2']+'.phot'
        sdss = ascii.read(file)
        
        sdss.rename_column('col5', 'RA')
        sdss.rename_column('col6', 'DEC')
        sdss.rename_column('col16', 'gmag')
        sdss.rename_column('col17', 'gmag_err')
        sdss.rename_column('col23', 'rmag')
        sdss.rename_column('col24', 'rmag_err')

        #if obj['Name2'] == 'N6791':
        #    file = DEIMOS_RAW + '/Photometry/sdss_gc/sdss_dr14_'+obj['Name2']+'.csv'
        #    sdss = ascii.read(file)
        #    sdss.rename_column('ra', 'RA')
        #    sdss.rename_column('dec', 'DEC')
        #    sdss.rename_column('g', 'gmag')
        #    sdss.rename_column('Err_g', 'gmag_err')
        #    sdss.rename_column('r', 'rmag')
        #    sdss.rename_column('Err_r', 'rmag_err')


        csdss   = SkyCoord(ra=sdss['RA']*u.degree, dec=sdss['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(csdss)  
        foo = np.arange(0,np.size(idx),1)
        mt = foo[d2d < dm*u.arcsec]

        # TRANSFORM SDSS TO DECALS
        r_sdss = sdss['rmag'][idx[d2d < dm*u.arcsec]]
        g_sdss = sdss['gmag'][idx[d2d < dm*u.arcsec]]
        r_decals,g_decals = transform_sdss2decals(r_sdss,g_sdss)

        # Get Ar/Ag
        allspec, Ar, Ag = get_ebv(allspec)

        allspec['rmag_o'][mt] =  r_decals - Ar[mt]
        allspec['gmag_o'][mt] =  g_decals - Ag[mt]
        allspec['rmag_err'][mt] = sdss['rmag_err'][idx[d2d < dm*u.arcsec]]
        allspec['gmag_err'][mt] = sdss['gmag_err'][idx[d2d < dm*u.arcsec]]

        allspec['phot_source'][mt] = 'sdss_gc'


    #####################
    ### DECAPS
    if obj['Phot'] == 'decaps':
        file = DEIMOS_RAW + '/Photometry/decaps/decaps_'+obj['Name2']+'.csv'
        decaps = Table.read(file)
        
        
        cls_dr10= SkyCoord(ra=decaps['ra_ok']*u.degree, dec=decaps['dec_ok']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        # REPLACE INF VALUES
        m = np.isfinite(decaps['mean_mag_g'])
        decaps['mean_mag_g'][~m] = -999
        m = np.isfinite(decaps['mean_mag_r'])
        decaps['mean_mag_r'][~m] = -999

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        allspec, Ar, Ag = get_ebv(allspec)

        mt = foo[d2d < dm*u.arcsec]
        allspec['rmag_o'][mt] = decaps['mean_mag_r'][idx[d2d < dm*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = decaps['mean_mag_g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = 0.1
        allspec['gmag_err'][mt] = 0.1

        allspec['phot_source'][mt] = 'decaps'


  #####################
    ## PANSTARRS DR2
    #  https://catalogs.mast.stsci.edu/
    if obj['Phot'] == 'PanS':
        file = DEIMOS_RAW + '/Photometry/PanS/PanS_'+obj['Name2']+'.csv'
        pans = ascii.read(file)
        m=(pans['rMeanPSFMag'] != -999) & (pans['gMeanPSFMag'] != -999)
        pans=pans[m]
        
        # TRANSFORM TO DECALS
        g_decals, r_decals = transform_ps12decals(pans['gMeanPSFMag'] , pans['rMeanPSFMag'])

        cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)  
        foo = np.arange(0,np.size(idx),1)

        # GET reddening 
        allspec, Ar, Ag = get_ebv(allspec)


        # INCREASED TO 2" TO GET CENTRAL GLOBULAR CLUSTER MEMBERS
        ds = 2.0
        mt = foo[d2d < ds*u.arcsec]
        allspec['rmag_o'][mt] = r_decals[idx[d2d < ds*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = g_decals[idx[d2d < ds*u.arcsec]] - Ag[mt]
        
        allspec['rmag_err'][mt] = pans['rMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['gmag_err'][mt] = pans['gMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]

        allspec['phot_source'][mt] = 'PanS'


 #####################
    ## PANSTARRS DR2
    #  https://catalogs.mast.stsci.edu/
    if obj['Phot'] == 'PanS1':
        file = DEIMOS_RAW + '/Photometry/PanS/PanS1_'+obj['Name2']+'.csv'
        pans = ascii.read(file)
        m=(pans['rPSFMag'] != -999) & (pans['gPSFMag'] != -999)
        pans=pans[m]
        
        # TRANSFORM TO DECALS
        g_decals, r_decals = transform_ps12decals(pans['gPSFMag'] , pans['rPSFMag'])

        cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)  
        foo = np.arange(0,np.size(idx),1)

        # GET reddening 
        allspec, Ar, Ag = get_ebv(allspec)


        # INCREASED TO 2" TO GET CENTRAL GLOBULAR CLUSTER MEMBERS
        ds = 2.25
        mt = foo[d2d < ds*u.arcsec]
        allspec['rmag_o'][mt] = r_decals[idx[d2d < ds*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = g_decals[idx[d2d < ds*u.arcsec]] - Ag[mt]
        
        allspec['rmag_err'][mt] = pans['rPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['gmag_err'][mt] = pans['gPSFMagErr'][idx[d2d < ds*u.arcsec]]

        allspec['phot_source'][mt] = 'PanS'

    
    #####################
    ### USE GAIA IF THERE ARE NO OTHER OPTIONS
    if obj['Phot'] == 'gaia':
        file = DEIMOS_RAW + '/Gaia_DR3/gaia_dr3_'+obj['Name2']+'.csv'
        gaia = ascii.read(file)
        
        # TRANSFORM USING Table 5.7
        #https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8 

        G_BP_RP = gaia['bp_rp']
        G       = gaia['phot_g_mean_mag']

        Gr   = -0.09837 + 0.08592*G_BP_RP + 0.1907*G_BP_RP**2 - 0.1701*G_BP_RP**3 + 0.02263*G_BP_RP**4
        Gg   =  0.2199  - 0.6365*G_BP_RP  - 0.1548*G_BP_RP**2 + 0.0064*G_BP_RP**3
        rmag =  G - Gr
        gmag =  G - Gg
        err  = (2.5/np.log(10)) / gaia['phot_g_mean_flux_over_error']
        gaia_err  = np.sqrt(err**2 + 0.07**2)

        # TRANSFORMATION IS TOTALLY OFF
        rmag = gaia['phot_rp_mean_mag']
        gmag = gaia['phot_g_mean_mag']-0.3


        cgaia   = SkyCoord(ra=gaia['ra']*u.degree, dec=gaia['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cgaia)  
        foo = np.arange(0,np.size(idx),1)

        allspec, Ar, Ag = get_ebv(allspec)

        mt = foo[d2d < dm*u.arcsec]
        allspec['rmag_o'][mt]   = rmag[idx[d2d < dm*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt]   = gmag[idx[d2d < dm*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = gaia_err[idx[d2d < dm*u.arcsec]] 
        allspec['gmag_err'][mt] = gaia_err[idx[d2d < dm*u.arcsec]] 

        allspec['phot_source'][mt] = 'gaia'

        

    # REMOVE SERENDIP STARS WITHOUT PHOTOMETRY
    m_serendip_nophot =  (allspec['serendip'] == 1) &  (allspec['rmag_o'] < 0) 
    allspec           = allspec[~m_serendip_nophot]


    # DETERMINE MV AND CONVERT CAT -> FEH
    m_miss_star = (allspec['rmag_o'] < 0)  & (allspec['v_err'] > 0)
    print('PHOT: Matched {} stars, missing {} star targets'.format(np.size(mt),np.sum(m_miss_star)))

    allspec = calc_rproj(allspec,obj)
    allspec = calc_MV_star(allspec,obj)
    allspec = CaT_to_FeH(allspec)


    return allspec


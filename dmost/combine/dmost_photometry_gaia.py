#!/usr/bin/env python

import numpy as np
import os,sys
    
from astropy.coordinates import SkyCoord


from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits

import sfdmap


###########################################
def get_ebv(allspec):

    DEIMOS_RAW  = os.getenv('DEIMOS_RAW')

    sdf_ext = sfdmap.SFDMap(DEIMOS_RAW+'SFDmaps/')
    EBV    = sdf_ext.ebv(allspec['RA'],allspec['DEC'])

    allspec['EBV'] = EBV

    Ar = 2.751 * allspec['EBV']
    Ag = 3.793 * allspec['EBV']


    return allspec, Ar, Ag
    

###########################################
def calc_MV_star(allspec,obj):
    
    # DISTANCE MODULUS AND REDDENING
    dmod = 5.*np.log10(obj['Dist_kpc']*1e3) - 5.


    # Jester 2005
    # https://classic.sdss.org/dr5/algorithms/sdssUBVRITransform.php    
    # V = g - 0.59*(g-r) - 0.01 
    #V    =  allspec['gmag_o'] - 0.59*gr_o -0.01
    # Jordi 2006
    # V = allspec['gmag_o'] - 0.565*gr_o -0.016
    # Using Lupton
    # V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054

    gr_o =  allspec['gmag_o'] - allspec['rmag_o'] 
    V    = allspec['gmag_o'] - 0.565*gr_o -0.016

    allspec['MV_o'] = V - dmod
    
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
def CaT_to_FeH(alldata):
    
    FeH = -99
    FeH_err = -99
    
    
    m = (alldata['MV_o'] != -99) & (alldata['ew_cat'] > 0.)
    

    mag     = alldata['MV_o'][m]
    CaT     = alldata['ew_cat'][m]
    CaTerr = alldata['ew_cat_err'][m]

    # #######Carrera 2013##########
    # [value, error]  using M_V
    a = [-3.45,0.04]
    b = [0.16,0.01]
    c = [0.41,0.004]
    d = [-0.53,0.11]
    e = [0.019, 0.002]
    FeH = a[0] + b[0]* mag + c[0]*CaT + d[0]*CaT**(-1.5) + e[0]*CaT*mag

    # PROPOGATE errORS -- ASSUME NO errOR ON MAGNITUDE FOR NOW...
    FeH_err = np.sqrt( (a[1]**2) +\
                       (mag*b[1]**2) +\
                       (c[1]**2*CaT**2 + c[0]**2*CaTerr**2) +\
                       (d[1]**2*CaT**(-1.5*2) + d[0]**2*(CaTerr**2)*CaT**(-2.5)) +\
                       (e[1]**2*(CaT*mag)**2 + CaTerr**2*(e[0]*mag)**2))

    
    alldata['ew_feh'][m]      = FeH
    alldata['ew_feh_err'][m]  = FeH_err

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

        mt  = foo[d2d < 1.*u.arcsec]
        mt2 = idx[d2d < 1.*u.arcsec]
        allspec['gaia_source_id'][mt]    = gaia['source_id'][mt2]
        allspec['gaia_pmra'][mt]         = gaia['pmra'][mt2] 
        allspec['gaia_pmra_err'][mt]     = gaia['pmra_error'][mt2]
        allspec['gaia_pmdec'][mt]        = gaia['pmdec'][mt2] 
        allspec['gaia_pmdec_err'][mt]    = gaia['pmdec_error'][mt2]
        allspec['gaia_parallax'][mt]     = gaia['parallax'][mt2] 
        allspec['gaia_parallax_err'][mt] = gaia['parallax_error'][mt2]
        allspec['gaia_rv'][mt]           = gaia['radial_velocity'][mt2] 
        allspec['gaia_rv_err'][mt]       = gaia['radial_velocity_error'][mt2] 

        allspec['gaia_aen'][mt]          = gaia['astrometric_excess_noise'][mt2] 
        allspec['gaia_aen_sig'][mt]      = gaia['astrometric_excess_noise_sig'][mt2]


        # SET NON_DETECTED BACK TO DEFAULT
        # GAIA DEFAULTS ARE ZERO (CONFUSING!)
        m = allspec['gaia_pmra_err'] == 0.0
        allspec['gaia_pmra'][m]  = -999.
        m = allspec['gaia_pmdec_err'] == 0.0
        allspec['gaia_pmdec'][m]  = -999.
        m = allspec['gaia_parallax_err'] == 0.0
        allspec['gaia_parallax'][m]  = -999.
        mrv = allspec['gaia_rv'] == 0.0
        allspec['gaia_rv'][mrv]  = -999.
        nrv = np.sum(allspec['gaia_rv'] > -999.)

        # SET GAIA FLAG 
        m = allspec['gaia_pmra'] > -999
        allspec['flag_gaia'][m] = 1

        print('GAIA: Matched {} stars and {} Gaia RVS'.format(np.size(mt),nrv))


    return allspec

###########################################
###########################################
def match_photometry(obj,allspec):
    
    DEIMOS_RAW      = os.getenv('DEIMOS_RAW')

    # POPULATE EBV
    allspec, Ar, Ag = get_ebv(allspec)
    nall            = np.size(allspec)
    nobj            = np.sum(allspec['v_err'] >= 0)
    nstar           = np.sum(allspec['v_err'] > 0)


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
        
        cmunf   = SkyCoord(ra=munozf['RA']*u.degree, dec=munozf['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = munozf['r'][idx[d2d < 1.*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = munozf['g'][idx[d2d < 1.*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < 1.*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < 1.*u.arcsec]]

        allspec['phot_source'][mt] = 'munozf'
                  

     
    #####################
    ### MUNOZ -- COMPLETE UNPUBLISHED CATALOGS
    if obj['Phot'] == 'munoz18_2':
        
        file = DEIMOS_RAW + '/Photometry/munoz18/munoz18_secondary.txt'

        munozf = ascii.read(file)

        cmunf   = SkyCoord(ra=munozf['ra']*u.degree, dec=munozf['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = munozf['r'][idx[d2d < 1.*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = munozf['g'][idx[d2d < 1.*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < 1.*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < 1.*u.arcsec]]
        
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

        
        csdss   = SkyCoord(ra=sdss['RA']*u.degree, dec=sdss['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(csdss)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = sdss['rmag'][idx[d2d < 1.*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = sdss['gmag'][idx[d2d < 1.*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = sdss['rmag_err'][idx[d2d < 1.*u.arcsec]]
        allspec['gmag_err'][mt] = sdss['gmag_err'][idx[d2d < 1.*u.arcsec]]

        allspec['phot_source'][mt] = 'sdss_gc'



  #####################
    ### LEGACY DR10
    if obj['Phot'] == 'ls_dr10':
        file = DEIMOS_RAW + '/Photometry/legacy_DR10/dr10_'+obj['Name2']+'.csv'
        ls_dr10 = ascii.read(file)
        
        ls_dr10.rename_column('dered_mag_g', 'gmag')
        ls_dr10.rename_column('dered_mag_r', 'rmag')

        # CORRECT BASS MAGNITUDES NORTHERN MAGNITUDES
        if np.median(ls_dr10['dec'] > 34.):
            ls_dr10['rmag'] =  -0.0382 * (ls_dr10['gmag'] - ls_dr10['rmag']) + 0.0108 + ls_dr10['rmag']

        
        cls_dr10 = SkyCoord(ra=ls_dr10['ra']*u.degree, dec=ls_dr10['dec']*u.degree) 
        cdeimos  = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = ls_dr10['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = ls_dr10['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

        allspec['phot_source'][mt] = 'ls_dr10'


   ###########################
    if obj['Phot'] == 'ls_dr10i':
        file = DEIMOS_RAW + '/Photometry/legacy_DR10/dr10_'+obj['Name2']+'.csv'
        ls_dr10 = ascii.read(file)
        
        # hack, if rmag isn't available
        gmag = ls_dr10['dered_mag_g']
        rmag = ls_dr10['dered_mag_i'] + 0.3


        cls_dr10 = SkyCoord(ra=ls_dr10['ra']*u.degree, dec=ls_dr10['dec']*u.degree) 
        cdeimos  = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = rmag[idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = gmag[idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01
      
        allspec['phot_source'][mt] = 'ls_dr10'

    ### DELVE
    if obj['Phot'] == 'delve':
        file = DEIMOS_RAW + '/Photometry/delve/delve_'+obj['Name2']+'.csv'
        delve = Table.read(file)
        
        
        cls_dr10= SkyCoord(ra=delve['RA']*u.degree, dec=delve['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = delve['rmag_o'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = delve['gmag_o'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = delve['rmag_err'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_err'][mt] = delve['rmag_err'][idx[d2d < 1.*u.arcsec]] 

        allspec['phot_source'][mt] = 'delve'


    ## PANSTARRS DR2
    #  https://catalogs.mast.stsci.edu/
    if obj['Phot'] == 'PanS':
        file = DEIMOS_RAW + '/Photometry/PanS/PanS_'+obj['Name2']+'.csv'
        pans = ascii.read(file)
        m=(pans['rMeanPSFMag'] != -999) & (pans['gMeanPSFMag'] != -999)
        pans=pans[m]
        
        # TRANSFORM TO SDSS USING Tonry et al 2012
        gr_p   = pans['gMeanPSFMag'] - pans['rMeanPSFMag']
        g_sdss = pans['gMeanPSFMag'] + 0.013 + 0.145*gr_p + 0.019*gr_p**2
        r_sdss = pans['rMeanPSFMag'] - 0.001 + 0.004*gr_p + 0.007*gr_p**2
        
        cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)  
        foo = np.arange(0,np.size(idx),1)

        # INCREASED TO 2" TO GET CENTRAL GLOBULAR CLUSTER MEMBERS
        ds = 2.0
        mt = foo[d2d < ds*u.arcsec]
        allspec['rmag_o'][mt] = r_sdss[idx[d2d < ds*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = g_sdss[idx[d2d < ds*u.arcsec]] - Ag[mt]
        
        allspec['rmag_err'][mt] = pans['rMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['gmag_err'][mt] = pans['gMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]

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

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt]   = rmag[idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt]   = gmag[idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = gaia_err[idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_err'][mt] = gaia_err[idx[d2d < 1.*u.arcsec]] 

        allspec['phot_source'][mt] = 'gaia'

    ### HST
    if obj['Phot'] == 'hst':
        file = DEIMOS_RAW + '/Photometry/hst/hst_'+obj['Name2']+'.dat'
        hst = ascii.read(file)
    
        hst.rename_column('F606', 'gmag')
        hst.rename_column('F814', 'rmag')
        cls_hst   = SkyCoord(ra=hst['RA']*u.degree, dec=hst['Dec']*u.degree) 


        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_hst)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = hst['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = hst['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

    ### HST - ACS
    if obj['Phot'] == 'ACS':
        file = DEIMOS_RAW + '/Photometry/hst/hst_'+obj['Name2']+'.fits'
        hst = Table.read(file)
        # REMOVE SUPER FAINT STARS
        m=hst['F814W_VEGA'] > 25
        hst=hst[m]

        hst.rename_column('F475W_VEGA', 'gmag')
        hst.rename_column('F814W_VEGA', 'rmag')
        cls_hst   = SkyCoord(ra=hst['RA']*u.degree, dec=hst['DEC']*u.degree) 

        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_hst)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = hst['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = hst['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01



#####################
    ### PANDAS
    if obj['Phot'] == 'pandas':
        file = DEIMOS_RAW + '/Photometry/PANDAS/PANDAS_'+obj['Name2']+'.csv'
        pandas = ascii.read(file)
        
        pandas.rename_column('g', 'gmag')
        pandas.rename_column('i', 'rmag')          # NEED TO TRANSFORM!!!
        pandas.rename_column('dg', 'gmag_err')
        pandas.rename_column('di', 'rmag_err')


        cpandas = SkyCoord(ra=pandas['RA']*u.degree, dec=pandas['Dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpandas)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = pandas['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = pandas['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

        allspec['phot_source'][mt] = 'pandas'

        

#####################
    ### MASSEY
    if obj['Phot'] == 'massey':
        file = DEIMOS_RAW + '/Photometry/other/massey_2007.fits'
        massey = Table.read(file)

        massey['V-R'] = massey['Vmag'] -  massey['V-R']       # CREATE R-mag
        massey.rename_column('Vmag', 'gmag')
        massey.rename_column('V-R', 'rmag')  
#        massey.rename_column('dg', 'gmag_err')
#        massey.rename_column('di', 'rmag_err')
        massey['RAJ2000'] = np.array(massey['RAJ2000'])
        massey['DEJ2000'] = np.array(massey['DEJ2000'])

        cmassey = SkyCoord(ra=massey['RAJ2000']*u.degree, dec=massey['DEJ2000']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmassey)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = massey['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = massey['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

        



    # DETERMINE MV AND CONVERT CAT -> FEH
    m_miss_star = (allspec['rmag_o'] < 2)  & (allspec['v_err'] > 0)
    print('PHOT: Matched {} stars, missing {} star targets'.format(np.size(mt),np.sum(m_miss_star)))

    allspec = calc_rproj(allspec,obj)
    allspec = calc_MV_star(allspec,obj)
    allspec = CaT_to_FeH(allspec)


    return allspec


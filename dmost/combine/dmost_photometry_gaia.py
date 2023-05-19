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

    # JORDI TRANSFORMATIONS
    #V-g   =     (-0.565 ± 0.001)*(g-r) - (0.016 ± 0.001)
    #V-g   =     (-0.569 ± 0.007)*(g-r) + (0.021 ± 0.004)  metal-poor Table 4
    
    gr_o =  allspec['gmag_o'] - allspec['rmag_o'] 
    V    =  allspec['gmag_o'] -0.569*gr_o + 0.021
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
    #alldata['ew_feh_flag'][m] = 1

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

        mt  = foo[d2d < 1.5*u.arcsec]
        mt2 = idx[d2d < 1.5*u.arcsec]
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
        allspec['gaia_flag'][m] = 1

        print('GAIA: Matched {} stars and {} Gaia RVS'.format(np.size(mt),nrv))


    return allspec

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

        mt = foo[d2d < 1.5*u.arcsec]
        allspec['rmag_o'][mt] = munozf['r'][idx[d2d < 1.5*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = munozf['g'][idx[d2d < 1.5*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < 1.5*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < 1.5*u.arcsec]]

         
               
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

        
        

     
    #####################
    ### MUNOZ -- COMPLETE UNPUBLISHED CATALOGS
    if obj['Phot'] == 'munoz18_2':
        
        file = DEIMOS_RAW + '/Photometry/munoz18/munoz18_secondary.txt'

        munozf = ascii.read(file)

        cmunf   = SkyCoord(ra=munozf['ra']*u.degree, dec=munozf['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.5*u.arcsec]
        allspec['rmag_o'][mt] = munozf['r'][idx[d2d < 1.5*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = munozf['g'][idx[d2d < 1.5*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < 1.5*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < 1.5*u.arcsec]]
        


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

        

       
 
    ### STETSON PHOTOMETRY
    if obj['Phot'] == 'stetson':
        
        # DOWNLOADED FROM STETSON"S WEBSITE, EDIT .pho to add ID column name
        file_pho = DEIMOS_RAW +'Photometry/stetson/'+obj['Name']+'.pho'
        file_pos = DEIMOS_RAW +'Photometry/stetson/'+obj['Name']+'.pos'

        pho = ascii.read(file_pho,data_start=1)
        pos = ascii.read(file_pos,data_start=1)
        if (np.size(pho) != np.size(pos)):
            print('STETSON FILES MIS-MATCHED')

        # MATCH TO ASPEC            
        ra = pos['col1']
        dec= pos['col2']


        # TRANSFORM PHOTOMETRY 
        # Jordi 2006: http://adsabs.harvard.edu/abs/2006A%26A...460..339J
        # FOR METAL POOR STARS
        VR = pho['V'] - pho['R']
        gr = (1.72)*(VR)  - (0.198)  
        r  = (0.34)*(VR)  + (0.015) + pho['R']
        g  = gr + r 

        # IF R ISN"T AVAILABLE, THEN JESTER et al 2005
        if (np.median(VR) < -10):
            BV = pho['B'] - pho['V']
            V  = pho['V']
            g  =  V + 0.60*(BV) - 0.12    
            r  =  V - 0.42*(BV) + 0.11    

        cstet   = SkyCoord(ra=ra*u.degree, dec=dec*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cstet)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = r[idx[d2d < 1.*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = g[idx[d2d < 1.*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

    
#####################
    ### LEGACY DR9
    if obj['Phot'] == 'ls_dr9':
        file = DEIMOS_RAW + '/Photometry/legacy_DR9/dr9_'+obj['Name2']+'.csv'
        ls_dr9 = ascii.read(file)
        
        ls_dr9.rename_column('dered_mag_g', 'gmag')
        ls_dr9.rename_column('dered_mag_r', 'rmag')

        
        cls_dr9   = SkyCoord(ra=ls_dr9['ra']*u.degree, dec=ls_dr9['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr9)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = ls_dr9['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = ls_dr9['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01

        

    ### DELVE
    if obj['Phot'] == 'delve':
        file = DEIMOS_RAW + '/Photometry/delve/delve_'+obj['Name2']+'.csv'
        delve = Table.read(file)
        
        
        cls_dr9   = SkyCoord(ra=delve['RA']*u.degree, dec=delve['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr9)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = delve['rmag_o'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = delve['gmag_o'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = delve['rmag_err'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_err'][mt] = delve['rmag_err'][idx[d2d < 1.*u.arcsec]] 

 
    #####################
    ### USE GAIA IF THERE ARE NO OTHER OPTIONS
    if obj['Phot'] == 'gaia':
        file = DEIMOS_RAW + '/Gaia_DR3/gaia_dr3_'+obj['Name2']+'.csv'
        gaia = ascii.read(file)
        
        # THIS IS WRONG
        gaia.rename_column('phot_g_mean_mag', 'gmag')
        gaia.rename_column('phot_rp_mean_mag', 'rmag')

        
        cgaia   = SkyCoord(ra=gaia['ra']*u.degree, dec=gaia['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cgaia)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < 1.*u.arcsec]
        allspec['rmag_o'][mt] = gaia['rmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['gmag_o'][mt] = gaia['gmag'][idx[d2d < 1.*u.arcsec]] 
        allspec['rmag_err'][mt] = 0.01
        allspec['gmag_err'][mt] = 0.01
       


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
    ### LEGACY DR9
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


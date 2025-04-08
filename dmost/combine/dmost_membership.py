import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy import units as u


from astropy.table import vstack
from scipy.stats import norm
from astropy.stats import sigma_clip


import glob

from dmost.core import dmost_utils
from dmost.combine import dmost_vdisp

DEIMOS_RAW     = os.getenv('DEIMOS_RAW')



######################################################
def mg_wmean(x, x_err):
    """
    Return the weighted average and standard deviation.
    values, weights -- NumPy ndarrays with the same shape.
    """
    weights = 1./x_err
    avg     = np.average(x, weights=weights)
    var     = np.average((x-avg)**2, weights=weights)
    
    return (avg, np.sqrt(var))



######################################################
def plot_isochrone_parsec_decam(dist,feh):
    
    DEIMOS_RAW = '/Users/mg37/Dropbox/DEIMOS/'
    iso = ascii.read(DEIMOS_RAW+'/Photometry/isochrones/iso_decam_t12.dat',header_start=13)

    # FIND MATCHING [Fe/H]
    marg = np.argmin(np.abs(iso['MH'] - feh))
    m    = (iso['MH'] == iso['MH'][marg]) & (iso['rmag'] < 25)
    single_iso = iso[m]
    label = single_iso['label']

    
    m = single_iso['label'] < 4
    ms_rbg = single_iso[m]
    ms_rbg.sort('rmag')

    r_iso  = ms_rbg['rmag'] + 5.*np.log10(dist*1e3) - 5.  
    gr_iso = ms_rbg['gmag'] - ms_rbg['rmag'] + 0.03

    return r_iso,gr_iso

######################################################
def plot_isochrone_HB(dist):

    iso = Table.read(DEIMOS_RAW+'/Photometry/isochrones/M92_fiducial_HB.dat',format='ascii',guess=False)
    
    r_iso  = iso['rmag'] + 5.*np.log10(dist*1e3) - 5. 
    gr_iso = iso['gmr'] 
    
    return r_iso,gr_iso 


######################################################
def membership_CMD(alldata,this_obj,cmd_min = 0.25):

    # GET ISOCHRONE PROPERTIES    
    dist= this_obj['Dist_kpc']
    feh = this_obj['feh_guess']

    r,gr       = plot_isochrone_parsec_decam(dist,feh)
    r_hb,gr_hb = plot_isochrone_HB(dist)
       
    dmin, emin = [],[]
    for star in alldata:
        err = np.sqrt(star['gmag_err']**2 + star['rmag_err']**2)

        d  = (r - star['rmag_o'])**2 + (gr - (star['gmag_o'] - star['rmag_o']))**2
        d2 = (r_hb - star['rmag_o'])**2 + (gr_hb - (star['gmag_o'] - star['rmag_o']))**2

        tmp = np.min(d)
        tmp2=np.min(d2)        
        if tmp2 < tmp:
            tmp=tmp2

        dmin.append(np.sqrt(tmp))
        emin.append(err)
            
    #  CMD Prob PLUS PHOTOMETRIC ERROR
    texp     = np.array(dmin)**2 / (2*(np.array(emin)**2 + cmd_min**2))
    Pmem_cmd = np.exp(-1.*texp)
 
    return Pmem_cmd


######################################################
# REMOVE STARS WITH STRONG NaI lines
# Criteria based on Schiavon+1997
def membership_NaI(alldata, this_obj):

    naI_lim =  1.

    m_rmv    = ((alldata['ew_naI'] - alldata['ew_naI_err']) > naI_lim) & (alldata['MV_o'] < 4)
    Pmem_NaI = ~m_rmv

    return Pmem_NaI



######################################################
# REMOVE FOREGROUND STARS WITH LARGE GAIA PARALLAX 
def membership_parallax(alldata, this_obj):

    nstar         = np.size(alldata)
    Pmem_parallax = np.ones(nstar)

    obj_parallax  = 1./this_obj['Dist_kpc']

    # ONLY TEST STARS WITH AVAILABLE PARALLAX
    m_parallax   =(alldata['gaia_parallax_err'] > 0)


    # REMOVE STARS WITH LARGE MEASURED PARALLAX
    parallax_criteria = alldata['gaia_parallax'] - 3. * alldata['gaia_parallax_err']
    mrm               = parallax_criteria > 2. * obj_parallax        
    Pmem_parallax     = ~(mrm & m_parallax)

    return Pmem_parallax



######################################################
def membership_v_crude(alldata,this_obj):

    # NOT USED TO DETERMINE ANY PROPERTIES     
    min_v = this_obj['v_guess'] - 25
    max_v = this_obj['v_guess'] + 25

    Pmem_crude_v = (alldata['v'] > min_v) & (alldata['v'] < max_v)

    return Pmem_crude_v


######################################################
def membership_vdisp(alldata,this_obj,Pmem_tmp,crude_cut=25.):
    
    min_v = this_obj['v_guess'] - crude_cut
    max_v = this_obj['v_guess'] + crude_cut

    crude_v = (alldata['v'] > min_v) & (alldata['v'] < max_v) & (Pmem_tmp > 0.2)


    # FIRST ITERATION
    v_true   = this_obj['v_guess']
    sig_true = this_obj['s_guess']
    v    = alldata['v'][crude_v]
    verr = alldata['v_err'][crude_v]

    if np.sum(crude_v) > 5:
        v_true,sig_true = mg_wmean(v,verr)
  
    #  V Prob PLUS PHOTOMETRIC ERROR
    texp     = np.array(alldata['v'] - v_true)**2 / (2*(alldata['v_err']**2 + (3.*sig_true)**2))
    Pmem_v = np.exp(-1.*texp)
 
    #  IF SUFFICIENT NUMBER OF MEMBERS, SECOND ITERATION
    Pmem_2 = Pmem_tmp * Pmem_v

    mnew = Pmem_2 > 0.2
    if np.sum(mnew) > 5:
        v    = alldata['v'][mnew]
        verr = alldata['v_err'][mnew]
        v_true,sig_true = mg_wmean(v,verr)

        #sampler, theta = dmost_vdisp.mcmc_vdisp(v,verr, v_true, sig_true,plot=0)
        #v_true,sig_true  = theta[0], theta[1]
        texp     = np.array(alldata['v'] - v_true)**2 / (2*(alldata['v_err']**2 + (3.*sig_true)**2))
        Pmem_v = np.exp(-1.*texp)
 


    return Pmem_v

def membership_feh(alldata, this_obj, Pmem_tmp):


    Pmem_feh = np.ones(np.size(Pmem_tmp))

    Pv_crude = membership_v_crude(alldata, this_obj)
    good_feh = (Pmem_tmp > 0.2) & (alldata['ew_feh_err'] > 0) &  (alldata['ew_feh_err'] < 10) & \
                                 (alldata['MV_o'] < 3.) & (alldata['tmpl_teff'] < 6500 ) & (Pv_crude == 1)

    if np.sum(good_feh) > 2:
        feh     = alldata['ew_feh'][good_feh]
        feh_err = alldata['ew_feh_err'][good_feh]

        # DETERMINE [FE/H] GUESSES
        feh_tmp, feh_sig_tmp = mg_wmean(feh,feh_err)
        print('{:0.2f}  {:0.2f}'.format(feh_tmp,feh_sig_tmp))

        if (feh_sig_tmp < 0.2):
            feh_sig_tmp = 0.2

        diff2 = (alldata['ew_feh'] - feh_tmp)**2
        texp  = np.array(diff2 / (2*(np.array(alldata['ew_feh_err'])**2 + (3.*feh_sig_tmp)**2)))

        # APPLY ONLY IN METAL-RICH DIRECTION
        mrich = (good_feh) & ((alldata['ew_feh'] - feh_tmp) > 0)
        Pmem_feh[mrich] = np.exp(-1.*texp[mrich])




    return Pmem_feh



######################################################
#def membership_MgI(alldata):
#    y = 0.26*alldata['ew_cat']  - 0.5206
#    mgI_lim1 = (alldata['ew_mgI'] - alldata['ew_mgI_err'] > 0.5)  & (alldata['ew_cat'] < 4)
#    mgI_lim2 = (alldata['ew_mgI'] > y)  & (alldata['ew_cat'] >= 4)
#    Pmem_MgI = ~(mgI_lim1 | mgI_lim2)
#    return Pmem_MgI


######################################################
# REMOVE FOREGROUND STARS WITH LARGE PROPER MOTIONS
# USE GUESS PM
def membership_PM(alldata, this_obj, Pmem_tmp):

    Pmem_pm = np.ones(np.size(Pmem_tmp))

    # IF THERE ARE ENOUGH MEASURED PMs
    m_pm    = (alldata['gaia_pmra_err'] > 0) & (alldata['gaia_pmdec_err'] > 0)
    mem_tmp = Pmem_tmp > 0.5
    if (np.sum(m_pm&mem_tmp) > 0):

        Pcrude_v        = membership_v_crude(alldata,this_obj)


        # FOR ALL STARS WITH PMs, CALC AN AVERGAGE WITH REJECTION
        pra_tmp  = sigma_clip(alldata['gaia_pmra'][mem_tmp&m_pm&Pcrude_v], sigma=2, maxiters=3)
        pdec_tmp = sigma_clip(alldata['gaia_pmdec'][mem_tmp&m_pm&Pcrude_v], sigma=2, maxiters=3,cenfunc='mean')
        
        pra_mean, pra_std  = np.mean(pra_tmp),np.std(pra_tmp)
        pdec_mean,pdec_std = np.mean(pdec_tmp),np.std(pdec_tmp)

        cos_dec  = np.cos((np.pi/180.)*this_obj['Dec'])
        diff2    = ((pra_mean - alldata['gaia_pmra'])*cos_dec)**2 + (pdec_mean - alldata['gaia_pmdec'])**2
        pm_sig   =  np.sqrt((pra_std*cos_dec)**2 + pdec_std**2)


        # REJECT STARS WITH ERROR CIRCLES FAR FROM PM CENTER
        pmerr = np.sqrt((alldata['gaia_pmra_err']*cos_dec)**2 + alldata['gaia_pmdec_err']**2)*u.arcsec


        pm_min = 2.0
        texp     = np.array(diff2 / (2*(np.array(pmerr)**2 + pm_min**2)))
        Pmem_pm[m_pm] = np.exp(-1.*texp[m_pm])


    return Pmem_pm





######################################################
def flag_HB_stars(alldata,this_obj):

    dist= this_obj['Dist_kpc']
    r_hb,gr_hb = plot_isochrone_HB(dist)


    dmin, emin = [],[]
    for star in alldata:
        err = np.sqrt(star['gmag_err']**2 + star['rmag_err']**2)

        d2 = (r_hb[1:] - star['rmag_o'])**2 + (gr_hb[1:] - (star['gmag_o'] - star['rmag_o']))**2
        tmp = np.min(d2)

        dmin.append(np.sqrt(tmp))
        emin.append(err)

    flag_HB = np.array(dmin) < np.sqrt(0.2**2 + np.array(emin)**2)

    
    return flag_HB


######################################################
def flag_variable_stars(alldata):

    flag_var = (alldata['flag_var'] ==1) | (alldata['var_short_flag'] ==1)

    return flag_var


######################################################
def flag_velocity_outliers(alldata,Pmem, low=3,high=3):

    x = alldata['v'][Pmem]
    c_std  = x.std()
    c_mean = x.mean()
    size = x.size
    critlower = c_mean - c_std * low
    critupper = c_mean + c_std * high

    Pmem_v = (alldata['v'] >= critlower) & (alldata['v'] <= critupper)
    
        
    return Pmem_v


######################################################
def flag_good_data(alldata):

    m_err = (alldata['v_err'] > 15) | (alldata['v_err'] < 0)

    # REMOVE SERENDIPS THAT DON"T PASS STRICTER QUALITY CUTS
    m_ser1 = (alldata['objname'] == 'SERENDIP') & (alldata['v_chi2'] > 50) & (alldata['SN'] < 50) 
    m_ser2 = (alldata['objname'] == 'SERENDIP') & (alldata['rmag_o'] < 0)

    flag_poor = m_err | m_ser1 | m_ser2
    flag_good = ~flag_poor

    return flag_good

######################################################
######################################################
def find_members(alldata,this_obj):
    

    flag_good       = flag_good_data(alldata)
    Pmem_cmd        = membership_CMD(alldata, this_obj)
    Pmem_EW         = membership_NaI(alldata, this_obj)
    Pmem_parallax   = membership_parallax(alldata, this_obj)


    all_flags1      = flag_good * Pmem_cmd  * Pmem_EW * Pmem_parallax
    Pmem_pm         = membership_PM(alldata, this_obj, all_flags1)


    all_flags2      = all_flags1 * Pmem_pm


    Pmem_feh        = membership_feh(alldata, this_obj,all_flags2)
    Pmem_v          = membership_vdisp(alldata,this_obj,all_flags2*Pmem_feh)


    # FLAG HB AND SET FEH TO ZERO
    flag_HB       = flag_HB_stars(alldata,this_obj)
    flag_var      = flag_variable_stars(alldata)

    Pmem_inclusive   =  all_flags2 * Pmem_v * Pmem_feh


    Pmem      =  Pmem_inclusive * ~flag_var 


    return Pmem_inclusive, Pmem#, Pmem_pm,Pmem_v,Pmem_feh



    

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

def set_cmd_lims(rmag):
    
    m=(rmag > 10)& (rmag < 30)
    rmag=rmag[m]
    
    try:
        cmin = np.max(rmag)+0.25
        cmax = np.min(rmag[rmag >0])-0.25
    except:
        cmin,cmax = 24,13

    if cmin > 24:
        cmin = 24
    if cmax < 13:
        cmax = 14

    return cmin,cmax

######################################################
def plot_isochrone_padova(dist,str_iso):

    iso = ascii.read(DEIMOS_RAW+'/Photometry/isochrones/iso_t12_z'+str(str_iso)+'.dat')

    #A(g)/(E(B-V)) = 3.793    
    #Ag = 3.793 * EBV
    #Ar = 2.751 * EBV

    r_iso = iso['rmag'] + 5.*np.log10(dist*1e3) - 5.  # + Ar
    gr_iso = iso['gmag'] - iso['rmag'] #+ (Ag - Ar)
    
    return r_iso,gr_iso



######################################################
def plot_isochrone_HB(dist):

    iso = Table.read(DEIMOS_RAW+'/Photometry/isochrones/M92_fiducial.dat',format='ascii',guess=False)
    hb = iso['typ'] == 1
    

    r_iso = iso['rmag'][hb] + 5.*np.log10(dist*1e3) - 5. 
    gr_iso = iso['gmr'][hb] + 0.2
    
    return r_iso,gr_iso 

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
def membership_v_crude(alldata,this_obj,cmd_mem=0,sig=3.):

    # REMOVE BAD FITS
    m_good = (alldata['v_chi2'] < 100) & (alldata['v_err'] < 25)


    # MIN MAX     
    min_v = this_obj['v_guess'] - this_obj['s_guess']*sig
    max_v = this_obj['v_guess'] + this_obj['s_guess']*sig

    Pmem_crude_v = (alldata['v'] > min_v) & (alldata['v'] < max_v) & m_good

    return Pmem_crude_v


######################################################
def membership_v(alldata,this_obj,cmd_mem=0,crude_cut=50.,sig=3.5):
    
    min_v = this_obj['v_guess'] - crude_cut
    max_v = this_obj['v_guess'] + crude_cut

    crude_v = (alldata['v'] > min_v) & (alldata['v'] < max_v)

    v = alldata['v'][crude_v]
    verr = alldata['v_err'][crude_v]
    if np.sum(cmd_mem) > 0:
        cm = (cmd_mem == 1)
        v = alldata['v'][cm&crude_v]
        verr = alldata['v_err'][cm&crude_v]

    sampler, theta = dmost_vdisp.mcmc_vdisp(v,verr, this_obj['v_guess'],5,plot=0)
    
    min_v = theta[0] - sig*theta[1]
    max_v = theta[0] + sig*theta[1]

    vmem = (alldata['v'] > min_v) & (alldata['v'] < max_v)
    if np.sum(cmd_mem) > 0:
        vmem = (alldata['v'] > min_v) & (alldata['v'] < max_v) & (cm)

    

    return Pmem_vmem

######################################################
def membership_NaI(alldata, this_obj):

    # ALL STARS
    Pmem_NaI = (alldata['ew_naI'] < 1.0) 

    # TIGHTER CONSTRAINT ON RGBs
    naI_lim2 = this_obj['NaI_lim']

    m_rmv    = ((alldata['ew_naI']-alldata['ew_naI_err']) > naI_lim2) & (alldata['MV_o'] < 4)
    Pmem_NaI[m_rmv] = 0

    return Pmem_NaI

######################################################
def membership_MgI(alldata):

    y = 0.26*alldata['ew_cat']  - 0.5206

    mgI_lim1 = (alldata['ew_mgI'] - alldata['ew_mgI_err'] > 0.5)  & (alldata['ew_cat'] < 4)
    mgI_lim2 = (alldata['ew_mgI'] > y)  & (alldata['ew_cat'] >= 4)


    Pmem_MgI = ~(mgI_lim1 | mgI_lim2)

    return Pmem_MgI

######################################################
def membership_parallax(alldata, this_obj):
    nstar         = np.size(alldata)
    Pmem_parallax = np.ones(nstar)

    obj_parallax  = 1./this_obj['Dist_kpc']
    diff_parallax = alldata['gaia_parallax'] - obj_parallax
    p_over_e = np.abs(alldata['gaia_parallax']/alldata['gaia_parallax_err'])

    # ONLY TEST STARS WITH AVAILABLE PARALLAX
    m_good   =(alldata['gaia_parallax'] > -100)

    # REMOVE STARS WITH LARGE MEASURED PARALLAX
    # diff > sig*error
    # diff > 0.5
    # parallax_over_error > sig
    sig = 5
    mrm = (np.abs(diff_parallax) > sig*alldata['gaia_parallax_err']) & (np.abs(diff_parallax) > 0.5) & (p_over_e > sig)
        
    Pmem_parallax = ~(mrm & m_good)


    return Pmem_parallax



######################################################
def membership_PM(alldata, Pmem):

    # IF THERE ARE ENOUGH MEASURED PMs
    m_pm    = alldata['gaia_pmra_err'] > 0
    Pmem_pm = Pmem
    if (np.sum(m_pm&Pmem) > 0):


        # FOR ALL STARS WITH PMs, CALC AN AVERGAGE WITH REJECTION
        ra_tmp  = sigma_clip(alldata['gaia_pmra'][Pmem&m_pm], sigma=2, maxiters=3)
        dec_tmp = sigma_clip(alldata['gaia_pmdec'][Pmem&m_pm], sigma=2, maxiters=3,cenfunc='mean')
        
        ra_mean, ra_std  = np.mean(ra_tmp),np.std(ra_tmp)
        dec_mean,dec_std= np.mean(dec_tmp),np.std(dec_tmp)

        # DETERMINE DISTANCE FROM PM CENTER  -- UNITS DON"T MATTER
        c_pm    = SkyCoord(ra_mean*u.arcsec,dec_mean*u.arcsec)
        c_stars = SkyCoord(alldata['gaia_pmra']*u.arcsec,alldata['gaia_pmdec']*u.arcsec) 
        
        # REJECT STARS WITH ERROR CIRCLES FAR FROM PM CENTER
        sep  = c_stars.separation(c_pm)
        m_pm_reject = sep > (3*np.sqrt(ra_std**2 + dec_std**2)*u.arcsec) + \
                        (u.arcsec*np.sqrt(alldata['gaia_pmra_err']**2 + alldata['gaia_pmdec_err']**2))
        
        Pmem_pm = ~(m_pm_reject)

    return Pmem_pm


######################################################
def membership_MgI(alldata):

    Pmem_MgI = (alldata['ew_mgI'] < 0.65)

    return Pmem_MgI

######################################################
def membership_CMD(alldata,this_obj):

    # GET ISOCHRONE PROPERTIES    
    dist= this_obj['Dist_kpc']
    iso = this_obj['iso_guess']

    r,gr       = plot_isochrone_padova(dist,iso)
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
            

    #  CMD THRESHOLD PLUS PHOTOMETRIC ERROR
    Pmem_cmd = np.array(dmin) < np.sqrt(0.2**2 + np.array(emin)**2)

    return Pmem_cmd


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
def flag_poor_data(alldata):

    m_err = alldata['v_err'] > 12

    m_vchi2 = (alldata['v_chi2'] > 25) & (alldata['SN'] < 50)

    flag_poor = m_err | m_vchi2 
        
    return flag_poor

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
######################################################
def find_members(alldata,this_obj):
    

    Pmem_cmd      = membership_CMD(alldata, this_obj)
    Pmem_crude_v  = membership_v_crude(alldata, this_obj)
    Pmem_NaI      = membership_NaI(alldata, this_obj)
    Pmem_MgI      = membership_MgI(alldata)
    Pmem_parallax = membership_parallax(alldata, this_obj)

    Pmem_pm       = membership_PM(alldata, Pmem_cmd&Pmem_crude_v&Pmem_NaI)

    # FLAG HB AND SET FEH TO ZERO
    flag_HB       = flag_HB_stars(alldata,this_obj)
    alldata['ew_feh'][(flag_HB ==1)] = -999.

    flag_var      = flag_variable_stars(alldata)
    flag_poor     = flag_poor_data(alldata)

    Pmem         =  Pmem_cmd & Pmem_crude_v & Pmem_parallax & Pmem_NaI & Pmem_MgI & Pmem_pm
    Pmem_v       = flag_velocity_outliers(alldata, Pmem&~flag_var&~flag_poor)

   

    Pmem_pure    =  Pmem & ~flag_var & Pmem_v & ~flag_poor


    return Pmem, Pmem_pure, Pmem_cmd, Pmem_crude_v, Pmem_NaI, Pmem_pm, Pmem_parallax, Pmem_MgI, Pmem_v



    

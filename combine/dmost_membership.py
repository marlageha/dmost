import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii, fits


from astropy.table import vstack
from scipy.stats import norm

import glob
import dmost_utils,dmost_vdisp

DEIMOS_RAW     = os.getenv('DEIMOS_RAW')


def mk_all_plots(alldata,this_obj, mem=0,Pmem=Pmem):

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(20,6))
    plt.rcParams.update({'font.size': 14})

    #############################
    # SPATIAL PLOT
    ax1.plot(alldata['RA'],alldata['DEC'],'ko',alpha=0.7,label='all targets')
    ax1.set_xlabel('RA [deg]')
    ax1.set_ylabel('DEC [deg]')

    half_light = plt.Circle((this_obj['RA'], this_obj['Dec']),this_obj['r_eff_arcm']/60., \
                            fill=None,color='g',label='half-light radius')
    ax1.add_patch(half_light)
    if (mem == 1):
        ax1.plot(alldata['RA'][Pmem],alldata['DEC'][Pmem],'ro',alpha=0.7,label='members')

    ax1.legend()

    #############################
    # CMD PLOT
    gr = alldata['gmag_o'] - alldata['rmag_o']

    ax2.plot(gr,alldata['rmag_o'],'ko',alpha = 0.7)
    ax2.set_ylim(23.5,14)
    ax2.set_xlim(-0.5,1.75)
    ax2.set_xlabel('(g-r)_o')
    ax2.set_ylabel('r_o')


    if (mem == 1):
        ax2.plot(gr[Pmem],alldata['rmag_o'][Pmem],'ro',alpha=0.7,label='members')

    r_iso,gr_iso = plot_isochrone_padova(this_obj['Dist_kpc'],this_obj['EBV_SF11'],this_obj['iso_guess'])
    ax2.plot(gr_iso,r_iso,'go',ms=3)

    #############################
    vmin = -150+this_obj['v_guess']
    vmax =  150+this_obj['v_guess']
    bn = np.arange(vmin,vmax,2.5)
    bins = [(bn[i]+bn[i+1])/2. for i in range(len(bn)-1)]

    N, x  = np.histogram(alldata['v'], bins=bn)
    ax3.fill_between(bins,N, step="pre", alpha=0.4,color='k')

    if (mem==1):                             
        N, x  = np.histogram(alldata['v'][Pmem], bins=bn)                             
        ax3.fill_between(bins,N, step="pre", alpha=0.4,color='r')

    ax3.set_xlabel('Velocity (km/s)')

    #############################
    #############################

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(20,6))
    vmin = -75+this_obj['v_guess']
    vmax =  75+this_obj['v_guess']

    #############################
    m_ew_na = np.abs(alldata['ew_naI_err']) < 5 
    ax1.plot(alldata['v'][m_ew_na],alldata['ew_naI'][m_ew_na],'ko',alpha=0.5)
    ax1.errorbar(alldata['v'][m_ew_na],alldata['ew_naI'][m_ew_na],yerr=alldata['ew_naI_err'][m_ew_na],fmt='ko',alpha=0.5)

    ax1.set_xlim(vmin,vmax)
    if (mem==1):                             
        ax1.plot(alldata['v'][m_ew_na&Pmem],alldata['ew_naI'][m_ew_na&Pmem],'ro',alpha=0.5)

    ax1.set_xlabel('v')
    ax1.set_ylabel('EW_NaI')

    #############################
    m_ew_na = np.abs(alldata['ew_feh_err']) < 5 
    ax2.plot(alldata['v'][m_ew_na],alldata['ew_feh'][m_ew_na],'ko',alpha=0.5)
    ax2.set_xlim(vmin,vmax)
    if (mem==1):                             
        ax2.plot(alldata['v'][m_ew_na&Pmem],alldata['ew_feh'][m_ew_na&Pmem],'ro',alpha=0.5)

    ax2.set_xlabel('v')
    ax2.set_ylabel('[Fe/H]_EW')

    #############################
    ax3.plot(alldata['rproj_kpc'],alldata['v'],'ko',alpha=0.5)
    ax3.errorbar(alldata['rproj_kpc'],alldata['v'],yerr = alldata['v_err'],fmt='.',alpha=0.5)

    ax3.set_ylim(vmin,vmax)
    if (mem==1):                             
        ax3.plot(alldata['rproj_kpc'][Pmem],alldata['v'][Pmem],'ro',alpha=0.5)
        ax3.axvline(this_obj['r_eff_arcm']/60.,color='g')

    ax3.set_ylabel('v')
    ax3.set_ylabel('projected radius [kpc]')




######################################################
def plot_isochrone_padova(dist,iso):

    iso = ascii.read(DEIMOS_RAW+'/Photometry/isochrones/iso_t12_z'+str(iso)+'.dat')


    r_iso = iso['rmag'] + 5.*np.log10(dist*1e3) - 5.  # + Ar
    gr_iso = iso['gmag'] - iso['rmag'] #+ (Ag - Ar)
    
    return r_iso,gr_iso


######################################################
def membership_CMD(zspec,obj):

    # GET ISOCHRONE PROPERTIES    
    dist= obj['Dist_kpc']
    iso = obj['iso_guess']

    r,gr = plot_isochrone_padova(dist*1e3,iso)

        
    dmin = []
    emin = []
    for star in zspec:
        err = np.sqrt(star['gmag_err']**2 + star['rmag_err']**2)

        d = (r - star['rmag_o'])**2 + (gr - (star['gmag_o'] - star['rmag_o']))**2
        #d2 = (r_hb - star['RMAG'])**2 + (gr_hb - (star['gmag_o'] - star['rmag_o']))**2

        tmp = np.min(d)
        #tmp2=np.min(d2)        
        #if tmp2 < tmp:
        #    tmp=tmp2
        dmin.append(tmp)  
        emin.append(err)
    
    emin = np.array(emin)
    m    =  (emin > 1) 
    emin[m] = 0
    m    =  (emin > 0.3) & (star['rmag_err'] < 22)   #. account for bad errors in UMa2    
    emin[m] = 0
    
    #*********************************
    # CMD THRESHOLD == 0.1 PLUS ERRORS
    mem = np.array(dmin) < 0.1 + emin   # SET THRESHOLD PLUS PHOTOMETRIC ERROR
    
    return mem





    
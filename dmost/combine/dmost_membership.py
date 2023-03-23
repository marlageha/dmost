import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii, fits


from astropy.table import vstack
from scipy.stats import norm

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

    
    #print('Vmin Vmax = {:0.2f}, {:0.2f}'.format(theta[0] - sig*theta[1],theta[0] + sig*theta[1]))

    return vmem

######################################################
def membership_CMD(zspec,obj):

    # GET ISOCHRONE PROPERTIES    
    dist= obj['Dist_kpc']
    iso = obj['iso_guess']

    r,gr = plot_isochrone_padova(dist,iso)
    r_hb,gr_hb = plot_isochrone_HB(dist)

        
    dmin = []
    emin = []
    mem = []
    for star in zspec:
        err = np.sqrt(star['gmag_err']**2 + star['rmag_err']**2)

        d = (r - star['rmag_o'])**2 + (gr - (star['gmag_o'] - star['rmag_o']))**2
        d2 = (r_hb - star['rmag_o'])**2 + (gr_hb - (star['gmag_o'] - star['rmag_o']))**2

        tmp = np.min(d)
        tmp2=np.min(d2)        
        if tmp2 < tmp:
            tmp=tmp2
        
        dmin.append(np.sqrt(tmp))#+err)  
        emin.append(err)
        

        
    m_good = np.array(dmin) < (0.2)# + emin)
    
    #*********************************
    # CMD THRESHOLD == 0.1 PLUS ERRORS
    mem = np.zeros(np.size(dmin))
    m_good = np.array(dmin) < (0.2)
    mem[m_good] = 1
    
    
    return mem,dmin
    
def find_members(alldata,this_obj):
    
    cmd_mem,dmin = membership_CMD(alldata,this_obj)
    v_mem        = membership_v(alldata,this_obj,cmd_mem=cmd_mem,sig=3)
    Pmem         = (cmd_mem == 1) & v_mem & (alldata['ew_naI'] < 1.5)
    Pmem         = np.multiply(Pmem, 1)
    return Pmem



    

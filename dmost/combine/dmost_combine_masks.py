import numpy as np
import os
import glob

from astropy import table
from astropy.table import Table,Column
from astropy import units as u
from astropy.io import ascii,fits

from astropy.coordinates import SkyCoord

from dmost import dmost_utils
from dmost.combine import dmost_photometry_gaia, dmost_membership
from scipy import stats



###################################
# CREATE ALLSPEC DATA STRUCTURE
def create_allstars(nmasks,nstars):

    cols = [filled_column('objname','                  ',nstars),
            filled_column('RA',-999.,nstars),
            filled_column('DEC',-999.,nstars),
            filled_column('rproj_arcm',-999.,nstars),
            filled_column('rproj_kpc',-999.,nstars),


            # INDIVIDUAL MASK PROPERTIES
            filled_column('nexp',-99,nstars),
            filled_column('nmask',-99,nstars),
            filled_column('masknames','                                                  ',nstars),

         
            # COMBINED FINAL PROPERTIES
            filled_column('v',-999.,nstars),
            filled_column('v_err',-999.,nstars),
            filled_column('SN',-999.,nstars),


            # GALAXIES
            filled_column('marz_z',-999.,nstars),
            filled_column('marz_flag',-99,nstars),

            filled_column('serendip',-999.,nstars),




            # PHOTOMETRY
            filled_column('gmag_o',-999.,nstars),
            filled_column('rmag_o',-999.,nstars),
            filled_column('gmag_err',-999.,nstars),
            filled_column('rmag_err',-999.,nstars),
            filled_column('EBV',-999.,nstars),

            filled_column('MV_o',-999.,nstars),


            # EQUIVALENT WIDTHS
            filled_column('ew_cat',-999.,nstars),
            filled_column('ew_cat_err',-999.,nstars),

            filled_column('ew_naI',-999.,nstars),
            filled_column('ew_naI_err',-999.,nstars),

            filled_column('ew_mgI',-999.,nstars),
            filled_column('ew_mgI_err',-999.,nstars),

            filled_column('ew_feh',-999.,nstars),
            filled_column('ew_feh_err',-999,nstars),

            filled_column('tmpl_teff',-999.,nstars),
            filled_column('tmpl_feh',-999.,nstars),


            # GAIA
            filled_column('gaia_pmra',-999.,nstars),
            filled_column('gaia_pmra_err',-999.,nstars),            
            filled_column('gaia_pmdec',-999.,nstars),
            filled_column('gaia_pmdec_err',-999.,nstars),
            filled_column('gaia_pmra_pmdec_corr',-999.,nstars),
            filled_column('gaia_parallax',-999.,nstars),
            filled_column('gaia_parallax_over_err',-999.,nstars),
            filled_column('gaia_aen',-999.,nstars),
            filled_column('gaia_aen_sig',-999.,nstars),
            filled_column('gaia_rv',-999.,nstars),
            filled_column('gaia_rv_err',-999.,nstars),

            filled_column('gaia_flag',-99,nstars),


            # VARIABLE VELOCITY FLAGS
            filled_column('var_pval',-999.,nstars),
            filled_column('var_max_v',-999.,nstars),
            filled_column('var_max_t',-999.,nstars),
            filled_column('var_flag',-99,nstars),

            filled_column('var_short_flag',-99,nstars),
            filled_column('var_short_max_t',-999.,nstars),

            # MEMBERSHIPS
            filled_column('prob_member',-99,nstars),


            # INDIVUDAL MASK DATA
            filled_column('mask_v',-999.*np.ones(nmasks),nstars),
            filled_column('mask_v_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_nexp',-999.*np.ones(nmasks),nstars),
            filled_column('mask_SN',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mjd',-999.*np.ones(nmasks),nstars),

            filled_column('mask_marz_z',-999.*np.ones(nmasks),nstars),
            filled_column('mask_marz_flag',-99*np.ones(nmasks),nstars),
            filled_column('mask_marz_tmpl',-999.*np.ones(nmasks),nstars),

            filled_column('mask_teff',-999.*np.ones(nmasks),nstars),
            filled_column('mask_logg',-999.*np.ones(nmasks),nstars),
            filled_column('mask_feh',-999.*np.ones(nmasks),nstars),

            filled_column('mask_cat',-999.*np.ones(nmasks),nstars),
            filled_column('mask_naI',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mgI',-999.*np.ones(nmasks),nstars),
            filled_column('mask_cat_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_naI_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mgI_err',-999.*np.ones(nmasks),nstars),
            
            filled_column('mask_var_short_flag',-999.*np.ones(nmasks),nstars),
            filled_column('mask_var_short_max_t',-999.*np.ones(nmasks),nstars),

           ]
            
    slits = Table(cols)
    return slits



######################################################
def deimos_google():
    key = '1V2aVg1QghpQ70Lms40zNUjcCrycBF2bjgs-mrp6ojI8'
    gid=1906496323
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    masklist = Table.read(url, format='csv')

    gid =0
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    objlist = ascii.read(url, format='csv')
    
    return objlist,masklist


######################################################
# FILL A COLUMN
def filled_column(name, fill_value, size):
    """
    Tool to allow for large strings
    """
    return Column([fill_value]*int(size), name)


######################################################
def combine_mask_velocities(stars, sys_mask = 0.7):
    
    # COMBINE STARS WITH MEASURED VELOCITIES
    mgood       = stars['dmost_v_err'] > 0.
    good_stars  = stars[mgood]
    
    v,verr_sys,teff,feh,ncomb = [-999.,-99.,-99.,-99.,0]
    if (np.size(good_stars) == 1):
        v     = good_stars['dmost_v']
        teff  = good_stars['chi2_teff']
        feh   = good_stars['chi2_feh']

        verr    = np.sqrt(good_stars['dmost_v_err']**2)
        verr_sys = np.sqrt(verr**2 + sys_mask**2)
        ncomb=ncomb+1

    
    if np.size(good_stars) > 1:
        vt,et,tt,ft = [],[],[],[]
        for obj in good_stars:
            vt   = np.append(vt,obj['dmost_v'])
            tt   = np.append(tt,obj['chi2_teff'])
            ft   = np.append(ft,obj['chi2_feh'])
            et   = np.append(et,obj['dmost_v_err'])
            ncomb=ncomb+1

        sum1 = np.sum(1./et**2)
        sum2 = np.sum(vt/et**2)
        sum3 = np.sum(tt/et**2)
        sum4 = np.sum(ft/et**2)

        v    = sum2/sum1
        teff = sum3/sum1
        feh  = sum4/sum1

        verr     = np.sqrt(1./sum1)
        verr_sys = np.sqrt(verr**2 + sys_mask**2)

    return v, verr_sys, teff, feh, ncomb

#####################################
#####################################
def set_binary_flag(alldata,sys_mask = 0.7):


    ns, nvar = 0,0
    for i,obj in enumerate(alldata):
        

        # FIRST SET INNER MASK VARIABLE FLAG
        mnv = any(obj['mask_var_short_flag'] == 0)
        if (np.sum(mnv) > 0):
            mxt = np.max(obj['mask_var_short_max_t'][mnv])
            alldata['var_short_max_t'][i]= mxt
            alldata['var_short_flag'][i] = 0
        

        mvs       = obj['mask_var_short_flag'] == 1
        if  (np.sum(mvs) > 0):
            mxt = np.max(obj['mask_var_short_max_t'][mvs])
            alldata['var_short_max_t'][i]= mxt
            alldata['var_short_flag'][i] = 1
        


        m = (obj['mask_v_err'] > 0)
        if np.sum(m) > 1:

            alldata['var_flag'][i]  = 0
            ns=ns+1

            v_mean = np.average(obj['mask_v'][m],weights=1./(obj['mask_v_err'][m]**2+ sys_mask**2))
            chi2   = np.sum((obj['mask_v'][m] - v_mean)**2/(obj['mask_v_err'][m]**2 + sys_mask**2))
            pv     = 1 - stats.chi2.cdf(chi2, np.sum(m)-1.)


#            if (pv < 1e-14):
#                fig, (ax1) = plt.subplots(1, 1,figsize=(8,5))
#                plt.errorbar(obj['mask_mjd'][m],obj['mask_v'][m],fmt='.',yerr = obj['mask_v_err'][m])
#                plt.title('{} {} {}'.format(obj['objname'],obj['mask_SN'][m],obj['masknames']))

            if (pv == 0) | (pv < 1e-14):
                pv = 1e-14

            lpv = np.log10(pv)

            alldata['var_pval'][i]  = lpv
            alldata['var_max_v'][i] = np.max(obj['mask_v'][m]) - np.min(obj['mask_v'][m])
            alldata['var_max_t'][i] = 24*(np.max(obj['mask_mjd'][m])-np.min(obj['mask_mjd'][m]))
            alldata['var_flag'][i]  = 0

            if lpv < -6:
                alldata['var_flag'][i]  = 1
                nvar = nvar + 1

    print('VVAR: Setting {} of {} repeats as velocity variable'.format(nvar,ns))

    return alldata



#####################################
def combine_mask_ew(stars):
    
    
    # COMBINE STARS WITH MEASURED VELOCITIES
    mgood       = stars['dmost_v_err'] > 0.
    good_stars  = stars[mgood]
    
    cat,cat_err,naI,naI_err,mgI,mgI_err, ncomb = [-99.,-99.,-99.,-99.,-99.,-99.,0]
    if (np.size(good_stars) == 1):
        cat      = good_stars['cat']
        cat_err  = good_stars['cat_err']
        naI      = good_stars['naI']
        naI_err  = good_stars['naI_err']
        mgI      = good_stars['mgI']
        mgI_err  = good_stars['mgI_err']

        ncomb=ncomb+1

    
    if np.size(good_stars) > 1:
        ct,cterr,na,naerr,mg,mgerr = [],[],[],[],[],[]
        for obj in good_stars:
            ct   = np.append(ct,obj['cat'])
            na   = np.append(na,obj['naI'])
            mg   = np.append(mg,obj['mgI'])
            
            cterr   = np.append(cterr,obj['cat_err'])
            naerr   = np.append(naerr,obj['naI_err'])
            mgerr   = np.append(mgerr,obj['mgI_err'])

            ncomb=ncomb+1
            
        sum1 = np.sum(1./cterr**2)
        sum2 = np.sum(ct/cterr**2)
        
        sum1n = np.sum(1./naerr**2)
        sum2n = np.sum(na/naerr**2)
        
        sum1m = np.sum(1./mgerr**2)
        sum2m = np.sum(mg/mgerr**2)
        
      
        cat = sum2/sum1
        naI = sum2n/sum1n
        mgI = sum2m/sum1m

        cat_err = np.sqrt(1./sum1)
        naI_err = np.sqrt(1./sum1n)
        mgI_err = np.sqrt(1./sum1m)

    return cat,cat_err,mgI,mgI_err,naI,naI_err, ncomb 



###########################################
def combine_mask_marz(star):
    '''
    Templates QUASAR     = 12
              Elliptical = 6
    '''
  
    mset       = star['marz_flag'] > -1
    marz_obj   = star[mset]
    marz_flag, marz_z, marz_tmpl = -99,-99,-99

    # COPY SINGLE MEASUREMENT
    if (np.size(marz_obj) == 1):
        marz_z    = marz_obj['marz_z'][0]
        marz_flag = marz_obj['marz_flag'][0]


    # PARSE MULTIPLE MEASUREMENTS
    if (np.size(marz_obj) >1):


        if np.any(marz_obj['marz_flag'] == 2):
            marz_z    = 2

        # IF ANY EXP IS QSO, SET AS 6
        if np.any(marz_obj['marz_flag'] == 3):
            marz_z    = 3

        # IF ANY EXP IS GOOD GALAXY, SET AS 4
        if np.any(marz_obj['marz_flag'] == 4):
            marz_z    = 4

        # IF ANY EXP IS QSO, SET AS 6
        if np.any(marz_obj['marz_flag'] == 6):
            marz_z    = 6

        if np.all(marz_obj['marz_flag'] == 1):
            marz_flag = 1
            
            
        # ELSE AVERAGE THE FLAGS
        if (marz_flag == -1):
            marz_flag = np.mean(marz_obj['marz_flag'])
            
    return marz_z, marz_flag





###########################################
def read_dmost_files(masklist):
    n=0

    data_dir = os.getenv('DEIMOS_REDUX')
    allslits = []

    for msk in masklist:

        maskname = msk['MaskName']        
        dmost_file = glob.glob(data_dir + '*'+maskname+'/dmost/dmost*'+maskname+'.fits')
        if np.size(dmost_file) > 0:

            slits, mask = dmost_utils.read_dmost(dmost_file[0])
            nexp = mask['nexp'][0]

            nslits = np.size(slits)
            maskname = filled_column('maskname',maskname,nslits)

            mjd      = filled_column('mjd',np.mean(mask['mjd']),nslits)
            nexp     = filled_column('nexp',nexp,nslits)

            # KEEP ONLY MASK AVERAGED QUANTITIES
            new_slits = Table([maskname, mjd,nexp,slits['objname'],slits['RA'],slits['DEC'],slits['collate1d_filename'],\
                             slits['collate1d_SN'], slits['marz_z'],slits['marz_flag'],slits['serendip'],\
                             slits['chi2_teff'],slits['chi2_logg'],slits['chi2_feh'],\
                             slits['dmost_v'],slits['dmost_v_err'],slits['v_nexp'],\
                             slits['vv_short_pval'], slits['vv_short_max_v'],slits['vv_short_max_t'],slits['vv_short_flag'],\
                             slits['cat'],slits['cat_err'],\
                             slits['naI'],slits['naI_err'],\
                             slits['mgI'],slits['mgI_err']])
                             
                            
            # CREATE OR APPEND TO ALL TABLE
            if (n==0):  allslits = new_slits
            if (n > 0): allslits = table.vstack([allslits,new_slits])
            n=n+1
        else:
            print('Skipping mask {}'.format(msk['MaskName']))



    return allslits, n

###########################################
def get_unique_spectra(allslits):

    cdeimos = SkyCoord(ra=allslits['RA']*u.degree, dec=allslits['DEC']*u.degree) 
    idx, d2d, d3d = cdeimos.match_to_catalog_sky(cdeimos)  
    mt = d2d < 1.5*u.arcsec

    nstars = np.size(allslits)

    return nstars

###########################################
def combine_mask_quantities(nmasks, nstars, sc_gal, allslits):

    # CREATE DATA TABLE
    single_mask =0
    if (nmasks ==1):
        nmasks = 2
        single_mask =1
    dmost_allstar  = create_allstars(nmasks, nstars) 

    test_allslits  = allslits.copy()
    for i,obj in enumerate(test_allslits):

        if (obj['RA'] != -99):
            
            dmost_allstar['RA'][i] = obj['RA']
            dmost_allstar['DEC'][i] = obj['DEC']
            dmost_allstar['objname'][i] = obj['objname']
            if (obj['serendip'] > 0):
                dmost_allstar['serendip'][i] = 1
            else:
                dmost_allstar['serendip'][i] = 0

            # FIND REPEAT SPECTRA
            ra_diff  = (obj['RA']  - allslits['RA']) * np.cos(obj['DEC']  * np.pi/180.) 
            dec_diff = (obj['DEC'] - allslits['DEC'])
            diff     = 3600.*np.sqrt(ra_diff**2 + dec_diff**2)
            
            
            # SET MATCHING THRESHOLD -- 1.5" arcseconds
            m = diff < 1.5
            
            nrpt = np.sum(m)
            for j,robj in enumerate(test_allslits[m]):

                if (j==0):
                    dmost_allstar['masknames'][i]   = obj['maskname']
                if (j > 0):
                    dmost_allstar['masknames'][i]   = dmost_allstar['masknames'][i]+'+'+robj['maskname']

                c1 = (j == 0) 
                c2 = (j > 0) & (single_mask ==0)

                if (c1 | c2):

                    dmost_allstar['mask_v'][i,j]     = robj['dmost_v']
                    dmost_allstar['mask_v_err'][i,j] = robj['dmost_v_err']
                    dmost_allstar['mask_SN'][i,j]    = robj['collate1d_SN']
                    dmost_allstar['mask_nexp'][i,j]  = robj['v_nexp']
                    dmost_allstar['mask_mjd'][i,j]   = robj['mjd']

                    dmost_allstar['mask_marz_flag'][i,j] = robj['marz_flag']
                    dmost_allstar['mask_marz_z'][i,j]    = robj['marz_z']

                    dmost_allstar['mask_teff'][i,j]  = robj['chi2_teff']
                    dmost_allstar['mask_logg'][i,j]  = robj['chi2_logg']
                    dmost_allstar['mask_feh'][i,j]   = robj['chi2_feh']

                    dmost_allstar['mask_cat'][i,j]   = robj['cat']
                    dmost_allstar['mask_naI'][i,j]   = robj['naI']
                    dmost_allstar['mask_mgI'][i,j]   = robj['mgI']
                    dmost_allstar['mask_cat_err'][i,j]  = robj['cat_err']
                    dmost_allstar['mask_naI_err'][i,j]  = robj['naI_err']
                    dmost_allstar['mask_mgI_err'][i,j]  = robj['mgI_err']
                  
                    dmost_allstar['mask_var_short_flag'][i,j]  = robj['vv_short_flag']
                    dmost_allstar['mask_var_short_max_t'][i,j] = robj['vv_short_max_t']

            # COMBINE VELOCITIES      

            v, verr, teff, feh, ncomb = combine_mask_velocities(test_allslits[m])
            dmost_allstar['v'][i]     = v
            dmost_allstar['v_err'][i] = verr
            dmost_allstar['nmask'][i] = nrpt
            dmost_allstar['nexp'][i]  = np.sum(test_allslits['nexp'][m])

            dmost_allstar['tmpl_teff'][i]  = teff
            dmost_allstar['tmpl_feh'][i]   = feh

            # COMBINE EW 
            cat,cat_err,mgI,mgI_err,naI,naI_err, ncomb = combine_mask_ew(test_allslits[m])
            dmost_allstar['ew_cat'][i]       = cat
            dmost_allstar['ew_cat_err'][i]   = cat_err
            dmost_allstar['ew_mgI'][i]       = mgI
            dmost_allstar['ew_mgI_err'][i]   = mgI_err
            dmost_allstar['ew_naI'][i]       = naI
            dmost_allstar['ew_naI_err'][i]   = naI_err

            # COMBINE MARZ
            zgal, zflag  = combine_mask_marz(test_allslits[m])
            dmost_allstar['marz_z'][i]   = zgal
            dmost_allstar['marz_flag'][i] = zflag

            # SET EXTRAGALACTIC VALUES
            mgal  = dmost_allstar['marz_flag']  > 2
            dmost_allstar['v'][mgal]     =  dmost_allstar['marz_z'][mgal]*3e5  
            dmost_allstar['v_err'][mgal] =  0


            # CALCULATE SN -- IS THIS RIGHT?
            msn = dmost_allstar['mask_SN'][i,:] > -1
            dmost_allstar['SN'][i] = np.sum(dmost_allstar['mask_SN'][i,msn]) /np.sqrt(np.size(dmost_allstar['mask_SN'][i,msn]))

            test_allslits['RA'][m] = -99
            
        
        
    # REMOVE EXTRA LINES
    m=dmost_allstar['RA'] != -999.0
    dmost_allstar = dmost_allstar[m]

    return dmost_allstar


######################################################
def combine_masks(object_name, **kwargs):


    DEIMOS_REDUX  = os.getenv('DEIMOS_REDUX')
    outfile       = DEIMOS_REDUX + '/dmost_alldata/dmost_alldata_'+object_name+'.fits'    
    if not ('objlist' in kwargs):
        objlist, masklist = deimos_google()
    else:
        objlist  = kwargs['objlist']
        masklist = kwargs['masklist']

    # PROPERTIES OF OBJECT
    object_properties = objlist[(objlist['Name2'] == object_name)]
    sc_gal            = SkyCoord(object_properties['RA'],object_properties['Dec'], unit=(u.deg, u.deg))


    # MASK LIST FOR OBJECT
    mobj   = (masklist['Object'] == object_name) #& (masklist['dmost'] == 1)
    nmasks = np.sum(mobj)
    print('{} Combining {} masks'.format(object_name,nmasks))


    # READ AND COMBINE ALL DMOST FILES
    alldata, nrun = read_dmost_files(masklist[mobj])
    if (nrun == 0):
        print('No masks run for this object, skipping')
        alldata=[]
        return alldata


    # HOW MANY UNIQUE SPECTRA?
    nstars = get_unique_spectra(alldata)
   

    # CREATE AND POPULATE FINAL DATA TABLE
    alldata  = combine_mask_quantities(nmasks, nstars, sc_gal, alldata)


    # SET BINARY FLAGS
    alldata  = set_binary_flag(alldata)


    # MATCH PHOTOMETRY, MATCH GAIA
    alldata = dmost_photometry_gaia.match_photometry(object_properties[0],alldata)
    alldata = dmost_photometry_gaia.match_gaia(object_properties[0],alldata)


    # CRUDE MEMBERSHIP
#    alldata['prob_member'] = dmost_membership.find_members(alldata,object_properties[0])


    # REMOVE MASK-LEVEL DATA (UPDATE LATER?)
    alldata.remove_columns(['mask_v','mask_v_err','mask_nexp','mask_SN','mask_mjd',\
                            'mask_marz_z','mask_marz_flag','mask_marz_tmpl',\
                            'mask_teff','mask_feh','mask_logg',\
                            'mask_cat','mask_cat_err','mask_naI','mask_naI_err','mask_mgI','mask_mgI_err',\
                            'mask_var_short_flag','mask_var_short_max_t'])


    print('{} Combined {} masks with {} unique stars'.format(object_name,nmasks,nstars))
    alldata.write(outfile, overwrite=True)
            

    return alldata


def combine_all():

    objlist, masklist = deimos_google()
    for obj in objlist:
        
        if obj['Phot'] != 'PanS':
            tmp  = combine_masks(obj['Name2'])



#####################################################    
def main(*args):


    mask = sys.argv[1]
    
    DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')
    
    alldata = combine_masks(object_name)
    
if __name__ == "__main__":
    main()
    

#!/usr/bin/env python

import numpy as np
import os,sys
import glob

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


from astropy.table import Table,Column,vstack
from astropy.io import ascii,fits
from astropy.time import Time
import astropy.units as u

import pyspherematch as sm

from dmost import dmost_utils
from dmost.core.dmost_utils import printlog


DEIMOS_RAW     = os.getenv('DEIMOS_RAW')


######################################################
# FILL A COLUMN
def filled_column(name, fill_value, size):
    """
    Tool to allow for large strings
    """
    return Column([fill_value]*int(size), name)


###################################
# CREATE MASK STRUCTURE
def create_mask(nexp):

    cols = [filled_column('maskname','        ',nexp), 
            filled_column('mask_ra',-1.,nexp),
            filled_column('mask_dec',-1.,nexp),
            filled_column('spec1d_filename','                                                                           ',nexp),
            filled_column('rawfilename','                            ',nexp), 
            filled_column('deimos_maskname','                 ',nexp), 

            filled_column('fname','        ',nexp),
            filled_column('mjd',-1.,nexp),
            filled_column('year','    ',nexp),
            filled_column('exptime',-1.,nexp),
            filled_column('nexp',-1,nexp),
            filled_column('vhelio',-1.,nexp),

            # SEEING AND SEEING-BASED CORRECTION TO LSF
            filled_column('airmass',-1.,nexp),
            filled_column('slitwidth',-1.,nexp),
            filled_column('seeing',-1.,nexp),
            filled_column('lsf_correction',-1.,nexp),  

            # dmost VALUES AND PROGRESS FLAGS
            filled_column('telluric_h2o',-1.,nexp),
            filled_column('telluric_o2',-1.,nexp),

            filled_column('flag_flexure',-1,nexp),
            filled_column('flag_telluric',-1,nexp),
            filled_column('flag_template',-1,nexp),
            filled_column('flag_emcee',-1,nexp),
            filled_column('flag_coadd',-1,nexp)
            
           ]

 
    mask = Table(cols)

    return mask


###################################
# CREATE ALLSPEC DATA STRUCTURE
def create_slits(nslits,nexp):

    cols = [filled_column('objname','                ',nslits),
            filled_column('objid','                ',nslits),
            filled_column('RA',-1.,nslits),
            filled_column('DEC',-1.,nslits),
            filled_column('slitname',['                       ']*nexp,nslits),

            # REDUCTION FLAG
            filled_column('flag_skip_exp',-1*np.ones(nexp,dtype='int'),nslits),

            # PYPEIT DETECTOR POSITION, USE TO READ SPEC1D FILES            
            filled_column('spat_pixpos',np.zeros(nexp),nslits),
            filled_column('det',['     ']*nexp,nslits),
            filled_column('serendip',-1,nslits),

            # SLIT INFO
            filled_column('rms_arc',-1.,nslits),
            filled_column('collate1d_filename','                                                  ',nslits),
            filled_column('collate1d_SN',-1.,nslits),

            # CHIP GAP + SEEING
            filled_column('opt_fwhm',np.zeros(nexp),nslits),
            filled_column('chip_gap_b',np.zeros(nexp),nslits),
            filled_column('chip_gap_r',np.zeros(nexp),nslits),
            filled_column('chip_gap_corr',np.ones(nexp),nslits),
            filled_column('chip_gap_corr_collate1d',1.,nslits),
                     
            # MARZ
            filled_column('marz_flag',-1,nslits),
            filled_column('marz_z',-1.,nslits),
            filled_column('marz_tmpl',-1.,nslits),

            # FLEXURE
            filled_column('SN',np.zeros(nexp),nslits),
            filled_column('fit_slope',np.zeros(nexp),nslits),
            filled_column('fit_b',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p0',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p1',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p2',np.zeros(nexp),nslits),
            filled_column('fit_lsf',np.zeros(nexp),nslits),
            filled_column('fit_lsf_corr',np.zeros(nexp),nslits),
            filled_column('rms_sky',np.zeros(nexp),nslits),
 

            # TELLURIC
            filled_column('telluric_h2o',np.zeros(nexp),nslits),
            filled_column('telluric_o2',np.zeros(nexp),nslits),
            filled_column('telluric_h2o_err',np.zeros(nexp),nslits),
            filled_column('telluric_o2_err',np.zeros(nexp),nslits),
            filled_column('telluric_w',np.zeros(nexp),nslits),
            filled_column('telluric_chi2',np.zeros(nexp),nslits),

          

            # CHI2 TEMPLATE
            filled_column('chi2_tfile','                                    ',nslits),
            filled_column('chi2_tgrid','      ',nslits),
            filled_column('chi2_tchi2',-1.,nslits),
            filled_column('chi2_v',-1.,nslits),
            filled_column('chi2_teff',-1,nslits),
            filled_column('chi2_logg',-1.,nslits),
            filled_column('chi2_feh',-1.,nslits),

            # EMCEE
            filled_column('emcee_v',-99.*np.ones(nexp),nslits),
            filled_column('emcee_w',-99.*np.ones(nexp),nslits),
            filled_column('emcee_v_err',-99.*np.ones(nexp),nslits),
            filled_column('emcee_w_err',-99.*np.ones(nexp),nslits),
            filled_column('emcee_v_err16',-99.*np.ones(nexp),nslits),
            filled_column('emcee_v_err84',-99.*np.ones(nexp),nslits),
            filled_column('emcee_w_err16',-99.*np.ones(nexp),nslits),
            filled_column('emcee_w_err84',-99.*np.ones(nexp),nslits),
            filled_column('emcee_f_acc',-99.*np.ones(nexp),nslits),
            filled_column('emcee_nsamp',-99.*np.ones(nexp),nslits),
            filled_column('emcee_burnin',-99.*np.ones(nexp,dtype=int),nslits),
            filled_column('emcee_converge',-99.*np.ones(nexp),nslits),
            filled_column('emcee_lnprob',-99.*np.ones(nexp),nslits),
            filled_column('emcee_skew',-99*np.ones(nexp),nslits),
            filled_column('emcee_kertosis',-99*np.ones(nexp),nslits),
            filled_column('emcee_good',-1*np.ones(nexp),nslits),

   
            # LOW SN V
            filled_column('coadd_v',-999.,nslits),
            filled_column('coadd_w',-99.,nslits),
            filled_column('coadd_v_err',-99.,nslits),
            filled_column('coadd_w_err',-99.,nslits),
            filled_column('coadd_v_err16',-99.,nslits),
            filled_column('coadd_v_err84',-99.,nslits),
            filled_column('coadd_f_acc',-99.,nslits),
            filled_column('coadd_converge',-99.,nslits),
            filled_column('coadd_burnin',-99,nslits),
            filled_column('coadd_nsamp',-99.,nslits),
            filled_column('coadd_lnprob',-99.,nslits),
            filled_column('coadd_skew',-99.,nslits),
            filled_column('coadd_kertosis',-99.,nslits),
            filled_column('coadd_good',-99,nslits),
            filled_column('coadd_flag',-99,nslits),

            # COMBINED VELOCITIES
            filled_column('dmost_v',-999.,nslits),
            filled_column('dmost_v_err',-99.,nslits),
            filled_column('dmost_v_err_rand',-99.,nslits),
            filled_column('v_nexp',-99,nslits),

            # SHORT BINARY FLAGS
            filled_column('var_short_pval',-99.,nslits),
            filled_column('var_short_max_v',-99.,nslits),
            filled_column('var_short_max_t',-99.,nslits),
            filled_column('flag_short_var',-99,nslits),


           # EQUIVALENT WIDTHS FROM COADD1D FILES
            filled_column('cat',-99.,nslits),
            filled_column('cat_err',-99.,nslits),
            filled_column('cat_all',-99.*np.ones(3),nslits),
            filled_column('cat_all_err',-99.*np.ones(3),nslits),
            filled_column('cat_chi2',-99.,nslits),
            filled_column('cat_gl',-99.,nslits),
            filled_column('naI',-99.,nslits),
            filled_column('naI_err',-99.,nslits),
            filled_column('mgI',-99.,nslits),
            filled_column('mgI_err',-99.,nslits),
            filled_column('mgI_err_rand',-99.,nslits),
            filled_column('naI_err_rand',-99.,nslits),
            filled_column('cat_err_rand',-99.,nslits)

           ]
            
    slits = Table(cols)
    return slits


#############################################################
def dmost_parse_telluric(tfile,fname):
    '''
    Given a telluric template (tfile) created for given exposure (fname), 
    pull out O2 and H2O values
    '''
    a = tfile.split(fname)
    b = a[1].split('_')
    
    c   =b[1].split('h')
    h2o = c[1]

    c  = b[2].split('.fits') 
    d  = c[0].split('o')
    o2 = d[1]    
    return h2o,o2


def parse_year(mjd):
    '''
    Get year from mjd, needed to find rawdata directories
    '''
    t = Time(mjd,format='mjd')
    a = t.to_value('jyear', subfmt='str')
    b=a.split('.')
    return b[0]


def parse_spat(pname):
    tmp = pname.split('-')
    tmp1 = tmp[1]
    tmp2 = tmp1.split('SLIT')
    spat = float(tmp2[1])

    return spat

#############################################################
def read_marz_output(marz_file):
    
    mz = ascii.read(marz_file)
    usecols = {3: "RA", 4: "DEC", 13: "SPEC_Z", 14: "ZQUALITY", 1: "SPECOBJID"}
    mz['col3'].name  = 'RA'
    mz['col4'].name  = 'DEC'
    mz['col13'].name = 'SPEC_Z'
    mz['col14'].name = 'ZQUALITY'
    mz['col11'].name = 'TYPE'
    mz["RA"]  *= 180.0 / np.pi
    mz["DEC"] *= 180.0 / np.pi
    
    return mz


#############################################################
def set_lsf_correction(mask,nexp):

    seeing_over_slitwidth = mask['seeing'][nexp] / mask['slitwidth'][nexp]
            
    # FIT DETERMINED FROM FINDING BEST TELLURIC LSF
    # SEE NOTEBOOK 
    m = 0.23547431
    b = 0.68572587
    correction_factor = m * seeing_over_slitwidth + b
    if correction_factor > 1.0:
        correction_factor = 1.0

    mask['lsf_correction'][nexp] = correction_factor

    return mask

#############################################################
def add_marz(data_dir,mask,slits,log):
    
    marz_file = data_dir+'../marz_files/marz_'+mask['maskname'][0]+'_MG.mz'
    if os.path.isfile(marz_file):
        mz_gal    = read_marz_output(marz_file)

        m1,m2,dd = sm.spherematch(slits['RA'], slits['DEC'],mz_gal['RA'],mz_gal['DEC'],1./3600)
        slits['marz_flag'][m1] = mz_gal['ZQUALITY'][m2]
        slits['marz_z'][m1]    = mz_gal['SPEC_Z'][m2]  + np.mean(mask['vhelio'])/3e5
        slits['marz_tmpl'][m1] = mz_gal['TYPE'][m2]

        ngal   = np.sum(mz_gal['ZQUALITY'] > 2) # GALAXIES
        printlog(log,'{} Add marz results with {} galaxies'.format(mask['maskname'][0],ngal))


    else:
        print(marz_file)
        printlog(log,'{} No MARZ FILE!'.format(mask['maskname'][0]))
        
    return slits

#############################################################
def add_chipgap_seeing(data_dir,mask,slits,log):


    for ii,msk in enumerate(mask):

        hdu = fits.open(data_dir + '/Science/' +msk['spec1d_filename'])

        for arg,obj in enumerate(slits):
            if obj['flag_skip_exp'][ii] == 0:

                # GET CHIP GAP
                data = hdu[obj['slitname'][ii]].data
                flux = data['OPT_COUNTS']
                wave = data['OPT_WAVE']
                pmin,pmax = dmost_utils.find_chip_gap(flux)  
                slits['chip_gap_r'][arg,ii] = wave[int(pmax)]
                slits['chip_gap_b'][arg,ii] = wave[int(pmin)]

                # GET FWHM
                shdr = hdu[obj['slitname'][ii]].header
                slits['opt_fwhm'][arg,ii] = 0.1185 * shdr['FWHM']


        # ADD OVERALL SEEING VALUE
        seeing_min_SN = 10. 

        # IN CASE NO STARS WITH GOOD SN, LOWER THRESHOLD
        mstar = (slits['SN'][:,ii] > seeing_min_SN ) & (slits['marz_flag'] < 2) & (slits['flag_skip_exp'][:,ii] == 0)
        while (np.sum(mstar) < 2):
            mstar = (slits['SN'][:,ii] > seeing_min_SN ) & (slits['marz_flag'] < 2) & (slits['flag_skip_exp'][:,ii] == 0)
            seeing_min_SN = seeing_min_SN - 0.5
            print(np.sum(slits['SN'][:,ii] > seeing_min_SN ),np.sum(slits['marz_flag'] < 2),np.sum(slits['flag_skip_exp'][:,ii] == 0))
            if seeing_min_SN < 0:
                mstar = (slits['marz_flag'] < 2) 


        mask['seeing'][ii] = np.nanmedian(slits['opt_fwhm'][mstar,ii])

        # IF WE CAN"T MEASURE SEEING, ASSUME ITS LARGE
        if (mask['seeing'][ii] < 0.1):
            mask['seeing'][ii] = 2.


        # ADD SLITWIDTH TO MASKS    
        DEIMOS_RAW = os.getenv('DEIMOS_RAW')
        rhdu       = fits.open(DEIMOS_RAW + 'rawdata_'+mask['year'][ii]+'/'+mask['rawfilename'][ii])
        desislits  = rhdu['DesiSlits'].data
        median_slitwidth      = np.median(desislits['slitWid'])
        mask['slitwidth'][ii] = 0.01*(round(median_slitwidth/0.01))
        

        # SET LSF CORRECTION -- DETERMINED FROM FIT
        mask = set_lsf_correction(mask,ii)


        printlog(log,'{} {} Slitwidth is {:0.1f}, Seeing is {:0.2f} arcsec, LSF correction is {:0.2f}'.format(mask['maskname'][0], mask['fname'][ii],\
                                                                            mask['slitwidth'][ii],mask['seeing'][ii],mask['lsf_correction'][ii]))


    return mask, slits


 
#############################################################
# MAKE SOME PLOTS
def mk_histograms(data_dir,mask,slits,nexp):

    hfile  = data_dir + '/QA/histograms_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(hfile)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))
    plt.rcParams.update({'font.size': 16})

    ax1.hist(slits['rms_arc'])
    ax1.set_xlabel('RMS_ARC [pixels]')
    ax1.set_title('Arc Solutions')
    ax1.axvline(0.4,c='r',label='RMS Threshold')
    ax1.legend(loc=1)

    for n in np.arange(0,nexp,1):
        
        m = (slits['marz_flag'] < 2) & (slits['rms_arc'] < 0.4) & (slits['flag_skip_exp'][:,n] == 0)
        ax2.hist(slits['opt_fwhm'][m,n],bins=20,alpha=0.5,\
                 label = 'seeing = {:0.2f}'.format(mask['seeing'][n]))

#    ax2.axvline(8,c='r',label='FWHM Threshold')
    ax2.legend()
        
    ax2.set_xlabel('Seeing:  OPT_FWHM [Arcsec]')
    ax2.set_title('Seeing')

    pdf.savefig()
    pdf.close()
    plt.close(fig)


    return


#############################################################
# POPULATE MASK VALUES
def populate_mask_info(data_dir,nexp,maskname,spec1d_files,log):

    mask = create_mask(nexp)

    log.write('Populating mask values for exposures')
    
    printlog(log,'{} Populating mask values for {} exposures'.format(maskname,nexp))


    for i,spec1d in enumerate(spec1d_files):

        # HEADER VALUES
        try:
            hdu      = fits.open(spec1d)
            hdr      = hdu[0].header
        except:
            print('Cannot find file {}'.format(spec1d))
            print('You probably need to run pypeit')
            return
            

        mask['maskname'][i]       = maskname
        mask['mjd'][i]            = hdr['MJD']
        mask['year'][i]           = parse_year(hdr['mjd'])

        mask['deimos_maskname'][i]= hdr['TARGET'].strip()
        mask['nexp'][i]           = nexp

        mask['spec1d_filename'][i]= spec1d.split('Science/')[1]
        mask['rawfilename'][i]    = hdr['FILENAME']
        mask['fname'][i]          = hdr['FILENAME'].split('.')[2]

        # AIRMASS, EXP FOR EACH EXPOSURE
        mask['airmass'][i] = hdr['AIRMASS']
        mask['exptime'][i] = hdr['EXPTIME']
        mask['mask_ra'][i] = hdr['RA']
        mask['mask_dec'][i]= hdr['DEC']


        # HELIOCENTRIC VELOCITY, ADD TO MEASURED VALUES
        mask['vhelio'][i]  = dmost_utils.deimos_helio(hdr['MJD'],hdr['RA'],hdr['DEC'])
        printlog(log,'{} {} Heliocentric velocity {:0.2f} kms'.format(maskname,mask['fname'][i],mask['vhelio'][i]))



    return mask


#############################################################
# POPULATE SLIT VALUES
def create_slits_from_collate1d(data_dir,mask,nexp,log):

    
    # READ COLLATE_REPORT
    cfile     = data_dir+'collate_report.dat'
    collate1d = ascii.read(cfile) 


    collate1d.sort('spec1d_filename')

    # FIND ALL UNIQUE OBJECTS BASED ON FILENAME
    obj_filenames   = np.unique(collate1d['filename'])
    nslits          = np.size(obj_filenames)
    
    
    # CREATE SLITS TABLE USING COLLATE FILE 
    slits = create_slits(nslits,nexp)

    # HACK FOR SINGLE EXPOSURES
    if (nexp == 1):
        slits = create_slits(nslits,2)

    

    # FOR EACH UNIQUE COLLATE1D FILE
    nserendip = 1
    ntolerance = 0
    for i,objname in enumerate(obj_filenames):

        m        = np.in1d(collate1d['filename'],objname)
        this_obj = collate1d[m]


        slits['collate1d_filename'][i] = this_obj['filename'][0]
        slits['objname'][i]    = this_obj['maskdef_objname'][0]
        slits['objid'][i]      = this_obj['maskdef_id'][0]

        slits['collate1d_SN'][i]       = np.sqrt(np.sum(this_obj['s2n']**2))

        slits['RA'][i]      = this_obj['objra'][0]
        slits['DEC'][i]     = this_obj['objdec'][0]
        slits['rms_arc'][i] = this_obj['wave_rms'][0]


        # SET IF SEREDIP SLIT
        if (this_obj['maskdef_objname'][0] == 'SERENDIP'):
            slits['serendip'][i]   = nserendip
            nserendip += 1

        for ii,this_exp in enumerate(mask['spec1d_filename']):

            # ENSURE EXPOSURES MATCH
            m = this_obj['spec1d_filename'] == mask['spec1d_filename'][ii] 
            slits['flag_skip_exp'][i,ii] = 1
            # FOR WIDELY SPACED EXPOSURES, NEED TO INCREASE TOLERANCE
            if (np.sum(m) == 2):
                printlog(log,'{} {} Consider re-running collate1d with larger tolerance'.format(i,ii))
                ntolerance = ntolerance+1
                if ntolerance > 80:
                    printlog(log,'Too many repeats.   Re-run collate1d, exiting dmost')
                    sys.exit()
                    
            if (np.sum(m) == 1):
                this_exp = this_obj[m][0]
                slits['slitname'][i,ii]      = this_exp['pypeit_name']
                slits['SN'][i,ii]            = this_exp['s2n']
                slits['det'][i,ii]           = this_exp['det'].strip()
                slits['spat_pixpos'][i,ii]   = parse_spat(this_exp['pypeit_name']) 
                slits['flag_skip_exp'][i,ii] = 0


    # SORT BY SN
    slits.sort('collate1d_SN')
    slits.reverse()
    

    printlog(log,'{} Created slit table with {} slits ({} serendips)'.format(mask['maskname'][0],nslits,nserendip))


    return slits
    



#############################################################
def create_single_mask(data_dir, maskname):
    '''
    Construct dmost table based on size of PyPeIT spec1d files

    '''

    # DEFINE DIRECTORIES, GRAB SPEC1D FILES
    spec1d_files = glob.glob(data_dir+'/Science/spec1d*fits')
    spec1d_files = np.sort(spec1d_files)
    nexp         = np.size(spec1d_files)


    # DMOST AND LOG FILE NAMES
    outfile      = data_dir+'/dmost/dmost_'+maskname+'.fits'
    logfile      = data_dir + maskname+'_dmost.log'
    if os.path.isfile(logfile):
        log          = open(logfile,'a')
        log.write('-----------------\n')

    if ~os.path.isfile(logfile):
        log          = open(logfile,'w')
    

    if (nexp == 0):
        printlog(log,'No spec1d files found!')
        print(data_dir)
        return [],[],[],[]



    # IF DMOST FILE EXISTS, READ DMOST IN
    if os.path.isfile(outfile):
        print('Reading existing file: ',outfile)
        slits,mask = dmost_utils.read_dmost(outfile)

        printlog
        printlog(log,'{} Slit table with {} slits'.format(mask['maskname'][0],np.size(slits)))


    else:

        # CREATE MASK
        mask = populate_mask_info(data_dir,nexp,maskname,spec1d_files,log)

        # CREATE SLITS FROM COLLATE1D
        slits = create_slits_from_collate1d(data_dir,mask,nexp,log)

        # ADD EXPOSURE LEVEL DATA
        mask,slits  = add_chipgap_seeing(data_dir,mask,slits,log)
        mk_histograms(data_dir,mask,slits,nexp)


    # ADD OR UPDATE MARZ
    slits = add_marz(data_dir,mask,slits,log)
    log.close()

    return slits, mask, nexp, outfile

#####################################################    
def main(*args):


    mask = sys.argv[1]
    
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')
    
    s,m = create_single_mask(mask)
    
if __name__ == "__main__":
    main()
    
    

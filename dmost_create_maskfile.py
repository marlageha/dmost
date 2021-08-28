#!/usr/bin/env python

import numpy as np
import os,sys

import matplotlib.pyplot as plt


from astropy.table import Table,Column,vstack
from astropy.io import ascii,fits
import glob
import time

import matplotlib.backends.backend_pdf

import astropy.units as u
from astropy.time import Time

import pyspherematch as sm

import  dmost_utils,dmost_flexure
import dmost_telluric, dmost_chi2_template, dmost_emcee, dmost_coadd_emcee



DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')

c = 299792.


# two layers of tables
# 1.  MASK LEVEL
#         nexp
#         [airmass]
#         [mjd]
#         [rawfile]
#         [spec1dfiles]
#         [mask telluric values]
# 2.  SLIT LEVEL
#     -- constructed from raw bintabs table
#     -- add in coadd1d file
#     -- add in data from dmost, 
# 
#     [slit identifiers]
#     [flexure parameters]
#     [telluric LSF, wguess]
#     vignetting limits
#     collate1d filename
#     collate1d SN
#     combined synthetic template
#     marz flags
#     [emcee velocities]
#     combined velocity
#     photometry
#     membership


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

    cols = [filled_column('spec1d_filename','                                                                        ',nexp),
            filled_column('rawfilename','                          ',nexp), 
            filled_column('deimos_maskname','              ',nexp), 
            filled_column('maskname','        ',nexp), 
            filled_column('fname','        ',nexp),
            filled_column('mjd',-1.,nexp),
            filled_column('year','    ',nexp),
            filled_column('airmass',-1.,nexp),
            filled_column('exptime',-1.,nexp),
            filled_column('vhelio',-1.,nexp),
            filled_column('telluric_h2o',-1.,nexp),
            filled_column('telluric_o2',-1.,nexp),
            filled_column('flag_flexure',0,nexp),
            filled_column('flag_telluric',0,nexp),
            filled_column('flag_template',0,nexp),
            filled_column('flag_emcee',0,nexp),
            filled_column('flag_coadd',0,nexp),
            filled_column('nexp',-1,nexp),
            
            # TELESCOPE DATA
            filled_column('mask_ra',-1.,nexp),
            filled_column('mask_dec',-1.,nexp),

           ]

 
    mask = Table(cols)

    return mask


###################################
# CREATE ALLSPEC DATA STRUCTURE
def create_slits(nslits,nexp):

    cols = [filled_column('slitname','                  ',nslits),
            filled_column('maskdef_id','                ',nslits),
            filled_column('RA',-1.,nslits),
            filled_column('DEC',-1.,nslits),
          
            filled_column('reduce_flag',-1*np.ones(nexp,dtype='int'),nslits),
            
            
            # DETECTOR POSITION, USE TO READ SPEC1D FILES
            # bname = 'SPAT{:04d}-SLIT{:04d}-DET{:02d}'.format(bspat,bslit,bdet)
            filled_column('rdet',np.zeros(nexp,dtype='int'),nslits),
            filled_column('bdet',np.zeros(nexp,dtype='int'),nslits),
            filled_column('rspat',np.zeros(nexp,dtype='int'),nslits),
            filled_column('bspat',np.zeros(nexp,dtype='int'),nslits),
            filled_column('rslit',np.zeros(nexp,dtype='int'),nslits),
            filled_column('bslit',np.zeros(nexp,dtype='int'),nslits),
            filled_column('rms_arc_r',np.zeros(nexp),nslits),
            filled_column('rms_arc_b',np.zeros(nexp),nslits),
            filled_column('opt_fwhm',np.zeros(nexp),nslits),
            filled_column('ccd_gap_b',np.zeros(nexp),nslits),
            filled_column('ccd_gap_r',np.zeros(nexp),nslits),
            filled_column('xpos',-1.,nslits),
            filled_column('ypos',-1.,nslits),
            filled_column('slitwidth',-1.,nslits),
            
            # COLLATE1D
            filled_column('collate1d_filename','                                         ',nslits),
            filled_column('collate1d_SN',-1.,nslits),
         
            # MARZ
            filled_column('marz_flag',-1,nslits),
            filled_column('marz_z',-1.,nslits),
            
            # FLEXURE
            filled_column('rSN',np.zeros(nexp),nslits),
            filled_column('bSN',np.zeros(nexp),nslits),
            filled_column('fit_slope',np.zeros(nexp),nslits),
            filled_column('fit_b',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p0',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p1',np.zeros(nexp),nslits),
            filled_column('fit_lsf_p2',np.zeros(nexp),nslits),
            filled_column('fit_lsf',np.zeros(nexp),nslits),
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
            filled_column('emcee_v',np.zeros(nexp),nslits),
            filled_column('emcee_w',np.zeros(nexp),nslits),
            filled_column('emcee_v_err',np.zeros(nexp),nslits),
            filled_column('emcee_w_err',np.zeros(nexp),nslits),
            filled_column('emcee_v_err16',np.zeros(nexp),nslits),
            filled_column('emcee_v_err84',np.zeros(nexp),nslits),
            filled_column('emcee_w_err16',np.zeros(nexp),nslits),
            filled_column('emcee_w_err84',np.zeros(nexp),nslits),
            filled_column('emcee_f_acc',np.zeros(nexp),nslits),
            filled_column('emcee_nsamp',np.zeros(nexp),nslits),
            filled_column('emcee_burnin',np.zeros(nexp,dtype=int),nslits),
            filled_column('emcee_converge',np.zeros(nexp),nslits),
            filled_column('emcee_lnprob',np.zeros(nexp),nslits),
   
            # LOW SN V
            filled_column('coadd_v',-1.,nslits),
            filled_column('coadd_w',-1.,nslits),
            filled_column('coadd_v_err',-1.,nslits),
            filled_column('coadd_w_err',-1.,nslits),
            filled_column('coadd_v_err16',-1.,nslits),
            filled_column('coadd_v_err84',-1.,nslits),
            filled_column('coadd_f_acc',-1.,nslits),
            filled_column('coadd_converge',-1.,nslits),
            filled_column('coadd_burnin',-1,nslits),
            filled_column('coadd_nsamp',-1.,nslits),
            filled_column('coadd_lnprob',-1.,nslits),

            # COMBINED VELOCITIES
            filled_column('dmost_v',-1.,nslits),
            filled_column('dmost_v_err',-1.,nslits),
            filled_column('v_nexp',1,nslits),
           
            
            filled_column('prob_member',-1,nslits),
            filled_column('gmag',-1,nslits),
            filled_column('rmag',1,nslits)
           ]
            
    slits = Table(cols)
    return slits

#############################################################
def dmost_parse_telluric(tfile,fname):
    a = tfile.split(fname)
    b = a[1].split('_')
    
    c   =b[1].split('h')
    h2o = c[1]

    c  = b[2].split('.fits') 
    d  = c[0].split('o')
    o2 = d[1]    
    return h2o,o2

def parse_year(mjd):
    t = Time(mjd,format='mjd')
    a = t.to_value('jyear', subfmt='str')
    b=a.split('.')
    return b[0]



#############################################################
# POPULATE MASK VALUES
def populate_mask_info(data_dir,nexp,maskname,spec1d_files):

    mask = create_mask(nexp)
    
    print('{} Populating mask values for {} exposures'.format(maskname,nexp))
    for i,spec1d in enumerate(spec1d_files):

        # HEADER VALUES
        hdu      = fits.open(spec1d)
        hdr      = hdu[0].header
        fnames   = hdr['FILENAME'].split('.')

        mask['maskname'][i]       = maskname
        mask['year'][i]           = parse_year(hdr['mjd'])
        mask['deimos_maskname'][i]= hdr['TARGET'].strip()
        mask['nexp'][i]           = nexp


        mask['spec1d_filename'][i]= spec1d.split('Science')[1]
        mask['rawfilename'][i]    = hdr['FILENAME']
        mask['fname'][i]          = fnames[2]

        # AIRMASS, EXP FOR EACH EXPOSURE
        mask['airmass'][i] = hdr['AIRMASS']
        mask['exptime'][i] = hdr['EXPTIME']
        mask['mask_ra'][i] = hdr['RA']
        mask['mask_dec'][i]= hdr['DEC']

        mask['mjd'][i]     = hdr['MJD']

        # HELIOCENTRIC VELOCITY, ADD TO MEASURED VALUES
        mask['vhelio'][i]  = dmost_utils.deimos_helio(hdr['MJD'],hdr['RA'],hdr['DEC'])
        
        
        # GET TELLURIC VALUES FROM DIRECTORIES
        tfile = glob.glob(data_dir+'/dmost/telluric_*'+mask['fname'][i]+'*.fits')
        if np.size(tfile) == 1:
            h2o,o2 = dmost_parse_telluric(tfile[0],mask['fname'][i])
            mask['telluric_h2o'][i] = h2o
            mask['telluric_o2'][i]  = o2


    return mask


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
def add_marz(data_dir,mask,slits):
    
    marz_file = DEIMOS_REDUX+'/marz_files/marz_'+mask['maskname'][0]+'_MG.mz'
    if os.path.isfile(marz_file):
        mz_gal    = read_marz_output(marz_file)
        m         = mz_gal['ZQUALITY'] > 1 # ZQ = 2 is ambigous, but run as star

        mz_gal = mz_gal[m]
        if np.sum(m) > 0:

            m1,m2,dd = sm.spherematch(slits['RA'], slits['DEC'],mz_gal['RA'],mz_gal['DEC'],1./3600)

            slits['marz_flag'][m1]   = mz_gal['ZQUALITY'][m2]
            slits['marz_z'][m1]      = mz_gal['SPEC_Z'][m2]

        ngal   = np.sum(mz_gal['ZQUALITY'] > 2) # GALAXIES

        print('{} Add marz results with {} galaxies'.format(mask['maskname'][0],ngal))
    else:
        print(marz_file)
        print('{} No MARZ FILE!'.format(mask['maskname'][0]))
        
    return slits



#############################################################
# POPULATE SLIT VALUES
def create_slits_from_bintab(data_dir,mask,nexp):

    
    # CREATE TABLE USING BINTABS
    # BluSlits contains mask design x/y positions
    rhdu      = fits.open(DEIMOS_RAW + 'rawdata_'+mask['year'][0]+'/'+mask['rawfilename'][0])
    bintab    = rhdu['ObjectCat'].data
    bluslits  = rhdu['BluSlits'].data  
    desislits = rhdu['DesiSlits'].data
    m         = bintab['ObjClass'] == 'Program_Target'

    bintab    = bintab[m]
    bluslits  = bluslits[m]
    desislits = desislits[m]
    nslits    = np.sum(m)
    
    

    # CREATE SLITS TABLE USING BINTABS
    slits = create_slits(nslits,nexp)

    # MATCH AGAINST COLLATE1D FILES
    collate1d = ascii.read(data_dir+'collate_report.dat')
    collate1d_files = np.unique(collate1d['filename'])


    # POPULATE USING BINTABS AND COLLATE1D DATA
    ncol1d = 0
    for i,(obj,bsl,dsl) in enumerate(zip(bintab,bluslits,desislits)):
        slits['RA'][i]         = obj['RA_OBJ']
        slits['DEC'][i]        = obj['DEC_OBJ']
        slits['slitname'][i]   = obj['OBJECT']
        slits['maskdef_id'][i] = obj['OBJECTID']
        slits['xpos'][i]       = (bsl['slitX1']+bsl['slitX2'])/2.
        slits['ypos'][i]       = bsl['slitY1']
        slits['slitwidth'][i]  = 0.01*(round(dsl['slitWid']/0.01))

        
        # GENERAL REDUCE FLAG
        slits['reduce_flag'][i,:] = 1

        
        # MATCHING COLLATE1D ON RA/DEC
        m = np.in1d(collate1d['maskdef_objname'],obj['OBJECT'])
        m1,m,dd = sm.spherematch(obj['RA_OBJ'], obj['DEC_OBJ'],collate1d['objra'],collate1d['objdec'],1./3600)

        # READ COLLATE1D AND ADD SN
        if (np.size(m) > 0):
            names = collate1d['filename'][m]
            slits['collate1d_filename'][i] = names[0]

            chdu = fits.open(data_dir+'collate1d/'+names[0])
            all_wave,all_flux,all_ivar, SN = dmost_utils.load_coadd_collate1d(chdu) 
            slits['collate1d_SN'][i]       = SN
            ncol1d+=1
            
        else:
            slits['reduce_flag'][i,:] = 0
        
    # SORT BY SN
    slits.sort('collate1d_SN')
    slits.reverse()
    
    print('{} Created slit table with {} slits'.format(mask['maskname'][0],nslits))
    print('{} Collate1d slit matches  {} slits'.format(mask['maskname'][0],ncol1d))
    

    return slits
    


#############################################################
def add_spec1d_fileinfo(data_dir,slits,mask,nexp):

    # FOR EACH EXPOSURE
    for ii,spec1d_file in enumerate(mask['spec1d_filename']): 

        hdu         = fits.open(data_dir+'Science/'+spec1d_file)
        header      = hdu[0].header
        nspec       = header['NSPEC']
        
        # SLIT HEADERS START AT ONE (NOT ZERO!)
        for i in np.arange(1,nspec+1,1,dtype='int'):

            slit_header = hdu[i].header
    
            m1,m2,dd = sm.spherematch(slits['RA'], slits['DEC'],[slit_header['RA']],[slit_header['DEC']],2./3600)
            
            
            if (np.size(m1) > 0) & (slit_header['DET'] < 5):
                arg=m1[0]
                slits['bdet'][arg,ii]       = slit_header['DET']
                slits['bslit'][arg,ii]      = slit_header['SLITID']
                slits['bspat'][arg,ii]      = round(slit_header['HIERARCH SPAT_PIXPOS'])
                slits['rms_arc_b'][arg,ii]  = slit_header['WAVE_RMS']

            if (np.size(m1) > 0) & (slit_header['DET'] >= 5):
                arg=m1[0]
                slits['rdet'][arg,ii]       = slit_header['DET']
                slits['rslit'][arg,ii]      = slit_header['SLITID']
                slits['rspat'][arg,ii]      = round(slit_header['HIERARCH SPAT_PIXPOS'])
                slits['rms_arc_r'][arg,ii]  = slit_header['WAVE_RMS']
                slits['opt_fwhm'][arg,ii]   = slit_header['FWHM']*0.12  # DEIMOS SPATIAL PIXEL SCALE

            if (np.size(m1) == 0) & (slit_header['HIERARCH MASKDEF_OBJNAME'] != 'SERENDIP'):
                print('MISSING SLITS',slit_header['NAME'])
                print(slit_header['HIERARCH MASKDEF_OBJNAME'])
                
                
                
        nbmiss = np.sum((slits['bdet'][:,ii] == 0)) 
        nrmiss = np.sum((slits['rdet'][:,ii] == 0)) 

        # SKIP SLITS WITH ONLY ONE DETECTOR, BUT COULD RECOVER THESE.
        slits['reduce_flag'][slits['bdet'][:,ii] == 0] = 0 
        slits['reduce_flag'][slits['rdet'][:,ii] == 0] = 0 

        print('{} {} There are {} blue and {} red slits missing data'.format(mask['maskname'][0],\
                                                                             mask['fname'][ii],nbmiss, nrmiss))
        
    return slits



#############################################################
def write_dmost(slits,mask,outfile):

    hdup = fits.PrimaryHDU(np.float32([1,2]))
    hdu1 = fits.BinTableHDU(mask,name = 'mask')
    hdu2 = fits.BinTableHDU(slits,name = 'slits')
    fhdu = fits.HDUList([hdup, hdu1,hdu2])
    fhdu.writeto(outfile,overwrite=True)
    
    
def read_dmost(outfile):
    
    hdu   = fits.open(outfile)
    mask  = hdu[1].data
    slits = hdu[2].data

    return slits, mask



#############################################################
def run_single_mask(maskname,flag_telluric=0,flag_template=0,flag_emcee=0,flag_flexure=0):


    # DEFINE DIRECTORIES, GRAB SPEC1D FILES
    DEIMOS_REDUX  = os.getenv('DEIMOS_REDUX')

    data_dir     = DEIMOS_REDUX+maskname+'/'
    spec1d_files = glob.glob(data_dir+'Science/spec1d*fits')
    nexp         = np.size(spec1d_files)
    if (nexp == 0):
        print('No spec1d files found!')
        print(data_dir)
        return

    # IF FILE EXISTS, READ DMOST IN
    outfile      = data_dir+'/dmost/dmost_mask_'+maskname+'.fits'
    if os.path.isfile(outfile):
        print('Reading existing file',outfile)
        slits,mask = read_dmost(outfile)

    else:

        # CREATE MASK
        mask = populate_mask_info(data_dir,nexp,maskname,spec1d_files)

        # CREATE SLITS, POPULATE WITH BINTAB AND COLLATE1D
        slits = create_slits_from_bintab(data_dir,mask,nexp)

        # ADD SPEC1d_FILE INFO
        slits = add_spec1d_fileinfo(data_dir,slits,mask,nexp)



    # ADD MARZ
    slits = add_marz(data_dir,mask,slits)


    # RUN FLEXURE
    if ~(np.sum(mask['flag_flexure']) == nexp):
        slits,mask = dmost_flexure.run_flexure(data_dir,slits,mask)
        write_dmost(slits,mask,outfile)


    # RUN TELLURIC
    tfile = glob.glob(data_dir+'/dmost/telluric_'+maskname+'*.fits')
    if ~(np.sum(mask['flag_telluric']) == nexp) | (flag_telluric == 1) | ~(np.size(tfile) == nexp):
        slits,mask  = dmost_telluric.run_telluric_mask(data_dir, slits, mask)
        write_dmost(slits,mask,outfile)


    # RUN CHI2 TEMPLATE FINDER ON COMBINED DATA
    if ~(np.sum(mask['flag_template']) == nexp) | (flag_template == 1):
        slits,mask  = dmost_chi2_template.run_chi2_templates(data_dir, slits, mask)
        write_dmost(slits,mask,outfile)


    # RUN EMCEE
    slits, mask  = dmost_emcee.run_emcee(data_dir, slits, mask,outfile)
    write_dmost(slits,mask,outfile)

    # RUN LOW SN EMCEE
    #if ~(np.sum(mask['flag_emcee']) == nexp) | (flag_emcee == 1):
    slits, mask  = dmost_coadd_emcee.run_coadd_emcee(data_dir, slits, mask,outfile)
    write_dmost(slits,mask,outfile)


    print()
    return slits,mask

#####################################################    
def run_many_masks(masknames,flag_telluric=0,flag_template=0,flag_emcee=0,flag_flexure=0):

    for masks in masknames:
        slits,mask = run_single_mask(masks,flag_telluric=0,flag_template=0,flag_emcee=0,flag_flexure=0)
    
    return


#####################################################    
def main(*args):


    mask = sys.argv[1]
    
    DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')
    
    s,m = run_single_mask(mask)
    
if __name__ == "__main__":
    main()
    
    

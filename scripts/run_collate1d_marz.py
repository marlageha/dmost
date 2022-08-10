#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import glob
import argparse


from astropy.table import Table,Column
from astropy.io import ascii,fits

from pypeit import pypeitsetup



##########################################################
# LOAD OUTPUT SPECTRUM FROM PYPEIT_COLLATE1D FOR MARZ
def load_coadd1d_for_marz(data_dir,file,vacuum=0,vignetted = 0):

    SN = 0
    jhdu = fits.open(file)
    hdr  = jhdu[0].header
    data = jhdu[1].data
    
    jhdu.close()

    npix = np.size(data['wave'])
    if (npix > 8000):

        vwave    = data['wave']
        all_flux = data['flux']
        all_ivar = data['ivar']


        # CONVERT PYPEIT OUTPUT WAVELENGTHS FROM VACUUM TO AIR
        all_wave = vwave
        if (vacuum == 0):
            all_wave = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)


        # TRIM ENDS
        all_wave=all_wave[0:8000]
        all_flux=all_flux[0:8000]
        all_ivar=all_ivar[0:8000]

        # REMOVE CRAZY 500-SIGMA VALUES
        cmask = (all_flux > np.percentile(all_flux,0.1)) & (all_flux < np.percentile(all_flux,99.9))

        m=np.median(all_flux[cmask])
        s=np.std(all_flux[cmask])
        mm = (all_flux > 500.*s + m) | (all_flux < m-50.*s)
        all_flux[mm] = m
        all_ivar[mm] = 1e6
        if (np.sum(mm) > 10):
            print('Removing more than 10 pixels of data')
        

    else:
        all_flux = np.zeros(8000)
        all_ivar = np.zeros(8000)
        all_wave = np.zeros(8000)
        
        all_flux[0:npix] = data['flux']
        all_ivar[0:npix] = data['ivar']
  
        vwave = data['wave']            
        all_wave[0:npix] = vwave / (1.0 + 2.735182E-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)
                   
        
    return all_wave,all_flux,all_ivar, hdr



########################################
def filled_column(name, fill_value, size):
    """
    Tool to allow for large strings
    """
    return Column([fill_value]*int(size), name)

########################################
# CREATE MARZ INPUT
def create_marz_input(mask,working_dir):

    # LOAD COLLATE1D FILES
    os.chdir(working_dir)
    Jfiles = glob.glob(working_dir + '/collate1d/J*')
    nslits = np.size(Jfiles)

    # LOAD REPORT FILE FOR METADATA
    collate_data = ascii.read(working_dir+'/collate_report.dat')

    # OUTFILE
    marz_file = 'marz_'+mask+'.fits'

    cols = [filled_column('RA',-1.,nslits),
            filled_column('DEC',-1.,nslits),
            filled_column('NAME','                           ',nslits),
            filled_column('MAG',-1.,nslits),
            filled_column('TYPE',' ',nslits)]
    slits = Table(cols)

    for i,file in enumerate(Jfiles):
            vwave,data_flux,data_ivar,hdr = load_coadd1d_for_marz(working_dir,file)

 
            # MATCH TO FILE FOR OBJECT INFO
            tmp_jfile = file.split('/collate1d/')
            mser = tmp_jfile[1] == collate_data['filename']

            # IDENTIFY SERENDIPS, ADD TO NAME
            serendip = ''
            if np.sum(mser) > 0:
                maskdef_name = collate_data['maskdef_objname'][mser][0]
                if (maskdef_name == 'SERENDIP'):
                    serendip = 'SDP_'

            t    = hdr[np.size(hdr)-1]
            tt   = t.split('SPAT')
            name = ('SPAT'+tt[1]).split(' ')

            slits['RA'][i]   = hdr['RA_OBJ ']  * (np.pi/180)
            slits['DEC'][i]  = hdr['DEC_OBJ']* (np.pi/180)
            slits['NAME'][i] = serendip+name[0]
            slits['TYPE'][i] = 'P'

            print(i,name)

            if (np.sum(data_flux) == 0):
                print(name)
                data_flux = data_flux+1
                data_ivar = data_ivar+1

            # MARZ REQUIRES AIR WAVELENTHS CONVERT FROM PYPEIT VACUUM
            awave = vwave / (1.0 + 2.735182e-4 + 131.4182 / vwave**2 + 2.76249E8 / vwave**4)

            # REMOVE WAVELENGTH ZERO VALUES FOR DISPLAY
            if (np.sum(awave) > 0):
                mzero = awave == 0
                awave[mzero] = 6000.
                data_flux[mzero] =0.


            if i==0:
                specs_flux = data_flux
                specs_var  = 1./data_ivar
                specs_wave = awave
            if (i> 0):
                specs_flux = np.vstack([specs_flux,data_flux])
                specs_var  = np.vstack([specs_var,1./data_ivar])
                specs_wave = np.vstack([specs_wave,awave])
 
                
    # WRITE TO FILE            
    fits.HDUList([
        fits.PrimaryHDU(specs_flux, do_not_scale_image_data=True),\
        fits.ImageHDU(specs_var, name="variance", do_not_scale_image_data=True),\
        fits.ImageHDU(specs_wave, name="wavelength", do_not_scale_image_data=True),\
        fits.BinTableHDU(slits, name="fibres"),\
    ]).writeto(marz_file, overwrite=True)

########################################
# RUN COLLATE1D
def run_collate1d(mask,tolerance):
 
    # RUN PYPEIT COLLATE1D
    collate1d = 'pypeit_collate_1d --spec1d_files Science/spec1d_*fits --toler '+tolerance+' --refframe heliocentric --spec1d_outdir ../junk_collate1d/'
    os.system(collate1d)
    
    # MOVE INTO DIRECTORY
    os.system('mv J* collate1d')
    
    
    Jfiles = glob.glob('/collate1d/J*')

   
    return Jfiles
  
#####################################################    
#####################################################    
def main(*args):

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", dest = "mask", default = " ",required = True)
    parser.add_argument("--tolerance", dest = "tolerance", default = "1.0",required = False)
    args = parser.parse_args()
    print(args.tolerance)

    print('Running Collate 1d and creating Marz file for ',args.mask)

    # DETERMINE IF COLLATE1D ALREADY RUN 
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')
    working_dir  = DEIMOS_REDUX+args.mask
    os.chdir(working_dir)
    Jfiles = glob.glob(working_dir + '/collate1d/J*')

    if np.size(Jfiles) < 2:
        
        # COLLATE1D
        print('Running collate1d')
        Jfiles = run_collate1d(args.mask,args.tolerance)

    # MARZ
    print('Creating Marz input file')
    create_marz_input(args.mask,working_dir)
    os.system('mv marz*fits ../marz_files')
    
if __name__ == "__main__":
    main()

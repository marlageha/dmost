#!/usr/bin/env python
import numpy as np
import os

from astropy import table
from astropy.table import Table
from astropy.io import ascii

from astropy.table import Table,vstack
from astropy.io import ascii,fits
import glob



###########################################
# CALCULATE GRATING PARAMETERS FROM HEADER
#   g_rule  - rule spacing [lines/mm]
#   grangle - rule spacing [degrees]
#   lambda_c - central wavelength [angstroms]
#
def deimos_grating(header):
    grt_number = header['GRATEPOS']
    grt_name   = header['GRATENAM']
    
    if ('MIRROR' in grt_name):
        g_rule = 0
    else:
        g_rule = grt_name
    
    # CHECK THIS IS CONSTANT WITH KECK?
    if (grt_number == 3):
        rawpos  = header['G3TLTRAW']
        grangle = (rawpos+29094)/2500.
        lambda_c = header['G3TLTWAV']
    else:
        rawpos = header['G4TLTRAW']
        grangle = (rawpos+40934)/2500. 
        lambda_c = header['G4TLTWAV']

    return grt_name,grangle, lambda_c


######################################
def parse_header(header,file):
    
    
    mask     = header['SLMSKNAM'] 
    exptime  = header['EXPTIME'] 
    obstype  = header['OBSTYPE'] 
    lamps    = header['LAMPS']
    grating  = header['GRATENAM']
    obj      = header['OBJECT']
    
    grt_name,grangle,lambda_c = deimos_grating(header)
   
    tmp   = file.split('/')
    fname = tmp[np.size(tmp)-1]

    exp_info = Table([[fname],[mask],[exptime],[obstype],[lamps],[grt_name],[obj],[grangle],[lambda_c]],\
                         names=('FNAME','MASK', 'EXPTIME','OBSTYPE','LAMPS','GRATENAME','OBJECT','GRANGLE','LAMBDA_C'))
    
    # ADD PLACEHOLDER
    exp_info['FLAT'] = 0
    exp_info['ARC']  = 0
    exp_info['SCI']  = 0

    # AND EVALAUTE
    exp_info = parse_file_type(exp_info)
    return exp_info




#################################
def parse_file_type(exp_info):
    tmp=0
    
    # IS IT A FLAT?
    if exp_info['LAMPS'][0] == 'Qz':
        exp_info['FLAT'][0] = 1
        tmp=tmp+1

    # IS IT AN ARC?
    if any(x in exp_info['LAMPS'][0] for x in ['Kr', 'Xe', 'Ar', 'Ne']):   
        exp_info['ARC'][0] = 1
        tmp=tmp+1

    # IS IT A SCIENCE (NOTE EXPOSURE TIME REQUIREMENT)
    mlamp = ('Off' in exp_info['LAMPS'][0])
    mobj  = ('Object' in exp_info['OBSTYPE'][0])
    mexp  = exp_info['EXPTIME'][0]  > 60.

    if (mlamp) & (mobj) & (mexp):
        exp_info['SCI'][0] = 1
        tmp=tmp+1

    # IF NONE OF THESE, THEN FREAK OUT    
    if (tmp==0) & (exp_info['EXPTIME'][0]  > 60.):
        print(exp_info)

        
    return exp_info



########################################3
def create_planfile(mask_table,rawdir):
    
    nflt = np.sum(mask_table['FLAT'] == 1)
    narc = np.sum(mask_table['ARC'] == 1)
    nsci = np.sum(mask_table['SCI'] == 1)

    if (nflt > 1) & (narc >= 1) & (nsci >= 1):
        planname = mask_table['MASK'][0] + '.plan'
        outfile = open(planname,'w')

        
        outfile.write('#Grating: {}   Grangle: {}\n'.format(mask_table['GRATENAME'][0],mask_table['GRANGLE'][0]))
        outfile.write('MASK: {}\n'.format(mask_table['MASK'][0]))
        outfile.write('RAWDATADIR: {}\n'.format(rawdir))
        for f in mask_table['FNAME'][mask_table['FLAT'] == 1]:
            outfile.write('FLATNAME: {}\n'.format(f))
        for f in mask_table['FNAME'][mask_table['ARC'] == 1]:
            outfile.write('ARCNAME: {}\n'.format(f))
        for f in mask_table['FNAME'][mask_table['SCI'] == 1]:
            outfile.write('SCIENCENAME: {}\n'.format(f))
    
        outfile.close()
        
    else:
        planname = mask_table['MASK'][0] + '.tmp'
        outfile = open(planname,'w')

        
        outfile.write('#Grating: {}   Grangle: {}\n'.format(mask_table['GRATENAME'][0],mask_table['GRANGLE'][0]))
        outfile.write('MASK: {}\n'.format(mask_table['MASK'][0]))
        outfile.write('RAWDATADIR: {}\n'.format(rawdir))
        for f in mask_table['FNAME'][mask_table['FLAT'] == 1]:
            outfile.write('FLATNAME: {}\n'.format(f))
        for f in mask_table['FNAME'][mask_table['ARC'] == 1]:
            outfile.write('ARCNAME: {}\n'.format(f))
        for f in mask_table['FNAME'][mask_table['SCI'] == 1]:
            outfile.write('SCIENCENAME: {}\n'.format(f))
    
        outfile.close()


#####################################################    
#####################################################    
#####################################################    
#####################################################    
def main():


    # RUN IN PRESENT DIRECTORY
    rawdir = os.path.basename(os.getcwd())
    rawfiles = glob.glob('DE*.fits*')


    nf=0
    for file in rawfiles:
        print(file)
        hdu    = fits.open(file)
        header = hdu[0].header
        exp_info = parse_header(header,file)

        if nf==0: allexp = exp_info
        if nf>0:  allexp = vstack([allexp,exp_info])
        nf=nf+1

    # SORT MASKS AND CREATE PLAN FILES
    masks = np.unique(allexp['MASK'])
    print(masks)
    for mask in masks:
        nmsk = allexp['MASK'] == mask
        create_planfile(allexp[nmsk],rawdir)


if __name__ == "__main__":
    main()


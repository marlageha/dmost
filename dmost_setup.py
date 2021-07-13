#!/usr/bin/env python

import numpy as np
import os, sys
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import ascii

from pypeit import pypeitsetup


######################################################
# READ DEIMOS OBJECT GOOGLE DOCUMENT
def deimos_google():
    key = '1V2aVg1QghpQ70Lms40zNUjcCrycBF2bjgs-mrp6ojI8'
    gid=1906496323
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    masklist = Table.read(url, format='csv')

    gid =0
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    objlist = ascii.read(url, format='csv')
    
    return objlist,masklist


########################################
# CREATE IDL-LIKE PLAN FOR SANITY
def read_planfile(planfile):
 
    plan = open(planfile)

    filename=[]
    for line in plan:
            pln = line.split()

            if (pln[0]=='RAWDATA:'):
                dirname = pln[1]
            
            # GET ALL FILE NAMES 
            if (pln[0]=='FLATNAME:'):
                filename = np.append(filename,pln[1])
            if (pln[0]=='ARCNAME:'):
                filename = np.append(filename,pln[1])
            if (pln[0]=='SCIENCENAME:'):
                filename = np.append(filename,pln[1])
                 
                
    return filename



########################################
def write_custom_pyfile(outfile):
    
           
    # OPEN NEW FILE AND WRITE CUSTOM FILE
    outfile = open(outfile,'w')
    outfile.write('# GENERATED BY set_up_pypeit.py\n\n')    
    outfile.write('[rdx]\n')
    outfile.write('spectrograph = keck_deimos\n')
    
    
    # SKIP DEFAULT PYPEIT FLEXURE CORRECTION  
    outfile.write('[flexure]\n')
    outfile.write('spec_method = skip \n')  
    
    # SKIP DEFAULT PYPEIT HELIOCENTRIC CORRECTION
    outfile.write('[calibrations]\n')
    outfile.write('[[wavelengths]]\n')
    outfile.write('refframe= observed\n') 
    
    # REDUCE DEFAULT PYPEIT OBJECT THRESHOLD DETECTION
    outfile.write('[reduce]\n')
    outfile.write('[[findobj]]\n')
    outfile.write('sig_thresh = 7.0\n\n')
  

    # READ DEFAULT FILE
    default_file = 'keck_deimos_A/keck_deimos_A.pypeit'

    n=0
    nline=7   # CHECK THIS!
    oldfile = open(default_file,'r')

    for line in oldfile:
        n=n+1
        if n > nline:
            outfile.write(line)
    
    outfile.close()
              
    return


############################################################
# SETUP SINGLE MASK
def setup_single_mask(msk, masklist):
    
    DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')

    
    m=masklist['MaskName'] == msk
    maskinfo = masklist[m][0]
   
    
    mask   = maskinfo['MaskName']
    dt     = str(maskinfo['DateObs'])
    rawdir = '/rawdata_'+dt[0:4]+'/'
    
    
    working_dir  = DEIMOS_REDUX+msk
    working_plan = DEIMOS_REDUX+msk+'/'+msk+'.plan'
    dropbox_plan = DEIMOS_RAW+rawdir+msk+'.plan'
    
    # CREATE DIRECTORY IF DOESNT EXIST 
    print(os.path.isdir(working_dir))
    if os.path.isdir(working_dir) != True:
        print('mkdir '+working_dir)
        os.mkdir(working_dir)
        os.mkdir(working_dir+'/dmost/')
        os.mkdir(working_dir+'/collate1d/')


    # COPY PLAN FILE IF DOESN"T EXIST
    if os.path.isfile(working_plan) != True:
        print('cp '+dropbox_plan+' '+working_dir)
        os.system('cp '+dropbox_plan+' '+working_dir) 


    # CREATE SYMBOLIC LINKS TO RAW FILES  
    filenames = read_planfile(working_plan)
    for f in filenames:

        dropbox_file = DEIMOS_RAW+rawdir+'/'+f
        if os.path.isfile(working_dir+'/'+f) != True:
            os.system('ln -s '+dropbox_file+' '+working_dir) 


    # RUN PYPEIT SETUP
    os.chdir(working_dir)
    setup = 'pypeit_setup -c=all -s keck_deimos -r .'
    os.system(setup)
    
        
    # MODIFY FILE
    pypeit_out_file = mask+'.pypeit'
    write_custom_pyfile(pypeit_out_file)
    
    
#####################################################    
#####################################################    
#####################################################    
#####################################################    
def main(*args):

    
    # READ DEIMOS OBJECT GOOGLE DOCUMENT
    objlist, masklist = deimos_google()

    for msk in sys.argv[1:]:
        print(msk)
        setup_single_mask(msk, masklist)
    
    
if __name__ == "__main__":
    main()

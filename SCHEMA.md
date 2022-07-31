# Schema for output files from dmost:
* [dmost_alldata_[object]](https://github.com/marlageha/dmost/blob/main/SCHEMA.md#dmost_alldata-schema):  Final combine output file from all DEIMOS observations of a stellar system (object)
* [dmost_[mask]](http://github.com/marlageha/dmost/blob/main/SCHEMA.md#schema-for-dmost-individual-mask-tables):  Two tables containing mask and slit information from a single DEIMOS mask


---

## dmost_alldata_[object] schema

Label | Unit | Definition
--- | --- | ---
`objname` | - | Name of target
`RA` | degree |  Target Right Ascension J2000
`DEC` | degree |  Target Declination J2000
`rproj_arcm` | arcmin |  Project radius of target from object center (arcmin)
`rproj_kpc` | kpc |  Project radius of target from object center (kpc)
`nmask` | - | Number of masks combined 
`nexp` | - | Number of individual exposures combined
`v` | kms | Heliocentric velocity  
`v_err` | kms | Heliocentric velocity error (-1 if not measured, 0 if extragalatic)
`serendip` | - | 0 if this is a object in design file, 1 if serendipitous detection
`marz_flag` | - | Visual flag set in [marz](https://samreay.github.io/Marz/#/detailed), 1 = likely star, 3 = possible galaxy, 4 = galaxy, 6 = QSO
`marz_z` | - | Redshift from marz.   This column is meaningful only if `marz_flag > 2`
`var_flag` | - | 1 if velocities are significantly variable between exposures, 0 if not


----

## dmost_[mask]: Mask schema

Label | Unit | Definition
--- | --- | ---
`maskname` | - | Name of mask
`mask_ra` | degree |  Mask center Right Ascension J2000
`mask_dec` | degree |  Mask center Declination J2000
`spec1d_filename` | - | pypeit spec1d filename in /Science
`rawfilename` | - | KOA filename of raw science data
`deimos_maskname` | - | Extended maskname
`fname` | - | Exposure number extracted from KOA name
`mjd` | seconds | MJD date at start of exposure
`year` | year | Year of observation
`exptime` | seconds | Exposure Time
`nexp` | - | Number of exposures for this mask
`vhelio` | kms | Heliocentric velocity correction (add to observed value)
`airmass` | - | Airmass at start of exposure
`slitwidth` | arcsecond | Slitwidth of science slits
`seeing` | arcsecond | Seeing determined from extracted stellar spectra
`lsf_correction` | - | Seeing-based correction to the Line Spread Profile
`telluric_h2o` | - | Telluric H2O value determined per exposure
`telluric_o2` | - | Telluric O2 value determined per exposure



## dmost_[mask]: Slit schema

Label | Unit | Definition
--- | --- | ---
`objname` | - | unique object identifier
`objid` | - | another unique object identifier
`RA` | degree |   Right Ascension J2000
`DEC` | degree | Declination J2000
`slitname` | -- | NEXP names to identify exposure in spec1d files 
`flag_skip_slit` | -- | 
`serendip` | -- | Is this a serendip? (=1)
`rms_arc` | AA | RMS of the wavelength solution
`collate1d_filename` | -- | Filename for coadded 1D spectrum
`collate1d_SN` | -- | Coadded 1D SN
`collate1d_filename` | -- | 
`collate1d_SN` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`` | -- | 
`dmost_v` | kms | Combined velocity for this slit
`dmost_v_err` | kms | Error on combined velocity


# Schema for dmost individual mask tables

## Mask schema

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



## Individual Exposure schema

Label | Unit | Definition
--- | --- | ---
`objname` | - | unique object identifier
`RA` | degree |   Right Ascension J2000
`DEC` | degree | Declination J2000
`slitname` | -- | Slit identifier, unique to each exposure
`flag_serendip` | -- | 
`flag_skip_slit` | -- | 
`flag_skip_exp` | -- | 
`spat_pixpos` | -- | 
`det` | -- | 
`rms_arc` | -- | 
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
`` | -- | 
`` | -- | 


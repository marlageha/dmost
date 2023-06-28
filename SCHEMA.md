# Schema for output files from dmost:

---

## dmost_alldata_[object].fits
 Final combine output file from all DEIMOS observations of a stellar system (object).  One row per star.

Label | Unit | Definition
--- | --- | ---
`objname` | - | Name of star target
`RA` | degree |  Target Right Ascension J2000
`DEC` | degree |  Target Declination J2000
`rproj_arcm` | arcmin |  Project radius of target from object center (arcmin)
`rproj_kpc` | kpc |  Project radius of target from object center (kpc)
`nexp` | - | Number of individual exposures combined
`t_exp` | sec | Total exposure time
`nmask` | - | Number of masks combined 
`masknames` | - | Name of individual masks
`collate1d_filename` | - | Name of associated coadded spectrum (not used for velocity analysis)
`slitwidth` | arcsecond | Slitwidth of science slits
`v` | kms | Heliocentric velocity  
`v_err` | kms | Heliocentric velocity error (-1 if not measured, 0 if extragalatic)
`SN` | - | Average per pixel SN
`marz_z` | - | Redshift from marz.   This column is meaningful only if `marz_flag > 2`
`marz_flag` | - | Extragalactic flag set in [marz](https://samreay.github.io/Marz/#/detailed), 1 = likely star, 3 = possible galaxy, 4 = galaxy, 6 = QSO
`coadd_flag` | - | 0 if velocity measured from individual exposures, 1 is coadd was required
`serendip` | - | 0 if this is a object in design file, 1 if serendipitous detection
`rmag_o`  | mag | Extinction corrected r-band magnitude from associated photometry source
`gmag_o`  | mag | Extinction corrected g-band magnitude from associated photometry source
`rmag_err`  | mag | Error on the extinction corrected r-band magnitude from associated photometry source
`gmag_err`  | mag | Error on the extinction corrected g-band magnitude from associated photometry source
`EBV`  | mag | E(B-V) value determined from SFD
`MV_o`  | mag | Extinction corrected absolute V-band magnitude 
`var_flag` | - | 1 if velocities are significantly variable between exposures, 0 if not
`ew_cat`  | Ang | Equivalent width of the combined Calcium triplet (CaT)
`ew_cat_err`  | Ang | Error on Equivalent width of the combined Calcium triplet (CaT)
`ew_NaI`  | Ang | Equivalent width of the NaI line
`ew_NaI_err`  | Ang | Error on Equivalent width of the NaI line
`ew_mgI`  | Ang | Equivalent width of the MgI line
`ew_mgI_err`  | Ang | Error on Equivalent width of the MgI line
`ew_feh`  | Ang | [Fe/H] metallicity based on CaT EW and MV -- only meaningful for RGB stars
`ew_feh_err`  | Ang | Error on [Fe/H] metallicity based on CaT EW
`gaia_source_id` | | Gaia DR3 source_id (see [GAIA DR3 schema](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html))
`gaia_pmra`  | mas/yr | Gaia DR3 proper motion RA 
`gaia_pmra_err`  | mas/yr | Gaia DR3 proper motion RA error
`gaia_pmdec`  | mas/yr | Gaia DR3 proper motion DEC
`gaia_pmdec_err`  | mas/yr | Gaia DR3 proper motion DEC error
`gaia_pmra_pmdec_corr`  | mas/yr | Gaia DR3 proper motion correlation
`gaia_parallax`  | mas | Gaia DR3 parallax
`gaia_parallax_err`  | mas | Gaia DR3 parallax error
`gaia_aen`  | mas | Gaia DR3 astrometric_excess_noise
`gaia_aen_sig`  | --- | Gaia DR3 astrometric_excess_noise significance (> 2)
`gaia_rv`  | km/s | Gaia DR3 radial_velocity 
`gaia_rv_err`  | km/s | Gaia DR3 radial_velocity error
`gaia_flag`  | --- | Set to unity if Gaia match
`vv_long_pval` | | Variability parameter between combined exposures (pval < -4 is variable)
`Pmem`  |  | Membership probability = 1 if good member star
`Pmem_pure`  |  | Membership probability = 1 if good member star, excludes velocity variables 


----

## dmost_[mask].fits

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
`dmost_v` | kms | Combined velocity for this slit
`dmost_v_err` | kms | Error on combined velocity


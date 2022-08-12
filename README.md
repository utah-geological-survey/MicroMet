# MicroMet
# Description

Scripts to process raw Eddy Covariance data for estimation of evapotranspiration and for QA/QC of micrometeorology data.

Based on Visual Basic scripts from Clayton Lewis (Utah Division of Water Resources) that were transcribed from Fortran scripts written by Dr. Lawrence Hipps (Utah State University).


# Data preparation from datalogger
All high-frequency data should be downloaded, backed up, and converted with card-convert to a CSV array. 

# Data Processing Workflow
1. Process data on the fly using EasyfluxDL; Provide immediate data through UGS portal 
2. QA/QC processed data to see if it meets quality checks
3. Reprocess data manually, focusing on low quality datasets
4. Upload refined data to Ameriflux

# Ameriflux
* [Levels of data processing](https://ameriflux.lbl.gov/data/aboutdata/data-processing-levels/)
* [Ameriflux data pipeline](https://ameriflux.lbl.gov/data/data-processing-pipelines/)

# Useful Libraries (R and Python)

## Data Prep and Processing
* https://github.com/atmos-python/atmos - An atmospheric sciences library for Python
* https://github.com/adamhsparks/EddyCleanR - Fills gaps and removes outliers in eddy covariance data.
* https://github.com/OzFlux/PyFluxPro3.2 - quality control, gap filling and partitioning of flux tower data.
* https://github.com/bgctw/REddyProc - Processing data from micrometeorological Eddy-Covariance systems 
* https://github.com/June-Spaceboots/EddyCovarianceProcessing - A collection of code used to processess Eddy Covaraince
* https://github.com/UofM-CEOS/flux_capacitor - Tools for processing flux data (eddy covariance).
* https://github.com/wsular/EasyFlux-DL-CR3000 - CR3000 datalogger program for Campbell open-path eddy-covariance systems
* https://github.com/Open-ET/flux-data-qaqc - Energy Balance Closure Analysis and Eddy Flux Data Post-Processing
* https://github.com/LI-COR/eddypro-gui - open source software application for processing eddy covariance data
* https://github.com/lsigut/openeddy - The R Package for Low Frequency Eddy Covariance Data Processing
* https://github.com/FLUXNET/ONEFlux - Open Network-Enabled Flux processing pipeline
* https://github.com/Open-ET/flux-data-qaqc - Energy Balance Closure Analysis and Eddy Flux Data Post-Processing


## Partitioning Fluxes
* https://github.com/usda-ars-ussl/fluxpart - Python module for partitioning eddy covariance flux measurements. 
* https://github.com/jnelson18/ecosystem-transpiration - Code and examples of how to estimate transpiration from eddy covariance data. 

## Remote Sensing Estimates
* https://github.com/NASA-DEVELOP/METRIC - For estimating daily evapotranspiration from Landsat data 
* https://github.com/kratzert/pyTSEB - two Source Energy Balance model for estimation of evapotranspiration with remote sensing data 
* https://github.com/spizwhiz/openet-ssebop-beta - Earth Engine SSEBop ET Model 
* https://github.com/pblankenau2/openet-core-beta - OpenET Core Components 
* https://github.com/tomchor/pymicra - A Python tool for Micrometeorological Analysis

## MISC
* https://github.com/sunxm19/Planetary_boundary_height_for_FFP - extracting boundary height within North America from NOAA regional data
* https://github.com/chuhousen/amerifluxr - An R programmatic interface for AmeriFlux data and metadata
* https://github.com/Open-ET - Open-ET packages

## Reference ET
* https://github.com/woodcrafty/PyETo - Python package for calculating reference/potential evapotranspiration (ETo).
* https://github.com/cgmorton/RefET-GEE - ASCE Standardized Reference Evapotranspiration Functions for Google Earth Engine (GEE) 
* https://github.com/usbr/et-demands - Dual crop coefficient crop water demand model
* https://github.com/Open-ET/openet-refet-gee - ASCE Standardized Reference Evapotranspiration Functions for Google Earth Engine (GEE)

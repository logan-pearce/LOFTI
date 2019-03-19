# LOFTI
#### Or: Logan's OFTI
This is my implementation of Orbits for the Impatient developed by Blunt et.al. 2017 (http://iopscience.iop.org/article/10.3847/1538-3881/aa6930/pdf).  It was developed independantly from that group's efforts based on this paper. 

This is the version of LOFTI that fits using relative astrometry of one companion around a central star.  These are the scripts I used to do the fitting and generate plots associated with the Astronomical Journal Article Pearce et. al. 2019 (Publisher: https://iopscience.iop.org/article/10.3847/1538-3881/aafacb/pdf ; also contained in this repository)


Repository contents:
  
lofti_GSC6214_mpi.py the script for fitting orbital elements to my astrometry for GSC 6214-210 system.
orbit_fitter_results_GSC6214.ipynb Jupyter notebook for analyzing fit results and generating plots *not polished! very rough!*
Custom_corner_plots_for_OFTI_results.ipynb Jupyter notebook in which I created a custom corner plot.  *Very not polished, very gross code but it works*
Pearce_2019_AJ_157_71.pdf Journal article using these scripts



## Author
The project was completed by Logan Pearce (http://www.loganpearcescience.com), under the mentorship of Dr Adam Kraus at the University of Texas at Austin.


## Acknowledgments

This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.

Software:
numpy, astropy, MySQL Connector/Python

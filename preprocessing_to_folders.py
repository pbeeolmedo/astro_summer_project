import numpy as np
import os
import send2trash
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star
from pp_functions import subclass_hist,write2pickle,numbertorun,chisq_for_filename,path_clear_and_create
import sklearn.preprocessing as skp

#fits_folder = "Data_Files/Spectra"
fits_folder = "Data_Files/Spectrum_Files"
#fits_folder = "/Volumes/Data_HDD/Spectra"

#output_folder = "/Volumes/Data_HDD"
#output_folder = "Data_Files/SEGUE"
output_folder = "/Users/Pablo/Desktop/SEGUE"

path_clear_and_create(output_folder)

numbertorun = numbertorun(fits_folder)

LOWER_CUTOFF_LOGLAM = 3.59
UPPER_CUTOFF_LOGLAM = 3.95
SPECTRUM_LENGTH = 3599
LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)

i = 0
for filename in glob.iglob(f"{fits_folder}/*.fits"):
	i +=1
	if i > numbertorun:
		break

	star = Star(filename)
	directory = f"{output_folder}/{star.spectral_subclass}/"
	os.makedirs(directory,exist_ok=True)
	filename2dump = f"{star.plate_quality_index}_{chisq_for_filename(star.chi_sq)}_{filename[-20:-5]}"

	if np.min(star.loglam_restframe) <= LOWER_CUTOFF_LOGLAM and np.max(star.loglam_restframe) >= UPPER_CUTOFF_LOGLAM:
		flux_interp = np.interp(LOGLAM_GRID, star.loglam_restframe, star.flux)
		normalised_flux = flux_interp/np.max(flux_interp)
		np.save(f"{directory}{filename2dump}",np.array(normalised_flux))
		print(f"{i}|Subclass:{star.spectral_subclass}")
	else:
		np.save(f"{output_folder}/Bad_Wavelength_Range_Error/{filename2dump}",np.array([0]))
		print(f"{i}|Iteration skipped : bad wavelength range")


# Useful lines/prints for debugging/ troubleshooting --------------
#!/usr/bin/env python
import time
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import os
import glob
from stellarpy import Star
from pp_functions import chisq_for_filename,path_clear_and_create,numbertorun,list_index_splitter

def hpc_pp2folder(list=[0,-1]):
	fits_folder="/Volumes/Data_HDD/Spectra"
	#fits_folder = "Data_Files/Spectrum_Files"
	output_folder="/Users/Pablo/Desktop/SEGUE"
	error_folder_name = f"{output_folder}/Error"
	start = list[0]
	end = list[1]

	LOWER_CUTOFF_LOGLAM = 3.59
	UPPER_CUTOFF_LOGLAM = 3.95
	SPECTRUM_LENGTH = 3599
	LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)
	i = start
	for filename in glob.glob(f"{fits_folder}/*.fits")[start:end]:
		i += 1
		star = Star(filename)
		spectral_subclass = star.spectral_subclass #this star property is v.important -> given its own variable

		if spectral_subclass == "" or None:
			spectral_subclass = "No-Subclass"
			directory = f"{error_folder_name}/"
		else:
			directory = f"{output_folder}/{spectral_subclass}/"

		os.makedirs(directory,exist_ok=True)
		filename2dump = f"{star.plate_quality_index}_{chisq_for_filename(star.chi_sq)}_{spectral_subclass}_{filename[-20:-5]}"

		if np.min(star.loglam_restframe) <= LOWER_CUTOFF_LOGLAM and np.max(star.loglam_restframe) >= UPPER_CUTOFF_LOGLAM:
			flux_interp = np.interp(LOGLAM_GRID, star.loglam_restframe, star.flux)
			processed_flux = flux_interp
			np.save(f"{directory}{filename2dump}",np.array(processed_flux))
			print(f"{i}|Subclass:{spectral_subclass}")
		else:
			np.save(f"{error_folder_name}/{filename2dump}",np.array([0]))
			print(f"{i}|Iteration skipped : bad wavelength range")
# HPC parts --------------

fits_folder="/Volumes/Data_HDD/Spectra"
#fits_folder = "Data_Files/Spectra"
#fits_folder = "Data_Files/Spectrum_Files"

#output_folder = "/Volumes/Data_HDD/SEGUE"
#output_folder = "Data_Files/SEGUE"

output_folder="/Users/Pablo/Desktop/SEGUE"
error_folder_name = path_clear_and_create(output_folder)

numFiles = numbertorun(fits_folder,False)
inputs = list_index_splitter(numFiles,8)
print(inputs)
num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores-1)(delayed(hpc_pp2folder)(i) for i in inputs)

# Useful lines/prints for debugging/ troubleshooting --------------

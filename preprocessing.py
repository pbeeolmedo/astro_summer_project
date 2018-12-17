from astropy.table import Table
import numpy as np
import glob
import pandas as pd
from stellarpy import Star

flux_values = []
subclasses=[]

MAX_NUM_FILES = len(glob.glob("Data_Files/Spectra/*.fits"))
numbertorun = int(input(f"Enter number of files to run (max = {MAX_NUM_FILES}) : "))


UPPER_CUTOFF_LOGLAM = 3.95
LOWER_CUTOFF_LOGLAM = 3.59
SPECTRUM_LENGTH = 3599
MAX_CHI = 3

LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)
print(f" The spectrum grid values (of wavelengths) to be used for the pre-processing are : {LOGLAM_GRID}")

fits_folder = "Data_Files/Spectra"

for i in range(numbertorun):
	star = Star(glob.glob(fits_folder + "/*.fits")[i])
	if(star.subclass and star.chi_sq <= MAX_CHI and star.plate_quality == "good"):
		if np.min(star.loglam_restframe) <= LOWER_CUTOFF_LOGLAM and np.max(star.loglam_restframe) >= UPPER_CUTOFF_LOGLAM:
			flux_interp = np.interp(LOGLAM_GRID, star.loglam_restframe, star.flux)
			normalised_flux = flux_interp/np.max(flux_interp)
			flux_values.append(normalised_flux)
			subclasses.append(star.subclass)
			print("Subclass: "+ star.subclass)
			#star.spectrum_plot()
			print(star.z)
		else:
			print("Bad wavelength Range")
	else:
		print(f"Iteration skipped : subclass = {star.subclass} chi_sq = {star.chi_sq} plate quality = {star.plate_quality}")

processed_data_df = pd.DataFrame(flux_values)

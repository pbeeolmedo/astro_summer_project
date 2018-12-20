import numpy as np
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star
import pickle
import sklearn.preprocessing as skp

fits_folder = "Data_Files/Spectra"
#fits_folder = "/Volumes/Data_HDD/Spectra "

MAX_NUM_FILES = 0
for file in glob.iglob(f"{fits_folder}/*.fits"):
		MAX_NUM_FILES += 1
numbertorun = input(f"Enter number of files to run (max = {MAX_NUM_FILES}):")

if numbertorun:
	numbertorun = int(numbertorun)
else:
	numbertorun = 10

LOWER_CUTOFF_LOGLAM = 3.59
UPPER_CUTOFF_LOGLAM = 3.95
SPECTRUM_LENGTH = 3599
MAX_CHI = 3

LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)
print(f" The spectrum grid values (of wavelengths) to be used for the pre-processing are : {LOGLAM_GRID}")

flux_values = []
subclasses_list=[]
classes_list = []
subclasses_counter = Star.all_subclasses_dict
for i in range(numbertorun):
	star = Star(glob.glob(f"{fits_folder}/*.fits")[i])
	if(star.spectral_subclass and star.chi_sq <= MAX_CHI and star.plate_quality == "good"):
		if np.min(star.loglam_restframe) <= LOWER_CUTOFF_LOGLAM and np.max(star.loglam_restframe) >= UPPER_CUTOFF_LOGLAM:
			flux_interp = np.interp(LOGLAM_GRID, star.loglam_restframe, star.flux)
			normalised_flux = flux_interp/np.max(flux_interp)
			flux_values.append(normalised_flux)
			subclasses_list.append(star.spectral_subclass)
			classes_list.append(star.spectral_class)
			subclasses_counter[star.spectral_subclass] += 1
			print(f"{i}|Subclass:{star.spectral_subclass}")
			#star.spectrum_plot()
		else:
			print(f"{i}|Iteration skipped : bad wavelength range")
	else:
		print(f"{i}|Iteration skipped : subclass = {star.spectral_subclass} chi_sq = {star.chi_sq} plate quality = {star.plate_quality}")

subclasses_set = set(subclasses_list)

# Subclass Histogram Draft -----------
print(subclasses_counter)
plt.bar(range(len(subclasses_counter)),list(subclasses_counter.values()),align='center')
plt.xticks(range(len(subclasses_counter)),Star.all_subclasses)
plt.xlabel("Stellar Spectral Subclasses")
plt.ylabel("Count")
plt.title(f"Subclass Histogram for nubertorun = {numbertorun} ")
plt.show()
scaled_flux = skp.RobustScaler().fit_transform(flux_values)
#processed_dataframe = pd.DataFrame(flux_values)
#print(processed_data_df)

# Pickle write --------
data2dump = [flux_values,subclasses_list]

filename2dump = f"data-{len(subclasses_list)}-{len(subclasses_set)}-{MAX_CHI}"
with open(filename2dump,"wb") as file:
	pickle.dump(data2dump,file)

print(f"File dumped is {os.path.getsize(filename2dump)/1e6} megabytes")

# Read Pickle file -------
'''
with open(filename2dump,"rb") as file:
	testdata = pickle.load(file)
print(testdata)
print(data2dump)
print(f" Is chosen point the same in both? : {testdata[0][6][2360]==data2dump[0][6][2360]}")
'''

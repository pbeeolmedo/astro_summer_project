import numpy as np
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star
from pp_functions import subclass_hist,write2pickle,numbertorun
import sklearn.preprocessing as skp

#fits_folder = "Data_Files/Spectra"
fits_folder = "Data_Files/Spectrum_Files"
#fits_folder = "/Volumes/Data_HDD/Spectra"

numbertorun = numbertorun(fits_folder)

LOWER_CUTOFF_LOGLAM = 3.59
UPPER_CUTOFF_LOGLAM = 3.95
SPECTRUM_LENGTH = 3599
MAX_CHI = 3
LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)

i = 0
flux_values = []
subclasses_list=[]
classes_list = []
subclasses_counter = Star.all_subclasses_dict
subclasses_dict = Star.all_subclasses_dict
other_keys = []


for filename in glob.iglob(f"{fits_folder}/*.fits"):
	i +=1
	if i > numbertorun:
		break
	star = Star(filename)
	if(star.spectral_subclass and star.chi_sq <= MAX_CHI and star.plate_quality == "good"):
		if np.min(star.loglam_restframe) <= LOWER_CUTOFF_LOGLAM and np.max(star.loglam_restframe) >= UPPER_CUTOFF_LOGLAM:
			flux_interp = np.interp(LOGLAM_GRID, star.loglam_restframe, star.flux)
			normalised_flux = flux_interp/np.max(flux_interp)
			flux_values.append(normalised_flux)
			subclasses_list.append(star.spectral_subclass)
			classes_list.append(star.spectral_class)
			if star.spectral_subclass in subclasses_counter:
				subclasses_counter[star.spectral_subclass] += 1
			else:
				other_keys.append(star.spectral_subclass)
			print(f"{i}|Subclass:{star.spectral_subclass}")
			#star.spectrum_plot()
		else:
			print(f"{i}|Iteration skipped : bad wavelength range")
	else:
		print(f"{i}|Iteration skipped : subclass = {star.spectral_subclass} chi_sq = {star.chi_sq} plate quality = {star.plate_quality}")

# Pickle write -------------
subclasses_set = set(subclasses_list)
#scaled_flux = skp.RobustScaler().fit_transform(flux_values)
data2dump = [flux_values,subclasses_list]
filename2dump = f"Data_Files/data-{len(subclasses_list)}-{len(subclasses_set)}-{MAX_CHI}"
write2pickle(data2dump,filename2dump)

# Useful lines/prints for debugging/ troubleshooting --------------
print(subclasses_counter)
print(f"other keys = {other_keys}")

# Subclass Histogram Draft ------------------
subclass_hist(Star.all_subclasses_dict,Star.all_subclasses,numbertorun)
plt.show()

# Create and print Data Frame -----------------
#processed_dataframe = pd.DataFrame(flux_values)
#print(processed_data_df)

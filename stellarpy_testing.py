#!/usr/bin/env python
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star,FluxMatrix
from pp_functions import filename_data,subclass_hist

#path = '/Users/Pablo/Desktop/SEGUE'
path = '/Volumes/Data_HDD/SEGUE'

filenametest1 = '1_01-28_A0p_3130-54740-0430.npy'
filenametest2 = '1_01-31_G0_3144-54763-0191.npy'

data = FluxMatrix(path)
print(data.folder)
subclasses = data.subclass_list

# ------- Histogram Plot----------------
print(data.subclass_hist_dict)
subclass_hist(data.subclass_hist_dict,Star.all_subclasses,)
plt.show()

print(subclasses)

[matrix,overall_count,subclass_counter] = data.load_flux_matrix(min_files=20,max_files=100,max_chisq=2.3,plate_quality_choice=[1],exclusion_list=['WD'])
print(matrix)
subclass_hist(subclass_counter,Star.all_subclasses,overall_count)
plt.show()

print(filenametest2)
print(filename_data(filenametest2))

#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star,FluxMatrix
from pp_functions import filename_data,subclass_hist,write2pickle
from sys import getsizeof
import glob

path = 'Data_Files/SEGUE'
#path = '/Volumes/Data_HDD/SEGUE'


data = FluxMatrix(path)
print(data.folder)
subclasses = data.subclass_list

LOWER_CUTOFF_LOGLAM = 3.59
UPPER_CUTOFF_LOGLAM = 3.95
SPECTRUM_LENGTH = 3599
#LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)
a = np.load('LOGLAM_GRID.npy')
print(a)

# ------- Histogram Plot----------------
'''
print(data.subclass_hist_dict)
subclass_hist(data.subclass_hist_dict,Star.all_subclasses,'',semi_log=True,png_file=f"Data_Files/Histogram")
plt.show()


print(subclasses)
inclusion_list = [ s for s in subclasses if s.startswith('M')]
inclusion_list.append('B6')
inclusion_list.append('B9')
[matrix,overall_count,subclass_counter,plate_ids_used] = data.load_flux_matrix(2,100,1.4,[1],['WD','CV','Carbon'],inclusion_list)
print(matrix)
print(subclass_counter)
print(plate_ids_used)
np.save(f"Data_Files/{overall_count}",np.array(matrix))
subclass_hist(subclass_counter,Star.all_subclasses,'hello',True,f"Data_Files/Histogram")
plt.show()


output = data.load_flux_matrix(min_files=1,max_files=100,max_chisq=2.0,subclasses=['F9'])
subclass_hist(output[2],Star.all_subclasses,'hello',True,f"Data_Files/Histogram")
plt.show()
print(f"Size of output is::{getsizeof(output[0])}::{getsizeof(output[1])}::{getsizeof(output[2])}::{getsizeof(output[3])}")
write2pickle(output,f"Data_Files/OUTPUT_{output[1]}.bin")


for subclass in subclasses:
    maxChiSq = 1.5
    maxFiles = 100
    if data.subclass_hist_dict[subclass] < maxFiles:
        maxChiSq = 7
    output = data.load_flux_matrix(min_files=1,max_files=maxFiles,max_chisq=maxChiSq,subclasses=[f"{subclass}"])
    print(f"output is::{output[2]}::")
    print(f"Size of output is::{getsizeof(output[0])}::{getsizeof(output[1])}::{getsizeof(output[2])}::{getsizeof(output[3])}")
    median = np.median(output[0][0],axis=0)
    mean = np.mean(output[0][0],axis=0)
    print(median)
    print(mean)
    np.save(f"Data_Files/Average/Matrix_{subclass}",np.array(output[0]))
    np.save(f"Data_Files/Average/Median_{subclass}",np.array(median))
    np.save(f"Data_Files/Average/Mean_{subclass}",np.array(mean))
'''

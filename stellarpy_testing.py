#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from stellarpy import Star,FluxMatrix
from pp_functions import filename_data,subclass_hist,write2pickle
from sys import getsizeof
import glob
# has to be absolute path
path = 'Data_Files/SEGUE'
#path = '/Volumes/Data_HDD/SEGUE'

#a = glob.glob(f"{path}/K5/*.npy")

filenametest1 = '1_01-28_A0p_3130-54740-0430.npy'
filenametest2 = '1_01-31_G0_3144-54763-0191.npy'

data = FluxMatrix(path)
print(data.folder)
subclasses = data.subclass_list

# ------- Histogram Plot----------------

print(data.subclass_hist_dict)
subclass_hist(data.subclass_hist_dict,Star.all_subclasses,'',input_log=True)
plt.show()
print(subclasses)

[matrix,overall_count,subclass_counter] = data.load_flux_matrix(min_files=4,max_files=100,max_chisq=2.3,plate_quality_choice=[1],exclusion_list=['WD','CV','Carbon'])
print(matrix)
print(subclass_counter)
np.save(f"Data_Files/{overall_count}",np.array(matrix))
subclass_hist(subclass_counter,Star.all_subclasses,'hello',input_log=True)
plt.show()


output = data.load_flux_matrix(min_files=1000,max_files=15000,max_chisq=3.0,plate_quality_choice=[1],exclusion_list=['WD'])
print(f"Size of output is : {getsizeof(output[0])},{getsizeof(output[1])},{getsizeof(output[2])}")
write2pickle(output,f"Data_Files/OUTPUT_{output[1]}.bin")



#subclass_hist(subclass_counter1,Star.all_subclasses,'hello')
#plt.show()


#print(filenametest2)
#print(filename_data(filenametest2))

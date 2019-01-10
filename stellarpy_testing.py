#!/usr/bin/env python
from stellarpy import FluxMatrix
from pp_functions import filename_data

filenametest1 = '1_01-28_A0p_3130-54740-0430.npy'
filenametest2 = '1_01-31_G0_3144-54763-0191.npy'

data = FluxMatrix('/Users/Pablo/Desktop/SEGUE')
print(data.folder)
subclasses = data.subclass_list
print(data.subclass_hist_dict)
print(subclasses)

flux_values = data.load_flux_matrix(29,183)
print(flux_values)

print(filenametest2)
print(filename_data(filenametest2))

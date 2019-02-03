#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


path = 'Data_Files/SEGUE'
#path = '/Volumes/Data_HDD/SEGUE'
subclass = 'A0p'
base_filename = '1_01-09_A0p_2052-53401-0618'
listSubclass = []
for i in range(11):
    listSubclass.append(np.load(f"{path}/{subclass}/{base_filename}_{i}.npy"))
print(listSubclass)


x = range(len(listSubclass[0]))

plt.xlabel("wavelength index")
plt.ylabel("flux")
plt.title(f"{base_filename}  -  test graph")
for i in range(len(listSubclass)):
    plt.plot(x,listSubclass[i],label = 'id %s'%i)
plt.legend()
plt.show()








exit()

data = FluxMatrix(path)
subclasses = data.subclass_list

subclasses = [ s for s in subclasses if s.startswith('B')]

for subclass in subclasses:
    i = 0
    subclass_folder_npy_files = glob.glob(f"{self.folder}/{subclass}/*.npy")
    shuffle(subclass_folder_npy_files)
    for npy_file in subclass_folder_npy_files:
        print(npy_file)

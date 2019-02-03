#!/usr/bin/env python

path = 'Data_Files/SEGUE'
#path = '/Volumes/Data_HDD/SEGUE'


data = FluxMatrix(path)
subclasses = data.subclass_list

subclasses = [ s for s in subclasses if s.startswith('B')]

for subclass in subclasses:
    i = 0
    subclass_folder_npy_files = glob.glob(f"{self.folder}/{subclass}/*.npy")
    shuffle(subclass_folder_npy_files)
    for npy_file in subclass_folder_npy_files:

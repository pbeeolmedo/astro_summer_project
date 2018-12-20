# Some functions to help with preprocessing
from matplotlib import pyplot as plt
import pickle
import os
from glob import iglob
# ['WD', 'A0p', 'A0p', 'WD', 'WD', 'WD', 'A0p', 'CV', 'WD', 'A0p', 'WD', 'WD', 'WD', 'A0p', 'A0p',
#'A0p', 'WD', 'A0p', 'A0p', 'WD', 'A0p', 'CV', 'WD', 'A0p', 'CV', 'A0p', 'WD', 'A0p',
#'WD', 'A0p', 'WD', 'WD', 'WD', 'WD', 'A0p', 'WD', 'CV', 'A0p', 'WD', 'WD', 'WD', 'CV', 'A0p']

def numbertorun(fits_folder):
    MAX_NUM_FILES = 0
    for file in iglob(f"{fits_folder}/*.fits"):
    		MAX_NUM_FILES += 1
    numbertorun = input(f"Enter number of files to run (max = {MAX_NUM_FILES}):")

    if numbertorun:
    	return int(numbertorun)
    else:
    	return 10
        
def subclass_hist(dictionary,ordered_bin_labels,numbertorun):
    plt.bar(range(len(dictionary)),list(dictionary.values()),align='center')
    plt.xticks(range(len(dictionary)),ordered_bin_labels)
    plt.xlabel("Stellar Spectral Subclasses")
    plt.ylabel("Count")
    plt.title(f"Subclass Histogram for numbertorun = {numbertorun} ")

def write2pickle(data2dump,filename2dump):
    yesno = input(f"Pickle this yes or no (y/n)?")
    if yesno == 'y':
    	with open(filename2dump,"wb") as file:
            pickle.dump(data2dump,file)
    	print(f"File [{filename2dump}] dumped is {os.path.getsize(filename2dump)/1e6} megabytes")

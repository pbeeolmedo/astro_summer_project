# Some functions to help with preprocessing
from matplotlib import pyplot as plt
import pickle
import os
import re
import send2trash
from glob import iglob

def numbertorun(fits_folder):
    MAX_NUM_FILES = 0
    for file in iglob(f"{fits_folder}/*.fits"):
    		MAX_NUM_FILES += 1
    numbertorun = input(f"Enter number of files to run (max = {MAX_NUM_FILES}):")

    if numbertorun:
    	return int(numbertorun)
    else:
    	return 10

def path_clear_and_create(output_folder):
    if os.path.isdir(output_folder):
    	yesno = input(f"Are you sure you want to delete {output_folder} (y/n) ? :")
    	if yesno == "n":
    		raise Exception(f"{output_folder} was not deleted as asked." +
                            f"Retry scrpt with different a output_folder or delete old one ")
    	else:
    		send2trash.send2trash(output_folder)
    error_folder_name = f"{output_folder}/Error"
    os.makedirs(error_folder_name)
    return error_folder_name

def chisq_for_filename(chi_sq):
    integer = int(chi_sq)
    decimal = int(100*(chi_sq - integer))
    return f"{integer:02d}-{decimal:02d}"

def continuum_normalise(flux_values):
    contNorm_flux_values = flux_values
    return contNorm_flux_values

def subclass_hist(dictionary,ordered_bin_labels,number_of_stars='no input given'):
    plt.bar(range(len(dictionary)),list(dictionary.values()),align='center')
    plt.xticks(range(len(dictionary)),ordered_bin_labels,rotation='vertical')
    plt.xlabel("Stellar Spectral Subclasses")
    plt.ylabel("Count")
    plt.title(f"Subclass Histogram: Number of Stars = {number_of_stars} ")

def write2pickle(data2dump,filename2dump):
    yesno = input(f"Pickle this yes or no (y/n)? :")
    if yesno == 'y':
    	with open(filename2dump,"wb") as file:
            pickle.dump(data2dump,file)
    	print(f"File [{filename2dump}] dumped is {os.path.getsize(filename2dump)/1e6} megabytes")

# ------------------- POST 'SEGUE' FOLDER CREATION ------------------

def filename_data(filename=None):
    if filename is None:
        raise FileNotFoundError("Filename not specified")
    plate_quality = int(filename[0])
    chi_sq_integer = int(filename[2:4])
    chi_sq_decimal = int(filename[5:7])
    chi_sq = chi_sq_integer+(chi_sq_decimal/100)
    subclass = re.match('([^_]*)',filename[8:]).group(0)
    unique_id = filename[-19:-4]
    filetype = filename[-3:]
    return [plate_quality,chi_sq,subclass,unique_id,filetype]

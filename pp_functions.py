# Some functions to help with preprocessing
from matplotlib import pyplot as plt
import os
import re
import send2trash
from glob import iglob
import pickle
import numpy as np

def numbertorun(fits_folder,usr_input=True):
    MAX_NUM_FILES = 0
    for file in iglob(f"{fits_folder}/*.fits"):
    		MAX_NUM_FILES += 1
    if usr_input is True:
        numbertorun = input(f"Enter number of files to run (max = {MAX_NUM_FILES}):")
    else:
        numbertorun = MAX_NUM_FILES
    if numbertorun:
    	return int(numbertorun)
    else:
    	return 10

def path_clear_and_create(output_folder):
    if os.path.isdir(output_folder):
        yesno = input(f"Are you sure you want to delete {output_folder} (y/n) ? :")
        if yesno == "n":
            raise Exception(f"{output_folder} was not deleted as asked."+
            f"Retry scrpt with different a output_folder or delete old one ")
        else:
            send2trash.send2trash(output_folder)
    error_folder_name = f"{output_folder}/Error"
    os.makedirs(error_folder_name)
    return error_folder_name

def list_index_splitter(length_list,chunks=1):
    inputs = []
    len_chunks = int(length_list/chunks)
    for i in range(chunks):
        start = i*(len_chunks)
        if i == chunks-1:
            end = length_list
        else:
            end = (i+1)*(len_chunks)
        inputs.append([start,end])
    return inputs

def chisq_for_filename(chi_sq):
    integer = int(chi_sq)
    decimal = int(100*(chi_sq - integer))
    return f"{integer:02d}-{decimal:02d}"

def continuum_normalise(flux_values):
    contNorm_flux_values = flux_values
    return contNorm_flux_values

def subclass_hist(dictionary,ordered_bin_labels,title='no input given',semi_log=False,png_file=None):
    plt.bar(range(len(dictionary)),list(dictionary.values()),align='center',log=semi_log,zorder=3)
    plt.xticks(range(len(dictionary)),ordered_bin_labels,rotation='vertical')
    plt.xlabel("Stellar Spectral Subclasses")
    plt.ylabel("Count")
    plt.title(f"Subclass Histogram: Number of Stars = {sum(dictionary.values())}: {title} ")
    plt.grid(which="both",color='lightgrey',zorder=0,ls='--')
    if png_file is not None: plt.savefig(f"{png_file}-{sum(dictionary.values())}.png")

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

def flux_pprocessing(flux_values=None,method='max'):
    if flux_values is None:
        raise TypeError("Function requires 'flux_values' arrays as input (not None).")
    if method == 'div_max':
        processed_flux = flux_values/np.max(flux_values)
    elif method == 'div_median':
        processed_flux = flux_values/np.median(flux_values)
    elif method == 'div_mean':
        processed_flux = flux_values/np.mean(flux_values)
    elif method == 'minus_median':
        processed_flux = flux_values-np.median(flux_values)
    elif method == 'minus_median_div_max':
        processed_flux = flux_values-np.median(flux_values)
        processed_flux /= np.max(processed_flux)
    return processed_flux
# ------------------- POST 'MATRIX' CREATION ------------------

def train_test_split():
    return

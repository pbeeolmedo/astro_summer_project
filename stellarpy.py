#!/usr/bin/env python
from astropy.table import Table
import numpy as np
import os
import fnmatch
import re
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from pp_functions import filename_data,flux_pprocessing

class FluxMatrix(object):
    # Spectrum 2d
    def __init__(self,SEGUE_folder=None):
        if SEGUE_folder is None:
            raise FileNotFoundError("FITS file not specified")
        self.folder = SEGUE_folder
        self.loglamgrid = np.load(f'{SEGUE_folder}/LOGLAM_GRID.npy')
        self.subclass_list = [f.name for f in os.scandir(self.folder) if (f.is_dir()) and (not re.match('Error.*',f.name))]
        self.subclass_hist_dict = Star.all_subclasses_dict
        for subclass in self.subclass_list:
            subclass_directory = f"{self.folder}/{subclass}/"
            self.subclass_hist_dict[subclass] = len(fnmatch.filter(os.listdir(subclass_directory), '*.npy'))

    def load_flux_matrix(self,min_files=1,max_files=99999,min_chisq=0,max_chisq=9,plate_quality_choice=[1,0],\
                         exclusion_list=[''],inclusion_list=[''],subclasses=None,loglam=[0,3599],pp_method='div_max'):
        if subclasses is None:
            subclasses = self.subclass_list
        flux_values = []
        subclass_list =[]
        plate_ids_used = []
        overall_count = 0
        subclass_counter = dict(zip(self.subclass_hist_dict.keys(),[0]*len(self.subclass_hist_dict))) # setting all vals to zero
        print(subclasses)

        for subclass in subclasses:
            i = 0
            if (subclass in exclusion_list):
                print(f"{subclass}: Omitted : Exclusion list = {exclusion_list} ")
                continue
            if self.subclass_hist_dict[subclass] < min_files:
                print(f"{subclass}: Omitted : {self.subclass_hist_dict[subclass]} files but " +
                      f"minimum = {min_files}")
                continue
            for npy_file in glob.iglob(f"{self.folder}/{subclass}/*.npy"):
                if i == max_files:
                    print(f"{subclass}: Iteration number {i} and max is {max_files} ")
                    break

                filename = re.search('([^[\/]*$)',npy_file).group(0)
                [plate_quality,chi_sq,subclass,id,filetype] = filename_data(filename)

                if (subclass not in inclusion_list) and (not (min_chisq<chi_sq<max_chisq) or (plate_quality not in plate_quality_choice)):
                    print(f"{subclass}: Omitted : X^2 is {chi_sq:.2f} but " +
                          f"X^2 range is {min_chisq}:{max_chisq} or PQ is {plate_quality} but not in {plate_quality_choice}")
                    continue

                flux_array = np.load(npy_file)
                #processed_flux_array = flux_array/np.max(flux_array) #comment out after ppfunc is done
                processed_flux = flux_pprocessing(flux_array,method=pp_method)
                flux_values.append(processed_flux[loglam[0]:loglam[1]])
                subclass_list.append(subclass)
                plate_ids_used.append(id)
                i += 1
                if overall_count%50 == 0 : print(f"Overall count is {overall_count}")
                overall_count += 1
                subclass_counter[subclass] += 1

            matrix = [flux_values,subclass_list]
        return [matrix,overall_count,subclass_counter,plate_ids_used]


class Star(object):
    # Star object which requires a 'filename.fits' input and reads the HDUs it has
    def __init__(self,filename=None):
        if filename is None:
            raise FileNotFoundError("FITS file not specified")
        self.filename = filename
        self.hdu1 = Table.read(self.filename,hdu=1)
        self.hdu2 = Table.read(self.filename,hdu=2)

    all_classes = ["B","A","F","G","K","M"]
    all_subclasses = ['O','OB','B6','B9','A0','A0p','F2','F5','F9','G0','G2','G5','K1','K3','K5',\
                      'K7','M0','M0V','M1','M2','M2V','M3','M4','M5','M6','M7','M8','L0','L1','L2','L3',\
                      'L4','L5','L5.5','L9','T2','Carbon','WD','CV','STARFORMING']
    all_subclasses_dict = dict(zip(all_subclasses,[0]*len(all_subclasses)))

    # HDU 2 obtained properties
    @property
    def plate_quality(self):
        plate_q = self.hdu2['PLATEQUALITY'].data[0]
        return plate_q.strip().decode("utf-8")

    @property
    def plate_quality_index(self):
        # index is one (1) if plateQ is good else its zero (0)
        index = 0
        if self.plate_quality == "good":
            index = 1
        return index

    @property
    def spectral_subclass(self):
        subclass = self.hdu2['SUBCLASS'].data[0]
        return subclass.strip().decode("utf-8")

    @property
    def spectral_class(self):
        return self.spectral_subclass[0]

    @property
    def ra(self):
        return self.hdu2['PLUG_RA'].data

    @property
    def dec(self):
        return self.hdu2['PLUG_DEC'].data

    @property
    def chi_sq(self):
        return self.hdu2['RCHI2'].data[0]

    @property
    def z(self):
        return self.hdu2['Z'].data[0]

    # HDU 1 obtained properties
    @property
    def loglam(self):
        return self.hdu1['loglam'].data

    @property
    def flux(self):
        return self.hdu1['flux'].data

    # Other
    @property
    def loglam_restframe(self):
        loglam_rf = self.loglam - np.log10(self.z+1)
        return loglam_rf

    def spectrum_plot(self):
        ''' Plots flux vs restframe wavelengths'''
        plt.subplot(1,2,1)
        plt.scatter(self.loglam, self.flux, c='green', s=1)
        plt.xlabel("Loglam (observed) ")
        plt.ylabel("Flux")
        plt.title("Original Spectra for "+ self.subclass)
        plt.subplot(1,2,2)
        plt.scatter(self.loglam_restframe,self.flux, c='red', s=1)
        plt.xlabel("Loglam (restframe)")
        plt.ylabel("Flux")
        plt.title("Shifted Spectra for "+ self.subclass)
        plt.tight_layout()
        plt.show()

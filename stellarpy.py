from astropy.table import Table
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from collections import OrderedDict

class Spectrum_2D(object):
    # Spectrum 2d
    def __init__(self,loglam_obs,flux_obs):
        self.loglam_obs = loglam_obs
        self.flux_obs = flux_obs

class Star(object):
    # Star ...
    def __init__(self,filename=None):
        if filename is None:
            raise FileNotFoundError("FITS file not specified")
        self.filename = filename
        self.hdu1 = Table.read(self.filename,hdu=1)
        self.hdu2 = Table.read(self.filename,hdu=2)

    all_classes = ["O","B","A","F","G","K","M"]
    all_subclasses = ['O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9',\
                  'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',\
                  'A0', 'A1','A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',\
                  'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9',\
                  'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9',\
                  'K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9',\
                  'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    all_subclasses_dict = dict(zip(all_subclasses,[0]*len(all_subclasses)))

    # HDU 2 obtained properties
    @property
    def plate_quality(self):
        plate_q = self.hdu2['PLATEQUALITY'].data[0]
        return plate_q.strip().decode("utf-8")

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

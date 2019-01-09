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

    all_classes = ["O","B","A","F","G","K","M","L"]
    all_subclasses = ['O','OB','B6','B9','A0','A0p','F2','F5','F9','G0','G2','G5','K1','K3','K5',\
                      'K7','M0','M0V','M2V','M1','M2','M3','M4','M5','M6','M7','M8','L0','L1','L2','L3',\
                      'L4','L5','L5.5','L9','T2','C','WD','CV']
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

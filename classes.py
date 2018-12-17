from astropy.table import Table
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# magnitude
#ra and dec
# specttral params


class Spectrum_2D(object):
    # Spectrum
    def __init__(self,loglam_obs,flux_obs):
        self.loglam_obs = loglam_obs
        self.flux_obs = flux_obs
            
class Star(object):
    # Star ...
    def __init__(self,filename):
        if filename is None:
            raise FileNotFoundError("FITS file not specified")
        self.filename = filename

    @property
    def plate_quality(self):
        plate_q = Table.read(self.filename,hdu=2)['PLATEQUALITY'].data[0]
        return plate_q.strip().decode("utf-8")

    @property
    def star_subclass(self):
        star_subclass = Table.read(self.filename,hdu=2)['SUBCLASS'].data[0]
        return star_subclass.strip().decode("utf-8")

    @property
	def ra(self):
		return Table.read(self.filename,hdu=2)['PLUG_RA'].data

    @property
	def dec(self):
		return Table.read(self.filename,hdu=2)['PLUG_DEC'].data

    @property
    def chi_sq(self):
        return Table.read(fname,hdu=2)['RCHI2'].data[0]

    @property
    def z(self):
        return Table.read(fname,hdu=2)['Z'].data[0]

    @property
    def loglam(self):
        return Table.read(self.filename,hdu=1)['loglam'].data

    @property
    def flux(self):
        return Table.read(self.filename,hdu=1)['flux'].data

    @property
    def loglam_restframe(self):
        loglam_rf = self.loglam - np.log10(self.z+1)
        return loglam_rf

    @property
    def spectrum_plot(self):
        ''' Plots flux vs restframe wavelengths'''
    	plt.subplot(1,2,1)
    	plt.scatter(self.loglam, self.flux, c='green', s=1)
    	plt.xlabel("Loglam (observed) ")
    	plt.ylabel("Flux")
    	plt.title("Original Spectra for "+ self.star_subclass)
    	plt.subplot(1,2,2)
    	plt.scatter(self.loglam_restframe,self.flux, c='red', s=1)
    	plt.xlabel("Loglam (restframe)")
    	plt.ylabel("Flux")
    	plt.title("Shifted Spectra for "+ self.star_subclass)
    	plt.tight_layout()
    	plt.show()

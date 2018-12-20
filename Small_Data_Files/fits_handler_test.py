from astropy.table import Table
from astropy.io import fits
import numpy as np

t = Table.read('spec-3106-54714-0471.fits',2)
hdu = fits.open('spec-3106-54714-0471.fits')
print(hdu[2].header)
print("a")


#print(hdu[2].header)

#loglam = [1,2,3,4,5,6,7,8,9]
#b = np.array(a)

#c = np.where((loglam<max_cutoff) & (loglam>min_cutoff))
#print(b[c])

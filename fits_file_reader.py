from astropy.table import Table
import numpy as np
import glob
from matplotlib import pyplot as plt

numbertorun = 50

# t = Table.read(filename,hdu=1)
# Initialise
i = 0
flux_values = []
loglam_values = []

min_loglam = []
max_loglam = []



while i < numbertorun:
    fname = glob.glob("Spectra/*.fits")[i]
    table = Table.read(fname,hdu=1)
    loglam = table["loglam"].data
    flux = table["flux"].data

    flux_values.append(flux)
    loglam_values.append(loglam)
    min_loglam.append(np.min(loglam))
    max_loglam.append(np.max(loglam))

    i +=1
'''
plt.scatter(list(range(numbertorun)),min_loglam)
plt.scatter(list(range(numbertorun)),max_loglam)
plt.plot([0,50],[np.mean(min_loglam),np.mean(min_loglam)])
plt.plot([0,50],[np.mean(max_loglam),np.mean(max_loglam)])
plt.show()
'''
print(np.mean(max_loglam))

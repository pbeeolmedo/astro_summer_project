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

for i in range(numbertorun):
    print(i)
    fname = glob.glob("/Users/Pablo/OneDrive - UNSW/4th Year/Summer Physics /Research/02_Data_Files/Spectra/*.fits")[i]
    table = Table.read(fname,hdu=1)
    loglam = table["loglam"].data
    flux = table["flux"].data

    flux_values.append(flux)
    loglam_values.append(loglam)
    min_loglam.append(np.min(loglam))
    max_loglam.append(np.max(loglam))


plt.scatter(list(range(numbertorun)),min_loglam)
plt.scatter(list(range(numbertorun)),max_loglam)
plt.plot([0,50],[np.mean(min_loglam),np.mean(min_loglam)])
plt.plot([0,50],[np.mean(max_loglam),np.mean(max_loglam)])
plt.show()

print(np.mean(max_loglam))

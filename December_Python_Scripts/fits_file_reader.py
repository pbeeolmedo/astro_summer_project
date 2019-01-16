from astropy.table import Table
import numpy as np
import glob
from matplotlib import pyplot as plt

numbertorun = 2
filename = "/Users/Pablo/OneDrive - UNSW/4th Year/Summer Physics/Research/01_AstroProject_Main/Small_Data_Files/spec-3106-54714-0471.fits"
t = Table.read(filename,hdu=1)
print(t)

'''
# Initialise
i = 0
flux_values = []
loglam_values = []

min_loglam = []
max_loglam = []

for i in range(numbertorun):
    #print(i)
    fname = glob.glob("../02_Data_Files/Spectra/*.fits")[i]
    table = Table.read(fname,hdu=1)
    loglam = table["loglam"].data
    flux = table["flux"].data
    flux_values.append(flux)
    loglam_values.append(loglam)

    plt.plot(loglam,flux)
    min_loglam.append(np.min(loglam))
    max_loglam.append(np.max(loglam))



plt.scatter(list(range(numbertorun)),min_loglam)
plt.scatter(list(range(numbertorun)),max_loglam)
plt.plot([0,numbertorun],[np.mean(min_loglam),np.mean(min_loglam)])
plt.plot([0,numbertorun],[np.mean(max_loglam),np.mean(max_loglam)])
plt.show()

print('jj')
print(np.mean(min_loglam))
print(np.mean(max_loglam))
# Chose min = 3.59 and max = 3.95
'''

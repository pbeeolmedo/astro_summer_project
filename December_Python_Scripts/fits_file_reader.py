from astropy.table import Table
import numpy as np
import glob
from matplotlib import pyplot as plt

numbertorun = 2
filename = "Data_Files/Spectrum_files/spec-3105-54825-0170.fits"
t1 = Table.read(filename,hdu=1)
print(t1.colnames)

flux = t1['flux'].data
print(type(flux))
print(f"Flux is :{flux}")
ivar = t1['ivar'].data
print(f"Ivar is :{ivar}")
avg_ivar = np.mean(ivar)
ivar[np.isin(ivar,0.)]=avg_ivar

sigmas = np.sqrt(1/ivar)
print(f"Std = {sigmas}")

noise = np.random.normal(0, sigmas)
flux_w_noise = flux + noise
print(f"Flux and noise is : {flux_w_noise}")

flux_w_noise2 = flux + np.random.normal(0, np.sqrt(1/ivar))
print(f"Flux and noise is : {flux_w_noise2}")
t2 = Table.read(filename,hdu=2)

print(t2.colnames)
print(t2['ELODIE_TEFF'])
print(t2['Z'])
t3 = Table.read(filename,hdu=3)
print(t3.colnames)



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

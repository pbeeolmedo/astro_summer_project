from astropy.table import Table
import numpy as np
import glob
from matplotlib import pyplot as plt
import pandas as pd
from scipy import constants as const
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input

numbertorun = 15

# t = Table.read(filename,hdu=1)
# Initialise
i = 0
flux_values = []
loglam_values = []
upper_cutoff_loglam = 3.95
lower_cutoff_loglam = 3.59
df = pd.read_csv('Data_Files/segue_dataquery.csv')
rad_vels = df['elodiervfinal']
good_spec_info=[]
subclasses=[]

for i in range(numbertorun):
	fname = glob.glob("Data_Files/Spectra/*.fits")[i]
	hdu1 = Table.read(fname,hdu=1)
	plate_quality = Table.read(fname,hdu=2)['PLATEQUALITY'].data[0].strip().decode("utf-8")
	subclass = Table.read(fname,hdu=2)['SUBCLASS'].data[0].strip().decode("utf-8")
	loglam = hdu1["loglam"].data
	flux = hdu1["flux"].data  
	
	if plate_quality == "good":
		if np.min(loglam) <= lower_cutoff_loglam and np.max(loglam) >= upper_cutoff_loglam:
			good_indices = np.where((loglam>lower_cutoff_loglam) & (loglam<upper_cutoff_loglam))
			loglam = loglam[good_indices]
			flux = flux[good_indices]
			rad_vel = rad_vels[i]/const.c
			doppler_factor = np.sqrt((1+rad_vel)/(1-rad_vel))
			loglam_shifted = doppler_factor*loglam
			flux_interp = np.interp(loglam_shifted, loglam, flux)
			normalised_flux = flux_interp/np.max(flux_interp)
			loglam_values.append(loglam_shifted)
			flux_values.append(normalised_flux)
			good_spec_info.append(df.iloc[[i]].values)
			subclasses.append(subclass)
			plt.subplot(1,2,1)
			plt.scatter(loglam, flux, c='green', s=1)
			plt.xlabel("Loglam")
			plt.ylabel("Flux")
			plt.title("Original Spectra for "+subclass)
			plt.subplot(1,2,2)
			plt.scatter(loglam_shifted, normalised_flux, c='red', s=1)
			plt.xlabel("Loglam")
			plt.ylabel("Normalised Flux")
			plt.title("Shifted Spectra for "+subclass)
			plt.tight_layout()
			#plt.show()
		else:
			print("Bad wavelength Range")
	else:
		print(plate_quality)
			

X = np.array(loglam_values)
y = np.array(subclasses)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)



def model():
	a = Input(shape=(3599))
	b = Dense(128, activation='relu')(a)
	c = Dense(1, activation='sigmoid')(b)
	model=Model(inputs = [a], outputs = [c])
	return model
	
model=model()
model.summary()
	
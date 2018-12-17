from astropy.table import Table
import numpy as np
import glob
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
from scipy import constants as const
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras import optimizers
import sklearn.preprocessing as skp
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA


NUMBER_TO_RUN = 1058
BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE =0.001
MAX_NUM_FILES = len(glob.glob("Data_Files/Spectra/*.fits"))
# t = Table.read(filename,hdu=1)
# Initialise
i = 0
flux_values = []
UPPER_CUTOFF_LOGLAM = 3.95
LOWER_CUTOFF_LOGLAM = 3.59
SPECTRUM_LENGTH = 3599
MAX_CHI = 3
df = pd.read_csv('Data_Files/segue_dataquery.csv')
rad_vels = df['elodiervfinal']
good_spec_info=[]
subclasses=[]
LOGLAM_GRID = np.linspace(LOWER_CUTOFF_LOGLAM, UPPER_CUTOFF_LOGLAM, SPECTRUM_LENGTH)
print(LOGLAM_GRID)

def shifted_spec_plotter(loglam, flux, loglam_shifted, normalised_flux):
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
	plt.show()

chi_vals=[]
num_points=[]
for i in range(NUMBER_TO_RUN):
	fname = glob.glob("Data_Files/Spectra/*.fits")[i]
	hdu1 = Table.read(fname,hdu=1)
	plate_quality = Table.read(fname,hdu=2)['PLATEQUALITY'].data[0].strip().decode("utf-8")
	subclass = Table.read(fname,hdu=2)['SUBCLASS'].data[0].strip().decode("utf-8")
	chi = Table.read(fname,hdu=2)['RCHI2'].data[0]
	z = Table.read(fname,hdu=2)['Z'].data[0]
	if(subclass and chi<=MAX_CHI):
		loglam_obs = hdu1["loglam"].data
		flux = hdu1["flux"].data
		if plate_quality == "good":
			loglam_em = loglam_obs - np.log10(z+1)
			if np.min(loglam_em) <= LOWER_CUTOFF_LOGLAM and np.max(loglam_em) >= UPPER_CUTOFF_LOGLAM:
				flux_interp = np.interp(LOGLAM_GRID, loglam_em, flux)
				normalised_flux = flux_interp/np.max(flux_interp)
				flux_values.append(normalised_flux)
				#flux_values.append(flux_interp)
				#rad_vel = rad_vels[i]/const.c
				#doppler_factor = np.sqrt((1+rad_vel)/(1-rad_vel))
				#loglam_shifted = doppler_factor*loglam_obs
				#flux_interp = np.interp(loglam_shifted, loglam_obs, flux)
				#normalised_flux = flux_interp/np.max(flux_interp)
				#flux_values.append(normalised_flux)
				good_spec_info.append(df.iloc[[i]].values)
				subclasses.append(subclass)
				print("Subclass: "+subclass)
				#shifted_spec_plotter(loglam_obs, flux, LOGLAM_GRID, normalised_flux)	
			else:
				print("Bad wavelength Range")
		else:
			print(plate_quality)

	else:
		print("No subclass or chi too large: "+str(chi))
processed_data_df = pd.DataFrame(flux_values)
#data_nparray = processed_data_df.values
scaled_flux = skp.RobustScaler().fit_transform(flux_values)
#normalised_flux = skp.Normalizer().fit_transform(scaled_flux)

pca = PCA()

principalComponents = pca.fit_transform(scaled_flux)
print(principalComponents[0])
print(principalComponents[1])
print(principalComponents[2])
print(pca.explained_variance_ratio_)
x=np.arange(1,len(pca.explained_variance_ratio_)+1,1)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.show()

#print(X_pca)

X = list(zip(*[principalComponents[0], principalComponents[1], principalComponents[2], principalComponents[3], principalComponents[4], principalComponents[5], principalComponents[6], principalComponents[7]]))
#X = np.array(scaled_flux)
y = np.array(subclasses)

unique_labels = set(subclasses)
encoder = LabelBinarizer()
one_hot = encoder.fit_transform(list(unique_labels))
label_dict = dict(zip(unique_labels, one_hot))
y_one_hot = []
[y_one_hot.append(label_dict[label]) for label in y]
X_train, X_test, y_train, y_test = train_test_split(X[:(len(X)-5)], y_one_hot[:(len(y_one_hot)-5)], test_size=0.2)
X_test2 = X[:(len(X)-5)]
y_test2 = y_one_hot[:(len(y_one_hot)-5)]
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

def model():
	a = Input(shape=(8,))
	b = Dense(128, activation='relu')(a)
	c = Dense(64, activation='relu')(b)
	d = Dense(len(unique_labels), activation='softmax')(c)
	model=Model(inputs = [a], outputs = [d])
	return model

model=model()
model.summary()
oad = optimizers.Adam(lr=LEARNING_RATE, epsilon=None, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=oad, metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_test,y_test))

#scree test
#PCA
#CNN
#regularisation
#grid search
#grid search on cluster for CNN????
#Fabbro et al. (2018)
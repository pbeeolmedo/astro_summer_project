import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def equal_class_splitter(X, y, show_counts=False, test_split=0.2):
	unique_labels, y_encoded, label_dict, unique, counts = label_encoder(y)
	X_train=[]
	y_train=[]
	X_val=[]
	y_val=[]
	X_test=[]
	y_test=[]
	for subclass in unique:
		mask = np.isin(y, subclass)
		training_flux, X_testing, y_training_flux, y_testing = train_test_split(X[mask], y_encoded[mask], test_size=test_split, shuffle=True)
		X_train.extend(training_flux)
		y_train.extend(y_training_flux)
		
		X_val_flux, X_test_flux, y_val_flux, y_test_flux = train_test_split(X_testing, y_testing, test_size=test_split, shuffle=True)
		X_val.extend(X_val_flux)
		y_val.extend(y_val_flux)
		X_test.extend(X_test_flux)
		y_test.extend(y_test_flux)

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_val = np.array(X_val)
	y_val = np.array(y_val)
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	
	if(show_counts):
		unique_train, counts_train = np.unique(y_train, return_counts=True)
		unique_val, counts_val = np.unique(y_val, return_counts=True)
		unique_test, counts_test = np.unique(y_test, return_counts=True)
		print(label_dict)
		print(f"Train count: \n {unique_train} \n {counts_train}")
		print(f"Val count: \n {unique_val} \n {counts_val}")
		print(f"Test count: \n {unique_test} \n {counts_test}")
		
	return (X_train, y_train, X_val, y_val, X_test, y_test, unique_labels, label_dict)

def label_encoder(y):
	unique_labels = set(y)
	encoder = LabelEncoder()
	encoded_labels = encoder.fit_transform(list(unique_labels))
	label_dict = dict(zip(unique_labels, encoded_labels))
	y_encoded = []
	[y_encoded.append(label_dict[label]) for label in y]
	y_encoded=np.array(y_encoded)
	unique, counts = np.unique(y, return_counts=True)
	return unique_labels, y_encoded, label_dict, unique, counts
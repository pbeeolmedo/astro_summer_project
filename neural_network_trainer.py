import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras import optimizers
import sklearn.preprocessing as skp
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Define some constants: to be later determind via grid search
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 200

#Open pickle file containing spectra flux values and matching subclasses
with open ('data-924-11-3.bin', 'rb') as training_data_file:
    training_data = pickle.load(training_data_file)

#Create the flux and subclass label lists from the training data
flux_values = training_data[0][:]
subclasses = training_data[1]	
	
X = np.array(flux_values)
y = np.array(subclasses)

#One hot encode the spectral class labels
unique_labels = set(subclasses)
encoder = LabelBinarizer()
one_hot = encoder.fit_transform(list(unique_labels))
label_dict = dict(zip(unique_labels, one_hot))
y_one_hot = []
[y_one_hot.append(label_dict[label]) for label in y]

#Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X[:len(X)], y_one_hot[:len(y_one_hot)], test_size=0.2, shuffle=True)
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])

lr=[0.005, 0.001]
batch_size=[32, 128]
epochs=[200]
hu1=[128, 512]
hu2=[32, 128]
hu3=[8, 64]
d1=[0.2, 0.5]
d2=[0.2, 0.5]

parameters = dict(lr=lr, batch_size=batch_size, epochs=epochs, hu1=hu1, hu2=hu2, hu3=hu3, d1=d1, d2=d2)

#Define the neural network model
def model(lr=0.001, batch_size=64, epochs=200, hu1=128, hu2=64, hu3=32, d1=0.2, d2=0.2):
	model=Sequential()
	model.add(Dense(hu1, activation='relu', input_dim=X_train.shape[1]))
	model.add(Dropout(d1))
	model.add(Dense(hu2, activation='relu'))
	model.add(Dropout(d2))
	model.add(Dense(hu3, activation='relu'))
	model.add(Dense(len(unique_labels), activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#Run the model
#model.summary()
#oad = optimizers.Adam(lr=learning_rate, epsilon=None, amsgrad=False)
#model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val,y_val))

KC_model = KerasClassifier(build_fn=model, verbose=1)

grid = GridSearchCV(estimator=KC_model, param_grid=parameters, verbose=1)
grid_search = grid.fit(X_train, y_train)
print(grid_result.best_params_)


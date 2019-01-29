import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import EarlyStopping
from keras import optimizers
import sklearn.preprocessing as skp
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from collections import Counter
from sklearn.utils import class_weight
from functools import partial
import keras.backend as K
from itertools import product


#Open pickle file containing spectra flux values and matching subclasses
with open ('Data_Files/2680-minus_median.bin', 'rb') as training_data_file:
    training_data = pickle.load(training_data_file)


#Create the flux and subclass label lists from the training data
flux_values = training_data[0][:]
subclasses = training_data[1]	

updated_subclasses=[]
for subclass in subclasses:
	if 'M' in subclass:
		updated_subclasses.append('M')
	else:
		updated_subclasses.append("Other")

print(len(training_data[0][0]))

X = np.array(flux_values)
y = np.array(updated_subclasses)


#One hot encode the spectral class labels
unique_labels = set(updated_subclasses)
print(unique_labels)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(list(unique_labels))
label_dict = dict(zip(unique_labels, encoded_labels))
y_encoded = []
[y_encoded.append(label_dict[label]) for label in y]
y_encoded=np.array(y_encoded)

#Split data into training, validation and testing sets
X_train, X_testing, y_train, y_testing = train_test_split(X[:len(X)], y_encoded[:len(y_encoded)], test_size=0.4, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_testing, y_testing, test_size=0.3, shuffle=True)

subclass_counter = dict(Counter(updated_subclasses))

subclass_weights={}
for k, v in subclass_counter.items():
	if(v!=0):
		subclass_weights[label_dict[k]] = 1/v
	else:
		subclass_weights[label_dict[k]]=0

print(label_dict)
		
subclass_weights = {2: 10, 17: 10, 18: 10, 3: 33, 1: 3, 14: 3.1, 15:20,
					16: 14, 19: 32, 0:1, 5: 1, 6:1, 7:1, 
					8: 1, 9:1, 10:1, 11:1,12:1,13:1, 4:10, 20:10}
		
#sk_class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
#sk_class_weight_dict = dict(enumerate(sk_class_weights))
#print(label_dict)
#print(sk_class_weight_dict)

lr=0.0002
batch_size=100
epochs=150
hu1=256
hu2=128
d1=0.25
d2=0.25

#Define the neural network model
def model():
	model=Sequential()
	model.add(Dense(hu1, activation='relu', input_dim=X_train.shape[1]))
	model.add(Dropout(d1))
	model.add(Dense(hu2, activation='relu'))
	model.add(Dropout(d2))
	model.add(Dense(len(unique_labels), activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#Define early stopping callback 
es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')
	
#Run the model
model = model()
model.summary()
oad = optimizers.Adam(lr=lr, beta_1=0.999, beta_2=0.999, epsilon=None, amsgrad=False)
model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val,y_val))
score = model.evaluate(X_val, y_val, verbose=1)
print(model.metrics_names)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('MClassifier.h5')
y_preds = model.predict(X_test, batch_size=32, verbose=1)
pred_classes = []
classes_in_test_set = set(y_test)

for p in y_preds:
	pred_classes.append(np.argmax(p))

cm = confusion_matrix(y_test, pred_classes)

sub=[]
sub_encoded=[]
for k,v in label_dict.items():
	if v in classes_in_test_set:
		sub.append(k)
		sub_encoded.append(v)

sorted_sub, sorted_num = zip(*sorted(zip(sub, sub_encoded)))
classes = list(sorted_sub)

#The following code was copied from scikit_learn docs at:
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cm, classes=classes, normalize=True)
plt.show()	

# lr=[0.005]
# batch_size=[100, 128, 150]
# epochs=[200]
# hu1=[100, 128, 150]
# hu2=[32, 64, 100]
# hu3=[32, 64, 100]
# d1=[0.2, 0.25, 0.3]
# d2=[0.2, 0.25, 0.3]
# parameters = dict(lr=lr, batch_size=batch_size, epochs=epochs, hu1=hu1, hu2=hu2, hu3=hu3, d1=d1, d2=d2)


# KC_model = KerasClassifier(build_fn=model, verbose=1)
# grid = GridSearchCV(cv=3, estimator=KC_model, njobs=-1, param_grid=parameters, verbose=1)
# grid_search = grid.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)


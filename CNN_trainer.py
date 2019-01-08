import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, add
from keras import optimizers
import sklearn.preprocessing as skp
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import generate_random_color as grc
import itertools

#Define some constants: to be later determind via grid search
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 200

#Open pickle file containing spectra flux values and matching subclasses
with open ('data-17621-19-3.bin', 'rb') as training_data_file:
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
X_train, X_testing, y_train, y_testing = train_test_split(X[:len(X)], y_one_hot[:len(y_one_hot)], test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_testing, y_testing, test_size=0.2, shuffle=True)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

print(label_dict)

hyperparams = [0,32,2,2,0.5,32,2,4,0.5,0.007,32,5]
print(len(y_train))
print(len(y_val))
print(len(y_test))
filter1 = int(hyperparams[1])
kernel1 = int(hyperparams[2])
pool1 = int(hyperparams[3])
dropout1 = float(hyperparams[4])
filter2 = int(hyperparams[5])
kernel2 = int(hyperparams[6])
pool2 = int(hyperparams[7])
dropout2 = float(hyperparams[8])
learning_rate = float(hyperparams[9])
batch_size = int(hyperparams[10])
epochs = int(hyperparams[11])

#parameters = dict(lr=lr, batch_size=batch_size, epochs=epochs, hu1=hu1, hu2=hu2, hu3=hu3, d1=d1, d2=d2)



#Define the neural network model
def model():
	a = Input(shape=(X_train.shape[1],1))
	b = Conv1D(filter1, kernel1, activation='relu',data_format='channels_last', padding = 'same')(a)
	c = MaxPooling1D(pool_size=(pool1))(b)
	d = Dropout(dropout1)(c)
	e = Conv1D(filter2, kernel2, activation='relu', padding='same') (d)
	f = MaxPooling1D(pool_size=(pool2))(e)
	g = Dropout(dropout2)(f)
	h = Flatten()(g)
	i = Dense(128,activation='relu') (h)
	j = Dense(len(unique_labels), activation='softmax')(i)
	model=Model(inputs = [a], outputs = [j])
	return model

#Run the model
model = model()
model.summary()
oad = optimizers.Adam(lr=learning_rate, epsilon=None, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=oad, metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val,y_val))

y_preds = model.predict(X_test, batch_size=32, verbose=1)
pred_classes = []
classes_in_test_set = set(np.argmax(y_test, axis=1))
print(classes_in_test_set)

for p in y_preds:
	pred_classes.append(np.argmax(p))

test_class = np.argmax(y_test, axis=1)	
cm = confusion_matrix(test_class, pred_classes)
print(cm)
print(label_dict)

sub=[]
sub_one_hot=[]
for k,v in label_dict.items():
	if np.argmax(v) in classes_in_test_set:
		sub.append(k)
		sub_one_hot.append(np.argmax(v))

sorted_sub, sorted_num = zip(*sorted(zip(sub, sub_one_hot)))
print(sorted_sub)
print(sorted_num)
print(test_class[:])
print(pred_classes[:])
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

	
plot_confusion_matrix(cm, classes=classes)
plt.show()	
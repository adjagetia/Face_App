#!/bin/env python3
#SBATCH -N 1 # No. of computers you wanna use. Typically 1
#SBATCH -n 2 # No. of CPU cores you wanna use. Typically 1
#SBATCH -p gpu # This flag specifies that you wanna use GPU and not CPU
#SBATCH -o homework.out # output file name, in case your program has anything to output (like print, etc)
#SBATCH -t 24:00:00 # Amount of time
#SBATCH --gres=gpu:2 # No. of GPU cores you wanna use. Usually 2-3
import numpy
import pandas
import cv2
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

K = []
# load the dataset
dataframe = pandas.read_csv('dataset/face.csv', header=None)
dataset = dataframe.values

X_old = dataset[:,0]
for i in X_old:
	K.append(cv2.imread(i))
X=numpy.array(K)

Y = dataset[:,1]
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
print(Y)
print(encoder_Y)
dummy_y = np_utils.to_categorical(encoder_Y)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
X_train = X[0:train_size]
X_test = X[train_size:len(dataset)]
Y_train = encoder_Y[0:train_size]
Y_test = encoder_Y[train_size:len(dataset)]
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 50, 50).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 3, 50, 50).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]
# print(num_classes)
def model():
	# create model
	model = Sequential()
	model.add(Conv2D(32,3, 3, border_mode='same', input_shape=(3, 50, 50), activation='relu'))
	model.add(Dropout(0.15))
	model.add(Conv2D(32,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.15))
	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# serialize model to JSON
	#print("Saved model to disk")
	return model
# build the model
model = model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
#print("Saved model to disk")

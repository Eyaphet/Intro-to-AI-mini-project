
import numpy
import cv2
import os
import pandas
from threading import Thread
import sklearn
import sklearn.model_selection
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tensorflow import keras
from keras import layers
from cv2 import INTER_AREA

def getmodel():
	modeldir = os.path.join('Datadir','model.model')
	if(os.path.isfile(modeldir)):
		return tf.keras.models.load_model(modeldir)
	else:
		return None

def getname(ID):
	directory = os.path.join('Datadir','mappingfile.csv')
	if(not os.path.isfile(directory)):
		print("test")
		return None
	mappingfile = pandas.read_csv(directory)
	
	for i in range(len(mappingfile['ID'])):
		
		if(str(mappingfile['ID'][i]) == str(ID)):
			print(mappingfile['Name'][i])
			return mappingfile['Name'][i]


	
def getlabel():
	directory = 'Datadir'
	dirs=[x[1] for x in os.walk(directory)]
	return dirs[0]

def checkdir():
	directory = 'Datadir'
	if (not os.path.isdir(directory)):
		os.mkdir(directory)
	
def getid():
		directory = 'Datadir'
		file = os.path.join(directory,'mappingfile.csv')
		if( os.path.isfile(file) == False):
			ID = 13579
		else:
			name_data = pandas.read_csv(file)
			if(len(name_data) == 0):
				ID = 3579
			else:
				print(name_data)
				print(name_data['ID'].iloc[-1])
				ID = name_data['ID'].iloc[-1] + 1  
		return ID
	
def updateidfile(name,ID):
		directory = 'Datadir'
		file = os.path.join(directory,'mappingfile.csv')
		values = {'Name': name, 'ID':ID}
		if( os.path.isfile(file) is False):
			filedata = pandas.DataFrame(values,index=[0])
			#filedata = numpy.array(filedata)
		else:
			filedata = pandas.read_csv(file)
			print(filedata.values)
			if(ID in filedata.values):
				return
			filedata = filedata.append(values, ignore_index = True)
			#addname
		filedata.to_csv(file, index = False)
	
def updatemodel():
	#model = getuntrainedmodel(len(getlabel()))
	model = getanothermodel(len(getlabel()))
	trainmodel(model)
	directory = 'Datadir'
	filename = os.path.join(directory,'model.model')
	model.save(filename)
	#save model
	

def trainmodel(model):
	X_train,X_test, Y_train,Y_test = load_data()
	X_train,Y_train,X_test,Y_test = numpy.array(X_train), numpy.array(Y_train), numpy.array(X_test), numpy.array(Y_test)
	print(X_train.shape, Y_train.shape)
	X_train = X_train.reshape(len(X_train),50,50,1)
	X_test = X_test.reshape(len(X_test),50,50,1)
	model.fit(X_train, Y_train,epochs=10)
	#preparing the data to use with tflearn
	
	#new_y_train = []
	#new_y_test = []
	#k = len(getlabel())
	#for i in Y_train:
	#	encoded = numpy.zeros(k)
	#	encoded[i] = 1
	#	new_y_train.append(encoded)
	#for i in Y_test:
	#	encoded = numpy.zeros(k)
	#	encoded[i] = 1
	#	new_y_test.append(encoded)
	#print(new_y_train)
	#model.fit(X_train, new_y_train, n_epoch=100, validation_set=(X_test, new_y_test), show_metric = True) 


def getanothermodel(num):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=[50,50,1]))
	model.add(tf.keras.layers.Conv2D(64,3,activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
	model.add(tf.keras.layers.Dense(32,activation="relu"))
	model.add(tf.keras.layers.Conv2D(128,3,activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=3))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1024,activation="relu"))
	model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(num,activation="softmax"))
	model.compile(
		optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"]
	)
	print(model.summary())
	return model


	

	
	

		
def getuntrainedmodel(num):
	convnet = input_data(shape=[50,50,1])
	convnet = conv_2d(convnet, 32, 3, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	convnet = conv_2d(convnet, 128, 3, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	convnet = conv_2d(convnet, 64, 3, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)
	convnet = fully_connected(convnet, num, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
	model = tflearn.DNN(convnet, tensorboard_verbose=1)
	return model

def load_data(): #returns (X_train,Y_train,X_test,Y_test)
	dirs = getlabel()
	data = []
	label = []
	l = 0
	for i in dirs:
		filedirectory = os.path.join('Datadir',i)
		for k in range(1,41): #in range of the number of pics
			newdir = os.path.join(filedirectory,str(k)+'.jpg')
			image = cv2.imread(newdir, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (50,50), interpolation = INTER_AREA)
			imagearray = numpy.array(image) 
			imagearray.reshape(50,50,1)
			data.append(imagearray)
			label.append(l)
		l+=1
	
	return sklearn.model_selection.train_test_split(data, label, test_size=0.20, random_state=42)
	
		
def createIDdir(ID):
	directory = 'Datadir'
	directory = os.path.join(directory,str)
	os.makdir(directory)


def getdir(ID):
	directory = 'Datadir'
	if (not os.path.isdir(directory)):
		os.mkdir(directory)
	directory = os.path.join(directory, str(ID))
	if (not os.path.isdir(directory)):
		os.mkdir(directory)
	return directory   

#updatemodel()
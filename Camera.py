import numpy
import cv2
import os
import pandas
from threading import Thread
import sklearn
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression

from utilities import getmodel,getlabel,checkdir,getid,updateidfile,updatemodel,trainmodel,getuntrainedmodel,load_data,getdir,getname

class Camera():
	def __init__(self,name):
		self.name = name
		self.ID = getid()
		self.picnum = 40 #number of pictures to take
		self.faceclassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		self.savedpics = 0
		self.model = getmodel()
		print("here")
		self.vid = cv2.VideoCapture(0)
		
		
		self.collected = []
	
	def destroy(self):
		self.vid.release()
		

	def getframe(self): #output (status (0 for not done and 1 for done), frame)
		checkdir()
		updateidfile(self.name,self.ID)
		directory = getdir(self.ID)

		
	   # vid = cv2.VideoCapture(0)
		ret, frame = self.vid.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.faceclassifier.detectMultiScale(gray, 1.3, 5)
		for x,y,w,h in faces:#should remove this since we are checking for only one face
			face = gray[y:y+h,x:x+w]
			if faces is not None:
				self.savedpics += 1
				face = cv2.resize(face, (200,200))
				file_name_path = os.path.join(directory,str(self.savedpics) + ".jpg")
				cv2.imwrite(file_name_path, face)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		ret, retframe = cv2.imencode('.png', frame)#change this
		if(self.savedpics >= self.picnum):#done collecting the data
			#create a thread and run it here
			self.destroy()
			new_thread = Thread(target=updatemodel)
			new_thread.start()
			return (1,retframe.tobytes())
		else:
			return (0,retframe.tobytes())
		
	
		  
	def getdir(ID):
		directory = 'Datadir'
		if (not os.path.isdir(directory)):
			os.mkdir(directory)
		directory = os.path.join(directory, str(ID))
		if (not os.path.isdir(directory)):
			os.mkdir(directory)
		return directory    
	
	
	def detectface(self): #returns (status (1:found,0:searching,-1:not found,2:morethan two people, 3: no model), frame,id)
		ret, frame = self.vid.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.faceclassifier.detectMultiScale(gray, 1.3, 5)
		self.model = getmodel()
		if(self.model is None):
			self.destroy()
			return (3,0,0,None)
		status =0
		if (len(faces)>1):
			status = 2
		for x,y,w,h in faces:#should remove this since we are checking for only one face
			face = gray[y:y+h,x:x+w]
			if faces is not None:
				face = cv2.resize(face, (50,50))
				#face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				
				face = numpy.reshape(face,(50,50,1))
				
				newfaces = []
				newfaces.append(face)
				newfaces = numpy.array(newfaces)
				print(newfaces.shape)
				result = self.model.predict(newfaces)
				print(result[0])
				print(numpy.argmax(result[0]))
				if(result[0][numpy.argmax(result[0])]<0.50): #could be changed to increase accuracy
					label = 'Unknown'
					self.collected.append(-1)
				else:
					label = getlabel()[numpy.argmax(result)]
					self.collected.append(numpy.argmax(result))
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		ret, retframe = cv2.imencode('.png', frame)
		if(len(self.collected)==10):
			self.destroy()
			freqcount = dict()
			for i in self.collected:
				if i in freqcount.keys():
					freqcount[i] += 1
				
				else:
					freqcount[i] = 1
			print(freqcount)

			k = max(freqcount, key = freqcount.get)
			print(k)
			if(freqcount[k] >= 5 and status!= 2):#succesfully found a match
				print(getname(getlabel()[k]))
				return (1,retframe.tobytes(),getlabel()[k],getname(getlabel()[k]))
			elif(status == 2):
				return (status,retframe.tobytes(),getlabel()[k],getname(getlabel()[k]))
			else:
				return (-1,retframe.tobytes(),0,None)

		return (0,retframe.tobytes(),0,None)
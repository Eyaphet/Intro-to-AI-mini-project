# Intro-to-AI-mini-project
#Problem Description

A system that can be deployed in in-doors premises for both identity control and watching for suspects is required to be developed for a security company. This system is a facial recognition system that employs machine learning and works with normal camera and Kinect camera. 

#Introduction
Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed, [Arthur Samuel (1959)]. In different industries, machine learning has paved the way for technological accomplishments ranging its applications from database mining, self-customizing programs, to computer vision. Face recognition is one application of machine learning where it works to identify people using their facial image. Basically, the system captures a 2D or a 3D image of the face, extracts from it a facial representation (a set of metrics representing the face), then matches this representation with one corresponding the claimed identity. 

 


#Solution 

The system we built is a web application. Once this web application is run, the page for login or signup appears. The user can choose to login if they have an existing account created, but initially they must sign up. When the user chooses to signup, a frame appears that takes several pictures of the user which are used to train the model. This means each time a user signs up, a thread runs to train the model. After the signup process is completed, a status is sent from the server and a successful signup message appears on the status bar of the page. After that, the user can return to the login page to log in. When the user chooses to log in, a frame appears that takes a few pictures of the user, to avoid misclassification, and identifies the user gives them access. 


#Generating the data 
As face recognition is a direct application of computer vision, OpenCV, an open-source computer vision library that includes numerous computer vision algorithms, is used. OpenCV provides pretrained models, that can be read using a load method. First, a cv2.CascadeClassifier() is created and the ‘haarcascade_frontalface_default.xml’ file is loaded. Afterwards, the detection is done using the cv2.CascadeClassifier.detectMultiScale() method, which returns boundary rectangles for the detected faces. Haar Cascade Classifiers are basically a machine learning based approaches where a cascade function is trained from a lot of images both positive and negative, and based on the training, are then used to detect the objects in other images. For the case of this project, the ‘haarcascade_frontalface_default.xml’ is the file that was used as it contains the features set to detect the frontal face. Then, data is split into training and testing sets. 
Training the model: Convolutional Neural Network (CNN)
Convolutional Neural Network is a form of neural network that offers better performance with image recognition. It consists of three main layers called convolutional layer, pooling layer and fully connected layer.  
Convolutional layer is the layer where most of the computation takes place. It requires input data, a filter, and a feature map [1]. The input image is made of 3D matrix of pixels corresponding to RGB image. The filter is a 2D array of weights representing part of the image; hence, the process known as convolution is when the filter (feature detector) swipes across the receptive fields or the matrix looking for a specific feature. Next, the dot product is calculated between the input pixels and the filter which is then fed into an output array. Afterwards, the filter shifts by a stride, repeating the process until the filter has swept across the entire image [1]. The output array is the feature map. 
 

The activation function used in this layer is the rectified linear activation function/unit (ReLU). ReLU is a function that outputs the input if the input is greater than 0, and outputs 0 if the input is less than or equal to 0, as shown in the figure below (fig. 2).  ReLU is preferred as the activation function in this layer since it allows models to learn faster by providing the effective ability to backpropagate the gradient information to the input layers of the model while other activation functions, the sigmoid and tanh, suffer from the vanishing gradient problem.
 

The vanishing gradient problem occurs because as the number of nodes or layers in the neural network increases, the gradient of the loss function becomes smaller (vanishing) making the update on the weights small during back propagation, and this leads to slow learning of the model.  The model used has the padding set to zero (same padding). This padding ensures that the output has the same shape as the input data, and it is achieved by adding zeros at the edges of the output matrix.

#Pooling 
Pooling summarizes a set of adjacent units from the preceding layer with a single value by simply aggregating values within a respective field to populate the output array. It uses the same technique as the preceding layer, but the filter in this layer does not have any weights. Instead, the aggregation is carried out by making the filter either select the pixel with the maximum value to send to the output array, called max pooling or calculate the average value within the receptive field to send to the output array, called average pooling. The solution uses max pooling.

 

Fully connected layer is a feed forward neural network. The input to the fully connected layer is the output from the final Pooling or Convolutional Layer, which is flattened and fed into it. The output from the final or any Pooling and Convolutional Layer is a 3-dimensional matrix, to flatten that is to unroll all its values into a vector. The pixel values of the input image are not directly connected to the output layer in partially connected layers (pooling and convolutional layers). However, in the fully connected layer, each node in the output layer connects directly to a node in the previous layer. This layer performs the task of classification based on the features extracted through the previous layers and their different filters. While convolutional and pooling layers use ReLU functions, fully connected layers usually leverage a softmax activation function to get the probabilities of the input being a particular class.

#Dropout
Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. This method prevents the network from being unnecessarily highly dependent on specific weights which in turn prevents overfitting, a phenomena where a model models the training data too well that it won’t perform in data that are not part of the training.

#Loss function
The loss function used in our solution is the Sparse categorical cross entropy. This categorical cross entropy is a loss function that is used in multiclass classification tasks where an example can only belong to only one class. The categorical cross entropy used the following formula:
J(w)=-1/N ∑_(i=1)^N▒〖[yilog(y ̂"i" )+(1-yi)log(1-y ̂"i" )]〗
	 w refer to the model parameters, e.g. weights of the neural network
	yi is the desired label 
	y ̂i is the predicted label
The Sparse categorical cross entropy uses the same loss function as the categorical cross entropy but it uses integers to represent the labels which the latter uses one hot encoding to represent the classes (e.g [1,0,0],[0,1,0],[0,0,1] for a 3 class labelling)

#Libraries used
For the model
Since our model is built in python, for the basic array operation, we are using python libraries numpy and pandas, and for basic OS related operation that include manipulating directories to store training data and trained model, we are using python library OS. OpenCV is used to take and manage the pictures in our solution and we are creating and training our model using the Keras and TensorFlow models. To present the training data in a format that our model supports, we are using methods from the sklearn libraries in python. As explained earlier once our solution takes the pictures of the user it saves them and trains the model in another thread which meant we must use the threading library.
(Not sure how we are using tflearn but we should have an idea once its all done)  
For the web app
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
For the web app, we are using the flask library.


#Prediction process
The prediction process start by taking a picture through the available camera and from the picture taken we first isolate the face from the frame using the haar_cascade_classifier. This isolated face is first resized to fit to the desired input shape to our model. The resized face picture is fed to the model in which the model then produces the class number. We are opting to rely on aggregate result of 10 predictions to increase our accuracy. After we have done 10 predictions we select from the predicted results that appeared more than a threshold value of 5.

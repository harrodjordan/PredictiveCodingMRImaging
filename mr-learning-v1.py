er# Jordan Harrod 
# Stanford Summer Research Program - Funded by Amgen Foundation
# Created July 4, 2017

# The purpose of this code is to create a neural network that performs binary classification on MRI images with and without motion artifacts
# Artifacts have been introduced into images using mr-artifacts-v1.py, imported here 

#to do - update to include something that puts together the clean images so they're complex calued and 128x256

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import os, os.path
import tkinter as Tk
from tkinter import filedialog
from tkinter import *

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases'
artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases-wArtifacts'

#LOOP
#prompt user choose a file
#return name
#pick name/session/day information from name
#save name data to an array

#LOOP 
#for every name in the array
#pull all images with that name into a 3D array (one for clean, one for artifacts)
#save that array to a list

#print size of each list


clean_imgs = []
artifact_imgs = []
count = 0
temp = []
listofnames = [""]

path = file_path


for f in os.listdir(file_path):

	filename = str(filedialog.askopenfile())
	patient = filename.split("_") #figure out what the parse function is 

	if patient[:2] not in listofnames:
		listofnames.append(patient[:2])

	exit = input("Is that all of the images?")

	if exit == 'y':
		break 

#for this directory
# for all the images with this name in this directory
#append to a 3d image
# save to a 4d matrix

valid_images = [".jpg"]

for person in listofnames:
	for f in os.listdir(file_path):
		
		ext = os.path.splitext(f)[1]
		count = count + 1
		
		if ext.lower() not in valid_images:
			continue

		name = os.path.join(path,f)

		if os.path.isfile(os.path.join(path,f)) and str(person) in str(f):

			try:
				temp.append(np.asarray(PIL.Image.open(name).convert('L')))

			except FileNotFoundError:
				print("File " + ext + " Not found in Directory")
				continue
	
	clean_imgs = [clean_imgs, temp]
	temp = []


path = artif_path


for person in listofnames:
	for f in os.listdir(file_path):
		
		ext = os.path.splitext(f)[1]
		count = count + 1
		
		if ext.lower() not in valid_images:
			continue

		name = os.path.join(path,f)

		if os.path.isfile(os.path.join(path,f)) and str(person) in str(f):

			try:
				temp.append(np.asarray(PIL.Image.open(name).convert('L')))

			except FileNotFoundError:
				print("File " + ext + " Not found in Directory")
				continue
	
	artifact_imgs = [artifact_imgs, temp]
	temp = []

imgs_train = []

imgs_valid = []

imgs_test = []

count = 1

#the above variables are lists of 3D matricies 
#save first 25 to training list
#next 15 to validation list 
#last 10 to test list 

for (clean, artif) in zip(clean_imgs, artifact_imgs) :

	if 0 < count < 26:
		imgs_train.append(clean)
		imgs_train.append(artif)
		count = count + 1
		continue

	if 26 <= count < 41:
		imgs_valid.append(clean)
		imgs_valid.append(artif)
		count = count + 1
		continue

	if count >= 41:
		imgs_test.append(clean)
		imgs_test.append(artif)
		count = count + 1
		continue

# labels show which 3D matricies are which
label_train = ([0, 1])*25 
label_valid = ([0, 1])*15 
label_test = ([0, 1])*10 

# figure out what these variables should be when dealing with 3D matrix
n_input = 128*256

# this should still be fine - output a [1,0] if clean and a [0,1] if artifact
n_output = 2

x = tf.placeholder(tf.float32, [None, n_input])

#x_tensor = tf.reshape(x, [-1, 128, 256, 99]) 

y = tf.placeholder(tf.float32, [None, 2])

#Weight and bias should be 3D
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))

#update all below dimensions to be 3D as needed
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

y_true = tf.placeholder(tf.float32, [None, n_output])

cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))

correct_prediction = tf.equal(tf.argmax(net_output,1), tf.argmax(net_output,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# deal with this section once everything else is fixed
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 5
n_epochs = 10
batch_xs = []
batch_ys = []

for i in range(n_epochs):
	for batch in range(batch_size):
		
		end = int((batch*5 - 1))
		start = int((end - 5))
		batch_xs = imgs_train[start:end:1][:][:]
		batch_ys = label_train[start:end:1][:][:]
		sess.run(optimizer, feed_dict={net_input: batch_xs , y_true: batch_ys})
	print(sess.run(accuracy, feed_dict={
								x: imgs_valid,
								y: label_valid
					}))







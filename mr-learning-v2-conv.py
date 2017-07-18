# Jordan Harrod 
# Stanford Summer Research Program - Funded by Amgen Foundation
# Created July 11, 2017

# The purpose of this code is to create a neural network that performs binary classification on MRI images with and without motion artifacts
# Artifacts have been introduced into images using mr-artifacts-v1.py, imported here 

#to do 
# create a plot of y: fraction of misses (accuracy) x: probability of having an artifact (y_true)
# randomize the f

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

def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases'
artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases-wArtifacts'


clean_imgs = []
artifact_imgs = []
count = 0
temp = []
listofnames = [""]

path = file_path


for f in os.listdir(file_path):

	patient = split_at(f, "_",3)[0] #figure out what the parse function is 

	if patient not in listofnames:

		listofnames.append(patient)
listofnames = listofnames[1:]


#print(listofnames)

#for this directory
# for all the images with this name in this directory
#append to a 3d image
# save to a 4d matrix
count = -1

valid_images = [".jpg"]

for person in listofnames:
	for f in os.listdir(file_path):
		
		ext = os.path.splitext(f)[1]

		
		if ext.lower() not in valid_images:
			continue

		name = os.path.join(path,f)

		if os.path.isfile(os.path.join(path,f)) and person in f:

			try:
				temp.append(np.asarray(PIL.Image.open(name).convert('L')))

			except FileNotFoundError:
				print("File " + ext + " Not found in Directory")
				continue
	#print(np.asarray(temp).shape)
	clean_imgs.append(temp)
	temp = []

#print(np.array(clean_imgs).shape)

path = artif_path

count = -1

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
	
	artifact_imgs.append(temp)
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

	if 0 < count <= 7:
		imgs_train.append(clean)
		imgs_train.append(artif)
		count = count + 1
		continue
	if  8 <= count <= 10:
		imgs_valid.append(clean)
		imgs_valid.append(artif)
		count = count + 1
		continue

	if count >= 13:
		imgs_test.append(clean)
		imgs_test.append(artif)
		count = count + 1
		continue

# labels need to be fixed
label_train = np.matrix([1,0]*7)
label_train = np.repeat(label_train[:, :, np.newaxis], 99, axis=2)
label_valid = np.matrix([1,0]*3)
label_valid = np.repeat(label_valid[:, :, np.newaxis], 99, axis=2)
label_test = np.matrix([1,0]*3)
label_test = np.repeat(label_test[:, :, np.newaxis], 99, axis=2)


# figure out what these variables should be when dealing with 3D matrix
n_input = 128*256

# this should still be fine - output a [1,0] if clean and a [0,1] if artifact
n_output = 2

x = tf.placeholder(tf.float32, [99, 256, 256])

x_tensor = tf.reshape(x, [-1, 256, 256, 99]) 

y = tf.placeholder(tf.float32, [1, 99]) #check this later 

filter_size = 4
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 99, n_filters_1])

b_conv1 = bias_variable([n_filters_1])

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=x_tensor,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)

n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)

h_conv2_flat = tf.reshape(h_conv2, [-1, 4 * 4 * n_filters_2])

n_fc = 1024
W_fc1 = weight_variable([4 * 4 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([n_fc, 99])
b_fc2 = bias_variable([99])
y_pred = tf.nn.log_softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Create a graph of y: fraction of misses (accuracy) x: probability of having an artifact (y_true)


# deal with this section once everything else is fixed
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 14
n_epochs = 10
batch_xs = []
batch_ys = []

fraction_miss = []
prob_artifact = []

for i in range(n_epochs):
	for batch in range(batch_size):

		batch_xs = np.asarray(imgs_train[batch][:][:][:])
		batch_ys = np.asarray(label_train[batch])
		
		sess.run(optimizer, feed_dict={x: batch_xs , y: batch_ys, keep_prob: 0.5})

		fraction_miss.append(sess.run(accuracy, feed_dict={x: batch_xs , y: batch_ys, keep_prob: 0.5}))
		prob_artifact.append(np.mean(sess.run(y_pred, feed_dict={x: batch_xs , y: batch_ys, keep_prob: 0.5})))


for batch in range(6):
	print(sess.run(accuracy, feed_dict={
								x: imgs_valid[batch],
								y: label_valid[batch],
								keep_prob: 1.0
					}))

	

#prob_artifact = np.mean(prob_artifact[:],axis = 1)
#prob_artifact = np.mean(prob_artifact[:],axis = 1)
print(type(prob_artifact[1]))
plt.plot(np.asarray(prob_artifact), np.asarray(fraction_miss))
plt.show()


	






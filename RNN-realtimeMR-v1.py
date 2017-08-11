# Jordan Harrod 
# Stanford Summer Research Program - Funded by Amgen Foundation
# Created July 11, 2017

# The purpose of this code is to create a neural network that performs binary classification on MRI images with and without motion artifacts
# Artifacts have been introduced into images using mr-artifacts-v1.py, imported here 

#now with annotations for TensorBoard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import os, os.path
import tkinter as Tk
from tkinter import filedialog
from tkinter import *
from tensorflow.python import debug as tf_debug
from rnn_artifacts_v1 import *


import random

import argparse
import sys

FLAGS = None






def import_images():

	# Create artifacts - update paths as needed 
	#path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train'
	#path_out= r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_artifact'
	path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases'
	path_out = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases_RNN'

	# Create artifacts 
	response = input('Do you need to create artifacts? (y/n)')

	if response == 'y':
		create_artifacts(path, path_out)

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])
  
  # Import data
	# Choose whichever directory you plan to use for the training - DCE-50 cases is the smaller data set 
	#file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_clean'
	#artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_artifact'

	file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases-noArtifactsRandom-Jul2517'
	artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases_RNN'

	clean_imgs = []
	artifact_imgs = []
	count = 0
	temp = []
	listofnames = [""]

	path = file_path


	for f in os.listdir(file_path):

		patient = split_at(f, "_",3)[0]

		if patient not in listofnames:

			listofnames.append(patient)
	
	listofnames = listofnames[1:]

	

	valid_images = [".jpg"]

	for person in listofnames:
		
		for f in os.listdir(path):
		
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

		clean_imgs.append(temp)
		temp = []



	path = artif_path


	for person in listofnames:
		for f in os.listdir(path):
		
			ext = os.path.splitext(f)[1]
		
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


	start_train = 0
	stop_train = 10
	stop_valid = 13

	for (clean, artif) in zip(clean_imgs, artifact_imgs) :
		print(count)
		if start_train < count <= stop_train:
			imgs_train.append(clean)
			#imgs_train.append(artif)
			count = count + 1
			continue
		if  (stop_train+1) <= count <= stop_valid:
			imgs_valid.append(clean)
			#imgs_valid.append(artif)
			count = count + 1
			continue


	return imgs_train, imgs_valid 

def train(images, vimages):

	sess = tf.InteractiveSession()

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])

	def variable_summaries(var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)

		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

	sess = tf.InteractiveSession()

	learning_rate = 0.001
	training_iters = 1000
	batch_size = 256
	display_step = 10

# Network Parameters
	n_input_x = 128 # data input (img shape: 28*28)
	n_input_y = 256 
	n_steps = 99 # timesteps
	n_hidden = 128 # hidden layer num of features
	n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [99, 256, 128])
		ex = tf.slice(x,[0,0,0],[1,-1,-1])
		ex = x[:,:,:, np.newaxis]
		tf.summary.image('input', ex,1)

	with tf.name_scope('input_reshape'):
		x_tensor = x[:,:,:, np.newaxis]
		x_tensor = tf.transpose(x_tensor, perm = [3,0,1,2])
		tf.summary.image('input_reshape', x_tensor, 1)

# Define weights
	with tf.name_scope('weights'):
		weights = weight_variable([n_hidden, n_classes, 1, 1])
		variable_summaries(weights)
		#weights = {'out': tf.Variable(tf.rxandom_normal([n_hidden, n_classes]))}

	with tf.name_scope('biases'):
		biases = bias_variable([n_classes])
		variable_summaries(biases)
		#biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


	#def RNN(x, weights, biases):

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, n_steps, n_input)
		# Required shape: 'n_steps' tensors list of shape (batch_size, n_input_x, n_input_y)

		# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input_x, n_input_y)
	with tf.name_scope('unstack'):
		x = tf.unstack(x, 99, 0)
		tf.summary.scalar('unstack', x)

	# Define a lstm cell with tensorflow
	with tf.name_scope('LSTM_cell'):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, 
					forget_bias=1.0, state_is_tuple=False)

	state = tf.zeros([batch_size, lstm_cell.state_size])


	loss = []

	# Get lstm cell output
	with tf.name_scope('static_RNN'):
		output, state = tf.contrib.rnn.static_rnn(lstm_cell, x, initial_state=state)

	last_image = np.zeros([256,128], dtype=np.float32)
	last_image = last_image[:,:,np.newaxis,np.newaxis]

	for image in output:

		image = image[:,:,np.newaxis,np.newaxis]
		
		logits_last = tf.nn.conv2d(input =last_image, filter=weights, strides=[1,2,2,1], padding='SAME') + biases

		logits_curr = tf.nn.conv2d(input =image, filter=weights, strides=[1,2,2,1], padding='SAME') + biases

		if np.array(logits_last).shape != np.array(logits_last).shape:
			with tf.name_scope('pool'):
				ksize = [1, 1]
				strides = [2, 1]
				out_layer = tf.nn.max_pool(logits_curr, ksize=ksize, strides=strides, 
                           padding='SAME')

		with tf.name_scope('cross-entropy'):
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_last, 
				labels=logits_curr))

			loss.append(cost)


		with tf.name_scope('accuracy'):
	
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(logits_last, logits_curr)
	
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

				tf.summary.scalar('accuracy', accuracy)

		last_image = image

		with tf.name_scope('train'):
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Define loss and optimizer
		

			
			

			
		#return loss, accuracy

	#[loss, accuracy] = RNN(x, weights, biases)



	



	


	
# Evaluate model

  # Merge all the summaries and write them out to
  # git (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_test')
	print(FLAGS.log_dir + '/train')

	tf.global_variables_initializer().run()

	def feed_dict(num):
		print(np.asarray(images).shape)
		batch_xs = np.asarray(images[num][:][:])

		c = list(batch_xs)

		random.shuffle(c)

		batch_xs = zip(*c)

		return {x: batch_xs}

	def feed_dict_test(num):

		batch_xs = np.asarray(vimages[num][:][:])

		c = list(batch_xs)

		random.shuffle(c)

		batch_xs = zip(*c)

		return {x: batch_xs}


	batch_size = 10
	test_size = 3
	n_epochs = 800


	# Launch the graph
	for i in range(n_epochs):
		step = 0
		# Keep training until reach max iterations
		while step < batch_size:
			print(step)
			batch_x= feed_dict(step)
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x})
			train_writer.add_summary(summary, i)
			if step % 5 == 0:
				# Calculate batch accuracy
				acc = sess.run(accuracy, feed_dict={x: batch_x})
				# Calculate batch loss
				loss = sess.run(cost, feed_dict={x: batch_x})


				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))
			step += 1
		print("Optimization Finished!")

		# Calculate accuracy for 128 mnist test images
		while step < test_size:
			test_x = feed_dict_test(step)
			test_data = test_x
			print("Testing Accuracy:", \
			sess.run(accuracy, feed_dict={x: test_data}))
			test_writer.add_summary(summary, i)



def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	[images_train, images_valid] = import_images()
	train(images_train, images_valid)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
					  default=False,
					  help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000,
					  help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
					  help='Keep probability for training dropout.')
	parser.add_argument(
	  '--data_dir',
	  type=str,
	  default='/tmp/tensorflow/mnist/input_data',
	  help='Directory for storing input data')
	parser.add_argument(
	  '--log_dir',
	  type=str,
	  default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
	  help='Summaries log directory')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)















	






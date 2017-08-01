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


import random

import argparse
import sys

FLAGS = None






def import_images():

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])
  # Import data
	file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_clean'
	artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_artifact'

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

#the above variables are lists of 3D matricies 
#save first 25 to training list
#next 15 to validation list 
#last 10 to test list 

	for (clean, artif) in zip(clean_imgs, artifact_imgs) :
		#print(count)
		if 0 < count <= 130:
			imgs_train.append(clean)
			imgs_train.append(artif)
			count = count + 1
			continue
		if  131 <= count <= 150:
			imgs_valid.append(clean)
			imgs_valid.append(artif)
			count = count + 1
			continue

		#if count > 10:
			#imgs_test.append(clean)
			#imgs_test.append(artif)
			#count = count + 1
			#continueg
	
# labels need to be fixed
	label_train = np.matrix([1,0]*130)
	label_train = np.repeat(label_train[:, :, np.newaxis], 99, axis=2)
	label_valid = np.matrix([1,0]*20)
	label_valid = np.repeat(label_valid[:, :, np.newaxis], 99, axis=2)
	#label_test = np.matrix([1,0]*3)
	#label_test = np.repeat(label_test[:, :, np.newaxis], 99, axis=2)



	c = list(zip(imgs_train, label_train))

	random.shuffle(c)

	imgs_train, labels_train = zip(*c)

	#c = list(zip(imgs_valid, label_valid))

	#random.shuffle(c)

	#imgs_valid, labels_valid = zip(*c)

	return label_train, imgs_train, label_valid, imgs_valid 

def train(labels, images, vlabels, vimages):

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])

	sess = tf.InteractiveSession()


	learning_rate = 0.001
	training_iters = 100000
	batch_size = 128
	display_step = 10

# Network Parameters
	n_input = 28 # MNIST data input (img shape: 28*28)
	n_steps = 28 # timesteps
	n_hidden = 128 # hidden layer num of features
	n_classes = 10 # MNIST total classes (0-9 digits)

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


# tf Graph input
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [99, 256, 128])
		y = tf.placeholder(tf.float32, [1]) 
		ex = tf.slice(x,[0,0,0],[1,-1,-1])
		ex = x[:,:,:, np.newaxis]
		tf.summary.image('input', ex,1)

	with tf.name_scope('input_reshape'):
		#x_tensor = tf.reshape(x, [-1, 256, 256, 99]) 
		x_tensor = x[:,:,:, np.newaxis]
		x_tensor = tf.transpose(x_tensor, perm = [3,1,2,0])
		tf.summary.image('input_reshape', x_tensor, 1)

# Define weights
	with tf.name_scope('weights'):
		weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}

	with tf.name_scope('biases')
		biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


	def RNN(x, weights, biases):

	    # Prepare data shape to match `rnn` function requirements
	    # Current data input shape: (batch_size, n_steps, n_input)
	    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	    x = tf.unstack(x, n_steps, 1)

	    # Define a lstm cell with tensorflow
	    lstm_cell = tf.nn.rnn_cell(n_hidden, forget_bias=1.0)

	    # Get lstm cell output
	    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

	    # Linear activation, using rnn inner loop last output
	    return tf.matmul(outputs[-1], weights['out']) + biases['out']

	pred = RNN(x, weights, biases)

	# Define loss and optimizer
	with tf.name_scope('cross-entropy')
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model

	with tf.name_scope('accuracy'):
		
		with tf.name_scope('correct_prediction'):
			correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


  # Merge all the summaries and write them out to
  # git (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_test')
	print(FLAGS.log_dir + '/train')

	init =	tf.global_variables_initializer().run()



	def feed_dict(num):

		#print(np.asarray(imgs_train).shape)
		batch_xs = np.asarray(images[num][:][:])
		batch_ys = np.asarray(labels[num])
		#k = FLAGS.dropout

		
		return {x: batch_xs, y: batch_ys}

	def feed_dict_test():

		batch_xs = np.asarray(vimages)
		batch_ys = np.asarray(vlabels)

		return {x: batch_xs, y: batch_ys}


	batch_size = 259
	acc_temp = []
	n_epochs = 800

	# Launch the graph
	with tf.Session() as sess:
	    sess.run(init)
	    step = 1
	    # Keep training until reach max iterations
	    while step * batch_size < training_iters:
	        batch_x, batch_y = feed_dict(step)
	        # Reshape data to get 28 seq of 28 elements
	        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
	        # Run optimization op (backprop)
	        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
	        if step % display_step == 0:
	            # Calculate batch accuracy
	            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
	            # Calculate batch loss
	            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
	            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.5f}".format(acc))
	        step += 1
	    print("Optimization Finished!")

	    # Calculate accuracy for 128 mnist test images
	    test_len = 128
	    test_x, test_y = feed_dict()
	    test_data = test_x.reshape((-1, n_steps, n_input))
	    test_label = test_y
	    print("Testing Accuracy:", \
	        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	[labels_train, images_train, labels_valid, images_valid] = import_images()
	train(labels_train,images_train, labels_valid, images_valid)
	#test(labels_valid,images_valid)


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















	






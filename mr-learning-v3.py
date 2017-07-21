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




import argparse
import sys

FLAGS = None



def train():

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])
  # Import data
	file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases'
	artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/DCE-abdominal-50cases-wArtifactsRandom'


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

		if count > 10:
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

	sess = tf.InteractiveSession()
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
  # Create a multilayer model.

  # Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [99, 256, 256])
		y = tf.placeholder(tf.float32, [1, 99]) 

	with tf.name_scope('input_reshape'):
		x_tensor = tf.reshape(x, [-1, 256, 256, 99]) 
		ex = weight_variable([99, 256, 256, 1])
		tf.summary.image('input', ex, 99)




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
	
	def nn_layer(input_tensor, input_dim, depth, n_filters_1, layer_name, poolstride, act=tf.nn.relu):

	
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
	  	# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, input_dim, depth, n_filters_1])
				variable_summaries(weights)
		
			with tf.name_scope('biases'):
				biases = bias_variable([depth*n_filters_1])
				variable_summaries(biases)
		
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.nn.depthwise_conv2d(input=input_tensor, filter=weights, 
													strides=[1,2,2,1], padding='SAME') + biases
				tf.summary.histogram('pre_activations', preactivate)

			with tf.name_scope('pool'):
				ksize = [1, 1, 1, 1]
				strides = [1, 1, 1, poolstride]
				out_layer = tf.nn.max_pool(preactivate, ksize=ksize, strides=strides, 
                               padding='SAME')
				tf.summary.histogram('pooling', out_layer)
		
			activations = act(out_layer, name='activation')
			tf.summary.histogram('activations', activations)
			return activations

	n_input = 4
	n_filters = 1



	hidden1 = nn_layer(x_tensor, n_input, 99, n_filters, 'layer1', 1)
	hidden2 = nn_layer(hidden1, n_input, 99, n_filters, 'layer2', 1)
	hidden3 = nn_layer(hidden2, n_input, 99, n_filters, 'layer3', 1)
	hidden4 = nn_layer(hidden3, n_input, 99, n_filters, 'layer4', 1)
	hidden5 = nn_layer(hidden4, n_input, 99, n_filters, 'layer5', 1)
	hidden6 = nn_layer(hidden5, n_input, 99, n_filters, 'layer6', 1)
	#hidden6 = nn_layer(hidden5, n_input, 101376, n_filters, 'layer6', 16)
	hidden7 = nn_layer(hidden6, n_input, 99, n_filters, 'layer8', 1 )
	#hidden7 = nn_layer(hidden6, n_input, 1584, n_filters, 'layer7', 16)

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(hidden7, keep_prob)

	

	# Do not apply softmax activation yet, see below.
	y_ = nn_layer(dropped, n_input, 99, 1, 'layer8',1 , act=tf.identity)


	with tf.name_scope('cross_entropy'):
	# The raw formulation of cross-entropy,
	#
	# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
	#                               reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the
	# raw outputs of the nn_layer above, and then average across
	# the batch.
		#print(y_.shape)
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
		
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)

	tf.summary.scalar('cross_entropy', cross_entropy)

	
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
		cross_entropy)


	with tf.name_scope('accuracy'):
		
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # git (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

	def feed_dict(num):

		#print(np.asarray(imgs_train).shape)
		batch_xs = np.asarray(imgs_train[num][:][:][:])
		batch_ys = np.asarray(label_train[num])
		k = FLAGS.dropout
		
		return {x: batch_xs, y: batch_ys, keep_prob: k}

	batch_size = 14
	n_epochs = 10
	#print(np.asarray(imgs_train).shape)
	#print(np.asarray(imgs_valid).shape)
	#print(np.asarray(imgs_test).shape)
	for i in range(n_epochs):
		for batch in range((np.asarray(imgs_train).shape)[0]):
			#print(i)
			#print(batch)
			if i % 2 == 0:  # Record summaries and test-set accuracy
				summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(batch))
				test_writer.add_summary(summary, i)
				print('Accuracy at step %s: %s' % (i, acc))
			else:  # Record train set summaries, and train
				if i % 5 == 99:  # Record execution stats
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					summary, _ = sess.run([merged, train_step],
							  feed_dict=feed_dict(batch),
							  options=run_options,
							  run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
					train_writer.add_summary(summary, i)
					print('Adding run metadata for', i)
				else:  
					summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(batch))
					train_writer.add_summary(summary, i)
	train_writer.close()
	test_writer.close()


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()


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















	






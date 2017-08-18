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
from tensorflow.python import debug as tf_debug
from rnn_artifacts_v1 import *


import random

import argparse
import sys

FLAGS = None




with tf.device('/gpu:1'):

	def import_images():

		# Create artifacts - update paths as needed 
		#path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train'
		#path_out= r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_artifact'
		#path = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed'
		#path_out = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN'

		path = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed'
		path_out = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN'



		if os.path.isdir(path_out) == False:
			
			create_artifacts(path, path_out)

		def split_at(s, c, n):

			words = s.split(c)
			
			return c.join(words[:n]), c.join(words[n:])
	  
	  # Import data
		# Choose whichever directory you plan to use for the training - DCE-50 cases is the smaller data set 
		#file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_clean'
		#artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train_artifact'

		#file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN/clean'
		#artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

		file_path = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN/clean'
		artif_path = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

		assert os.path.isdir(file_path) == True, 'file_path already exists, please choose a different path to avoid overwriting'

		assert os.path.isdir(artif_path) == True, 'file_path already exists, please choose a different path to avoid overwriting'

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

		count = 1

		for person in listofnames:

			if count > 20: 
				break
			
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
			count = count + 1



		path = artif_path

		count = 1


		for person in listofnames:

			if count > 20:
				break

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
			count = count + 1

		imgs_train = []

		imgs_valid = []

		imgs_test = []

		count = 1


		start_train = 0
		stop_train = 16
		stop_valid = 20

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
		n_input_x = 80 # data input (img shape: 28*28)
		n_input_y = 180 
		n_steps = 99 # timesteps
		n_hidden = 128 # hidden layer num of features
		n_classes = 1 # MNIST total classes (0-9 digits)

	# tf Graph input
		with tf.name_scope('input'):
			x = tf.placeholder(tf.float32, [18, 180, 80])
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

		with tf.name_scope('proj_weights'):
			proj_weights = weight_variable([n_hidden, n_classes, 1, 1])
			variable_summaries(weights)

		with tf.name_scope('biases'):
			biases = bias_variable([n_classes])
			variable_summaries(biases)
			#biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

		with tf.name_scope('proj_biases'):
			proj_biases = bias_variable([n_classes])
			variable_summaries(biases)


		#def RNN(x, weights, biases):

			# Prepare data shape to match `rnn` function requirements
			# Current data input shape: (batch_size, n_steps, n_input)
			# Required shape: 'n_steps' tensors list of shape (batch_size, n_input_x, n_input_y)

			# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input_x, n_input_y)

		def layers(x, last_error, layer, name):

			with tf.name_scope(name):

				loss = last_error;

				with tf.name_scope('unstack'):
					x = tf.unstack(x, 18, 0)
					tf.summary.scalar('unstack', x)

				count = 0
				loop = 0

				new_x1 = [] #np.zeros([1, 256,64,1,1])
				new_x2 = [] #np.zeros([1, 256,64,1,1])
				new_x3 = [] #np.zeros([1, 256,64,1,1])
				new_x4 = [] #np.zeros([1, 256,64,1,1])

				for image in x:

					image = image[:,:,np.newaxis,np.newaxis]

					if count < 6:
					
						image  = tf.nn.conv2d(input =image, filter=weights, strides=[2,2,1,1], padding='SAME') + biases

						image = image[np.newaxis, :,:,:,:]

						new_x1.append(image)

						count = count + 1

						continue

					if count < 12:
					
						image = tf.nn.conv2d(input =image, filter=weights, strides=[2,2,1,1], padding='SAME') + biases

						image = image[np.newaxis, :,:,:,:]

						new_x2.append(image)

						count = count + 1

						continue

					if count < 18:

						image = tf.nn.conv2d(input =image, filter=weights, strides=[2,2,1,1], padding='SAME') + biases

						image = image[np.newaxis, :,:,:,:]

						new_x3.append(image)

						count = count + 1

						continue

					if count == 6 or count == 12 or count == 18:

						image = image[np.newaxis, :,:,:,:]

						new_x4.append(image)

						count = count + 1

						continue 

				new_x1 = new_x1[1:][:][:][:][:]
				new_x2 = new_x2[1:][:][:][:][:]
				new_x3 = new_x3[1:][:][:][:][:]
				new_x4 = new_x4[1:][:][:][:][:]


				for x in tf.unstack(tf.stack([new_x1, new_x2, new_x3])):

					count = 0

					x = tf.unstack(x,5,0)

					def nn_layer(input_tensor, input_dim, n_filters_1, layer_name):

		
							# Adding a name scope ensures logical grouping of the layers in the graph.
							with tf.name_scope(layer_name):
							# This Variable will hold the state of the weights for the layer
								with tf.name_scope('weights'):
									weights = weight_variable([input_dim, input_dim, n_filters_1])
									variable_summaries(weights)
							
								with tf.name_scope('biases'):
									biases = bias_variable([depth*n_filters_1])
									variable_summaries(biases)
							
								with tf.name_scope('Wx_plus_b'):
									activations = tf.nn.relu(tf.nn.conv2d(input=input_tensor, filter=weights, strides=[1,2,2,1], padding='SAME') + biases)

									tf.summary.image('input_times_gradient', input_tensor*(tf.gradient(weights)))

									feature_map = tf.slice(preactivate,[0,0,0,0],[-1,-1,-1,1])

									tf.summary.image(layer_name, feature_map,1)

								with tf.name_scope('pool'):

									ksize = [1, 1, 1, 1]

									strides = [1,1, 1, 1]

									out_layer = tf.nn.max_pool(activations, ksize=ksize, strides=strides, padding='SAME')
									tf.summary.histogram('pooling', out_layer)


								return out_layer

					n_input = 4
					n_filters = 1
					latent = []
					valid = []
					real = False 

					if layer == 1:
						image = new_x4[count]

					else:

						image = last_error

					for output in x:

						
						if real == True : 
						#layer 1:

							feat_proj = nn_layer(output, n_input, n_filters, 'layer1')

							#layer 2:

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer2')

							#layer 2:

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer3')

							#layer 2:

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer4')

							latent.append(feat_proj)

						else: 

							feat_real = nn_layer(image, n_input, n_filters, 'layer1')

							feat_proj = nn_layer(output, n_input, n_filters, 'layer1')


							#layer 2:

							feat_real = nn_layer(feat_real, n_input, n_filters, 'layer2')

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer2')

							#layer 2:

							feat_real = nn_layer(feat_real, n_input, n_filters, 'layer3')

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer3')

							#layer 2:


							feat_real = nn_layer(feat_real, n_input, n_filters, 'layer4')

							feat_proj = nn_layer(feat_proj, n_input, n_filters, 'layer4')

							valid = feat_real
							latent = feat_proj

							real = True



					# Define a lstm cell with tensorflow
					with tf.name_scope('LSTM_cell'):
						lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, 
									forget_bias=0.5, state_is_tuple=False)

					state = tf.zeros([batch_size, lstm_cell.state_size])


					# Get lstm cell output
					with tf.name_scope('static_RNN'):

						output, state = tf.contrib.rnn.static_rnn(lstm_cell, latent, initial_state=state)


					shape = output.get_shape().as_list()

					reshape_proj = tf.reshape(latent, [shape[0], shape[1]])

					reshape_real = tf.reshape(valid, [shape[0], shape[1]])

					hidden_proj = tf.nn.relu(tf.matmul(reshape_proj, proj_weights) + proj_biases)

					hidden_real = tf.nn.relu(tf.matmul(reshape_proj, proj_weights) + proj_biases)

					

					with tf.name_scope('cross-entropy'):
						cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_proj, 
								labels=hidden_real))

						loss.append(cost)


					with tf.name_scope('accuracy'):
				
						with tf.name_scope('correct_prediction'):

							correct_prediction = tf.equal(hidden_real, hidden_proj)
				
						with tf.name_scope('accuracy'):

							accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

							tf.summary.scalar('accuracy', accuracy)

					with tf.name_scope('deconv'):

						upsampled = tf.nn.conv2d_transpose(hidden_proj, [1, 1, 1, 1], [1, 256, 128, 1], 2, padding='SAME', name=None) # and then update this to properly upsample 

					count = count + 1
			
			return loss, upsampled, feat_proj

		[loss1, upsampled1, features1] = layers(x, [], 1, 'layer1')
		[loss2, upsampled2, features2] = layers(upsampled1, loss1, 2, 'layer2')
		[loss3, upsampled3, features3] = layers(upsampled2, loss2, 3, 'layer3')
		[loss4, upsampled4, features4] = layers(upsampled3, loss3, 4, 'layer4')

		with tf.name_scope('latent features'):

			features = tf.stack[features1, features2, features3, features4]
			tf.summary.scalar('latent featires', features)


		with tf.name_scope('train'):
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		
	# Evaluate model

	  # Merge all the summaries and write them out to
	  # git (by default)
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/rnn_test')
		print(FLAGS.log_dir + '/train')

		tf.global_variables_initializer().run()

		def feed_dict(num):

			batch_xs = np.asarray(images[num][:][:])

			c = list(batch_xs)

			random.shuffle(c)

			batch_xs = zip(*c)

			return batch_xs

		def feed_dict_test(num):

			batch_xs = np.asarray(vimages[num][:][:])

			c = list(batch_xs)

			random.shuffle(c)

			batch_xs = zip(*c)
			
			return batch_xs   


		batch_size = 10
		test_size = 3
		n_epochs = 800


		# Launch the graph
		for i in range(n_epochs):
			# Keep training until reach max iterations
			for batch in range(batch_size):
				batch_x= feed_dict(batch)
				# Run optimization op (backprop)
				sess.run(optimizer, feed_dict={x: batch_x})
				train_writer.add_summary(summary, i)
				if i % 5 == 0:
					# Calculate batch accuracy
					acc = sess.run(accuracy, feed_dict={x: batch_x})
					# Calculate batch loss
					loss = sess.run(cost, feed_dict={x: batch_x})


					print("Iter " + str(batch*batch_size) + ", Minibatch Loss= " + \
						  "{:.6f}".format(loss) + ", Training Accuracy= " + \
						  "{:.5f}".format(acc))
			print("Optimization Finished!")

			# Calculate accuracy for 128 mnist test images
			while step < test_size:
				test_x = feed_dict_test(batch)
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
		  default='/mnt/raid5/jordan/tmp/input_data',
		  help='Directory for storing input data')
		parser.add_argument(
		  '--log_dir',
		  type=str,
		  default='/mnt/raid5/jordan/tmp/summaries',
		  help='Summaries log directory')

		FLAGS, unparsed = parser.parse_known_args()
		tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)















		






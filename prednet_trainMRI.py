
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import cPickle
from data_utils import SequenceGenerator

import prednet_mod as PredNet
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import os, os.path
import tkinter as Tk
from tkinter import filedialog
from tkinter import *
from tensorflow.python import debug as tf_debug


from keras import backend as K
from keras import activations
from keras.layers import Recurrent
from keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from keras.engine import InputSpec
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

import argparse
import sys

FLAGS = None
np.random.seed(123)


def import_images():

	def split_at(s, c, n):
		words = s.split(c)
		return c.join(words[:n]), c.join(words[n:])
  # Import data
	file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-40cases-timeresolved-processed'


	clean_imgs = []
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


	imgs_train = []

	imgs_valid = []

	imgs_test = []

	count = 1


	for image in clean_imgs :
		#print(count)
		if 0 < count <= 130:
			imgs_train.append(image)
			count = count + 1
			continue
		if  131 <= count <= 150:
			imgs_valid.append(image)
			count = count + 1
			continue


	return imgs_train, imgs_valid 

def train(train_images, validate_images):

	WEIGHTS_DIR = os.getcwd()

	save_model = True  # if weights will be saved
	weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
	json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

	# Data files
	train_file = train_images #os.path.join(DATA_DIR, 'X_train.hkl')
	train_sources = train_images #os.path.join(DATA_DIR, 'sources_train.hkl')
	val_file = validate_images#os.path.join(DATA_DIR, 'X_val.hkl')
	val_sources = validate_images#os.path.join(DATA_DIR, 'sources_val.hkl')

	# Training parameters
	nb_epoch = 150
	batch_size = 1
	samples_per_epoch = 192
	N_seq_val = 30  # number of sequences to use for validation

	# Model parameters
	nt = 18
	n_channels, im_height, im_width = (1, 180, 80)
	input_shape = (n_channels, im_height, im_width) if K.image_dim_ordering() == 'th' else (im_height, im_width, n_channels)
	stack_sizes = (n_channels, 48, 96, 192)
	R_stack_sizes = stack_sizes
	A_filt_sizes = (3, 3, 3)
	Ahat_filt_sizes = (3, 3, 3, 3)
	R_filt_sizes = (3, 3, 3, 3)
	layer_loss_weights = np.array([1., 0., 0., 0.])
	layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
	time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
	time_loss_weights[0] = 0


	prednet = PredNet.PredNet(stack_sizes, R_stack_sizes,
	                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
	                  output_mode='error', return_sequences=True)

	inputs = Input(shape=(nt,) + input_shape)
	errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
	errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
	errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
	final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
	model = Model(input=inputs, output=final_errors)
	model.compile(loss='mean_absolute_error', optimizer='adam')

	train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
	val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

	lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
	callbacks = [LearningRateScheduler(lr_schedule)]
	if save_model:
	    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
	    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

	history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks,
	                    validation_data=val_generator, nb_val_samples=N_seq_val)

	if save_model:
	    json_string = model.to_json()
	    with open(json_file, "w") as f:
	        f.write(json_string)

def main(_):
	[images_train, images_valid] = import_images()
	train(images_train, images_valid)
	os.system('say "your program has finished"')






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


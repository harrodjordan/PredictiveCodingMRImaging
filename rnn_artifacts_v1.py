# Jordan Harrod 
# Stanford Summer Research Program - Funded by Amgen Foundation
# Created July 3, 2017
# Last Updated July 7, 2017

# The purpose of this code is to create motion artifacts in MRI images 



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as plot
import os, os.path
from PIL import Image
from scipy import ndimage
import random as rand 


with tf.device('/gpu:1'):
	# Creates artifact using images in path, export images to path_out 
	def create_artifacts(path, path_out):

		assert os.path.isdir(path_out) == False, 'file_path already exists, please choose a different path to avoid overwriting'
		
		imgs = []

		path_in = path

		# Only reading in JPG images for now
		valid_images = [".jpg"]
		count = 0

		# Creaet a list of the images in the folder
		for f in os.listdir(path):

			ext = os.path.splitext(f)[1]
			count = count + 1

			if ext.lower() not in valid_images:

				continue

			name = path + "/" + f

			try:
				imgs.append(np.asarray(Image.open(name).convert('L')))

			except FileNotFoundError:
				print("File " + ext + " Not found in Directory")
				continue

		
		# temporary variables for processing 
		temp_real = []
		temp_complex = []

		complex_images = []
		real_images = []

		#pull real-valued and complex-valued image 
		for image in imgs:

			#print(image.shape)

			temp_complex = image[:, 80:160]

			temp_real = image[:,0:80]

			#print(temp_complex.shape)
			#print(temp_real.shape)
			
			complex_images.append(temp_complex)
			real_images.append(temp_real)


		# perform an fft on the phase data (complex data) to change the phase encoding direction 

		new_complex = []
		temp_fourier = []
		count = 0


		for image in complex_images:
			#for RNN - only change the first image, we'll shuffle with labels later
			if count == 0:
				image.setflags(write=1)

				temp_fourier = np.fft.fft2(image)

				b = temp_fourier[::2]
				b[:] = 0

				temp_fourier = np.fft.ifft2(temp_fourier)

				new_complex.append(temp_fourier)
			else:
				new_complex.append(image)

		#put the images back tohether, either on two sides or as one magnitude image 
		sum_images_clean = []

		for (real, comp) in zip(real_images, complex_images):
			temp_image = np.empty((180, 80))
			real = np.asarray(real)
			comp = np.asarray(comp)*8*np.pi

			temp_image = np.absolute(real+comp)
			sum_images_clean.append(temp_image)

		sum_images_clean = np.asarray(sum_images_clean)

		sum_images_artif = []

		for (real, comp) in zip(real_images, new_complex):
			temp_image = np.empty((180, 80))
			real = np.asarray(real)
			comp = np.asarray(comp)*8*np.pi

			temp_image = np.absolute(real+comp)
			sum_images_artif.append(temp_image)

		sum_images_artif = np.asarray(sum_images_artif)
		#print(sum_images.shape)


		#save to a different directory

		file_path = path_out + "/clean"

		if not os.path.isdir(file_path):
			os.makedirs(file_path)

		for (f, image) in zip(os.listdir(path), sum_images_clean):

			save_path = file_path + "/" + f + "clean.jpg"

			image = np.asarray(np.absolute(image), dtype=float)

			plot.image.imsave(save_path, image)


		file_path = path_out + "/artifacts"

		if not os.path.isdir(file_path):
			os.makedirs(file_path)

		for (f, image) in zip(os.listdir(path), sum_images_artif):

			save_path = file_path + "/" + f + "artifacts.jpg"

			image = np.asarray(np.absolute(image), dtype=float)

			plot.image.imsave(save_path, image)



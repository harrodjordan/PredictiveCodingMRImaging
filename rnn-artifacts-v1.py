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


# Creates artifact using images in path, export images to path_out 
def create_artifacts(path, path_out)
    
    imgs = []

    path_in = get(path)

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
        
        temp_complex = image[:, 128:256]

        temp_real = image[:,0:128]
        
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
    sum_images = []

    for (real, comp) in zip(real_images, new_complex):
        temp_image = np.empty((256, 128))
        real = np.asarray(real)
        comp = np.asarray(comp)
            
        #temp_image[:,0:128] = real
        #temp_image[:,128:256] = abs(comp)
        temp_image = np.absolute(real+comp)
        sum_images.append(temp_image)

    sum_images = np.asarray(sum_images)
    #print(sum_images.shape)


    #save to a different directory

    file_path = get(path_out)

    assert os.path.isdir(file_path) == False, 'file_path already exists, please choose a different path to avoid overwriting'

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    for (f, image) in zip(os.listdir(path), sum_images):

        save_path = file_path + "/" + f + "train_artifact.jpg"

        image = np.asarray(np.absolute(image), dtype=float)

        plot.image.imsave(save_path, image)



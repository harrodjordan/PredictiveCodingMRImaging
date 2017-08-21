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


#import images
#def create_artifacts(image_path):

imgs = []

path = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017/Abdominal-DCE-40cases-timeresolved-processed_RNN'
#path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-150cases-REU/train'
#path = image_path

valid_images = [".jpg"]
count = 0

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

#pull real-valued and complex-valued image 

temp_real = []
temp_complex = []
sum_images = []
complex_images = []
real_images = []
count = 0

for image in imgs:
    
    temp_complex = image[:, 80:160]

    temp_real = image[:,0:80]
    
    complex_images.append(temp_complex) #change this to include the entire right half instead of just one pixel
    real_images.append(temp_real)
    #count = count + 1

#reconstruct while introducing random motion artifacts (use 3310 code)


# perform an fft on the phase data (complex data) to change the phase encoding direction 

new_complex = []
temp_fourier = []
count = 0
rand_num = rand.random()*10

for image in complex_images:
    image.setflags(write=1)

    temp_fourier = np.fft.fft2(image)

    b = temp_fourier[::2]
    b[:] = 0

    temp_fourier = np.fft.ifft2(temp_fourier)

    new_complex.append(temp_fourier)

    rand_num = rand.random()*10

	
# alternatively, create wrap around by reducing the sampling frequency in the phase-encoding direction
temp = np.asarray(new_complex)
complex_num = 0
#print(temp.shape)

for (real, comp) in zip(real_images, new_complex):
    temp_image = np.empty((256, 128))
    real = np.asarray(real)
    comp = np.asarray(comp)*(1/8*np.pi)

    #for (x,y) in zip(range(256),range(128)):
        
    #temp_image = np.absolute(real, comp)
        #complex_num = real[x,y]+(comp[x,y]*1j)
        #temp_image[x,y] = np.absolute(complex_num)
        
    #temp_image[:,0:128] = real
    #temp_image[:,128:256] = abs(comp)
    temp_image = np.absolute(real+comp)
    sum_images.append(temp_image)

sum_images = np.asarray(sum_images)
#print(sum_images.shape)


#save to a different directory

file_path = r'/mnt/raid5/jordan/Abdominal-DCE-40cases-timeresolved-processed_RNN/train_artifact"

if not os.path.isdir(file_path):
    os.makedirs(file_path)

for (f, image) in zip(os.listdir(path), sum_images):

    save_path = file_path + "/" + f + "train_artifact.jpg"

    image = np.asarray(np.absolute(image), dtype=float)

    plot.image.imsave(save_path, image)



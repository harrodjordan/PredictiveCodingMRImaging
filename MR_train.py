'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
#import matplotlib.pyplot as plt

import prednet 
from prednet import PredNet
import PIL.Image
from data_utils import SequenceGenerator
#WEIGHTS_DIR = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/'
#DATA_DIR =  r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/'

WEIGHTS_DIR = DATA_DIR = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017'

desired_im_sz = (180, 80)
categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]


def split_at(s, c, n):

	words = s.split(c)
			
	return c.join(words[:n]), c.join(words[n:])

# Create image datasets.
# Processes images and saves them in train, val, test splits.

splits = {s: [] for s in ['train', 'test']}
splits['test'] = test_recordings
not_train = splits['test']

#file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-40cases-timeresolved-processed_RNN/clean'
#artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

file_path = r'/mnt/raid5/jordan/Abdominal-DCE-40cases-timeresolved-processed_RNN_scaled/clean'

artif_path = r'/mnt/raid5/jordan/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

assert os.path.isdir(file_path) == True, 'file_path already exists, please choose a different path to avoid overwriting'



clean_imgs = []
artifact_imgs = []
count = 0
temp = []
listofnames = [""]

path = file_path


for f in os.listdir(file_path):

	patient = split_at(f, "_",4)[0]

	if patient not in listofnames:

		listofnames.append(patient)

listofnames = listofnames[2:]



valid_images = [".jpg"]

im_list = []
source_list = []

count = 0
X = []

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
				a = np.asarray(PIL.Image.open(name).convert('L'))

				result = np.zeros([256,128])

				result[:a.shape[0],:a.shape[1]] = a
				result = np.rot90(result)

#				print(a.shape)
#
#				plt.imshow(result)
#				plt.show()

				temp.append(np.asarray(result))

			except FileNotFoundError:
				print("File " + ext + " Not found in Directory")
				continue
	

	temp = np.array(temp)

	#temp = np.reshape(temp, [temp.shape[0],temp.shape[2],temp.shape[1]])



	im_loc = os.path.join(DATA_DIR, person)
	im_list.append([im_loc])
	source_list.append([str(person)] * 18)
	X.append(temp)


	
	temp = []
	count = count + 1

X = np.array(X)


X = X[1:]
im_list = im_list[1:]
source_list = source_list[1:]

print('Creating train data: ' + str(len(im_list)) + ' images')




	





save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# Data files
train_file = np.asarray(X[:15])
train_sources = np.asarray(listofnames[:15])
val_file = np.asarray(X[15:])
val_sources = np.asarray(listofnames[15:])


# Training parameters
nb_epoch = 500
batch_size = 1
samples_per_epoch = 15
N_seq_val = 5  # number of sequences to use for validation

# Model parameters

n_channels, im_height, im_width = (1, 128, 256)
input_shape = (im_height, im_width, n_channels)
#input_shape = (n_channels, im_height, im_width)  if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 18 # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode='all', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(input=inputs, output=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(data_file=train_file, source_file=train_sources, nt=nt, batch_size=batch_size, shuffle=True, sequence_start_mode='all')
val_generator = SequenceGenerator(data_file=val_file, source_file=val_sources, nt=nt, batch_size=batch_size, N_seq=N_seq_val, sequence_start_mode='all')

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks, validation_data=val_generator, nb_val_samples=N_seq_val)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

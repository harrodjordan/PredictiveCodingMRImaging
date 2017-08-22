'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

import prednet
from prednet import PredNet
import PIL.Image
from data_utils import SequenceGenerator
#from kitti_settings import *

WEIGHTS_DIR = DATA_DIR  = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017'

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

file_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-40cases-timeresolved-processed_RNN/clean'
artif_path = r'/Users/jordanharrod/Dropbox/Jordan-project/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

#file_path = r'/mnt/raid5/jordan/Abdominal-DCE-40cases-timeresolved-processed_RNN/clean'

#artif_path = r'/mnt/raid5/jordan/Abdominal-DCE-40cases-timeresolved-processed_RNN/artifacts'

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

n_plot = 40
batch_size = 1
nt = 18

#WEIGHTS_DIR = DATA_DIR = r'/mnt/raid5/jordan/Jordan-AmgenSSRP2017'

WEIGHTS_DIR = RESULTS_SAVE_DIR = DATA_DIR = r'/Users/jordanharrod/Dropbox/Jordan-project/Jordan-AmgenSSRP2017/'

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = np.asarray(X[:15])
test_sources = np.asarray(listofnames[:15])

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique')
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
#if data_format == 'channels_first':
X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
print(X_test.shape)
print(X_hat.shape)
X_test = np.squeeze(X_test)
X_hat = np.squeeze(X_hat)

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()


# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]

for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(np.array(PIL.Image.fromarray(X_test[i,t]).convert("L")), interpolation='none', cmap="gray")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(np.array(PIL.Image.fromarray(X_hat[i,t]).convert("L")), interpolation='none', cmap="gray")
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()

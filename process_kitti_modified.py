'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
#from bs4 import BeautifulSoup
import urllib
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
import PIL.Image
#from kitti_settings import *


pyth
		#x_encode = [n.encode("ascii", "ignore") for n in X]

		hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl').encode("ascii", "ignore"))
		hkl.dump(asciiList, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
	target_ds = float(desired_sz[0])/im.shape[0]
	im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
	d = int((im.shape[1] - desired_sz[1]) / 2)
	im = im[:, d:d+desired_sz[1]]
	return im


if __name__ == '__main__':

	process_data()

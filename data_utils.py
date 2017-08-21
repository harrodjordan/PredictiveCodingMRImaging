import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 dim_ordering=K.image_dim_ordering()):
        self.X = np.array(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = np.array(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        print(self.X.shape)
        
        self.im_shape = self.X[0].shape
        

        # if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
        #     self.possible_starts = np.array([i for i in range(self.X.shape[1] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        # elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
        #     curr_location = 0
        #     possible_starts = []
        #     while curr_location < data_file.shape[1] - self.nt + 1:
        #         if self.sources[0][curr_location] == 1:
        #             possible_starts.append(curr_location)
        #             curr_location += self.nt
        #         else:
        #             curr_location += 1
        #      #possible_starts #this is zero currently 
            
        
        
        self.possible_starts = np.ones(self.im_shape[0])


        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            current_batch_size = self.X.shape[0]

        #batch_x = np.zeros((self.im_shape[0], 1, self.im_shape[1], self.im_shape[2]), np.float32)

        batch_x = np.expand_dims(self.X[current_index], axis=1)

        batch_x = batch_x[None, :, :, :, :]

        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(1, np.float32)
            batch_y = batch_y[None,:]
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x

        return batch_x, batch_y

    def preprocess(self, X):
        return self.X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx])
        return X_all

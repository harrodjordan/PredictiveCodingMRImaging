import numpy as np
import tensorflow as tf 

from keras import backend as K
from keras import activations
from keras.layers import Recurrent




class PredNet(Recurrent):
    '''PredNet architecture - Lotter 2016.
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, but should not be confused with returning all of the layers of the model
            For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                For instance, to return the features of the LSTM "representational" units in the lowest layer, output_mode should be specificied as 'R0'.
                The possible unit types are 'R', 'Ahat', 'A', and 'E' corresponding to the 'representation', 'prediction', 'target', and 'error' units respectively.
        extrap_start_time: time step for which model will start extrapolating.
            Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

    # References
        - [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        - [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
        - [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
    '''
    def upsample_2d(incoming, kernel_size, name="UpSample2D"):
    #UpSample 2D.
    #Input:
       # 4-D Tensor [batch, height, width, in_channels].
    #Output:
     #   4-D Tensor [batch, pooled height, pooled width, in_channels].
    # Arguments:
    #     incoming: `Tensor`. Incoming 4-D Layer to upsample.
    #     kernel_size: `int` or `list of int`. Upsampling kernel size.
    #     name: A name for this layer (optional). Default: 'UpSample2D'.
    # Attributes:
    #     scope: `Scope`. This layer scope
        def get_incoming_shape(incoming):
        #Returns the incoming data shape 
            if isinstance(incoming, tf.Tensor):
                return incoming.get_shape().as_list()
            elif type(incoming) in [np.array, np.ndarray, list, tuple]:
                return np.shape(incoming)
            else:
                raise Exception("Invalid incoming layer.")


        def autoformat_kernel_2d(strides):
            if isinstance(strides, int):
                return [1, strides, strides, 1]
            elif isinstance(strides, (tuple, list, tf.TensorShape)):
                if len(strides) == 2:
                    return [1, strides[0], strides[1], 1]
                elif len(strides) == 4:
                    return [strides[0], strides[1], strides[2], strides[3]]
                else:
                    raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 2 or 4 is supported.")
            else:
                raise Exception("strides format error: " + str(type(strides)))

        input_shape = utils.get_incoming_shape(incoming)
        assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
        kernel = utils.autoformat_kernel_2d(kernel_size)

        with tf.name_scope(name) as scope:
            inference = tf.image.resize_nearest_neighbor(
                incoming, size=input_shape[1:3] * tf.constant(kernel[1:3]))
            inference.set_shape((None, input_shape[1] * kernel[1],
                            input_shape[2] * kernel[2], None))

    # Add attributes to Tensor to easy access weights
        inference.scope = scope

    # Track output tensor.
        tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

        return inference



    def __init__(self, stack_sizes, R_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time = None,
                 **kwargs):

        def InputSpec(self, dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None):
        
            self.dtype = dtype
            self.shape = shape

            if shape is not None:
                self.ndim = len(shape)
            else:
                self.ndim = ndim
            
            self.max_ndim = max_ndim
            self.min_ndim = min_ndim
            self.axes = axes or {}
        
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers - 1), 'len(A_filt_sizes) must equal len(stack_sizes) - 1'
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes
        self.units = 1 #might have to update this at a later time 

        self.pixel_max = pixel_max
        self.error_activation = error_activation
        self.A_activation = A_activation
        self.LSTM_activation = LSTM_activation
        self.LSTM_inner_activation = LSTM_inner_activation

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        self.channel_axis = -1
        self.row_axis = -3
        self.column_axis = -2

        super(PredNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(self.input_spec, ndim=5)]

    def get_output_shape_for(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers,)
        else:
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2**self.output_layer_num
            out_nb_col = input_shape[self.column_axis] / 2**self.output_layer_num
            if self.dim_ordering == 'th':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def get_initial_states(self, x):
        def InputSpec(self, dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None):
        
            self.dtype = dtype
            self.shape = shape

            if shape is not None:
                self.ndim = len(shape)
            else:
                self.ndim = ndim
            
            self.max_ndim = max_ndim
            self.min_ndim = min_ndim
            self.axes = axes or {}
        
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = np.zeros_like(x)  # (samples, timesteps) + image_shape
        non_channel_axis = -2
        for _ in range(2):
            base_initial_state = np.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = np.sum(base_initial_state, axis=1)  # (samples, nb_channels)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}
        if self.extrap_start_time is not None:
           states_to_pass.append('ahat')  # pass prediction in states so can use as actual for t+1 when extrapolating
           nlayers_to_pass['ahat'] = 1
        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                output_size = stack_size * nb_row * nb_col  # flattened size

                reducer = np.zeros((input_shape[self.channel_axis], output_size)) # (nb_channels, output_size)
                initial_state = np.dot(base_initial_state, reducer) # (samples, output_size)
                
                output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = tf.reshape(initial_state, output_shp)
                initial_states += [initial_state]


        if self.extrap_start_time is not None:
            initial_states += [tf.variable(0, int)]  # the last state will correspond to the current timestep
        return initial_states

    def build(self, input_shape):
        def InputSpec(self, dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None):
        
            self.dtype = dtype
            self.shape = shape

            if shape is not None:
                self.ndim = len(shape)
            else:
                self.ndim = ndim
            
            self.max_ndim = max_ndim
            self.min_ndim = min_ndim
            self.axes = axes or {}

        self.input_spec = [InputSpec(self.input_spec, shape=input_shape)]
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}

        for l in range(self.nb_layers):
            for c in ['i', 'f', 'c', 'o']:
                if c == 'c':
                    self.conv_layers[c].append(tf.sigmoid(tf.layers.conv2d(self.R_stack_sizes[l], self.R_filt_sizes[l], self.R_filt_sizes[l], [1,1], 'SAME')))
                else:
                
                    self.conv_layers[c].append(tf.tanh(tf.layers.conv2d(self.R_stack_sizes[l], self.R_filt_sizes[l], self.R_filt_sizes[l], [1,1], 'SAME')))

            self.conv_layers['ahat'].append(tf.nn.relu(tf.layers.conv2d(self.stack_sizes[l], self.Ahat_filt_sizes[l], self.Ahat_filt_sizes[l], [1,1], 'SAME')))

            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(tf.nn.relu(tf.layers.conv2d(self.stack_sizes[l+1], self.A_filt_sizes[l], self.A_filt_sizes[l], [1,1], 'SAME')))

        self.upsample = upsample_2D(imcoming, newsize, name)
        self.pool = tf.nn.max_pool()

        self.trainable_weights = []
        nb_row, nb_col = (input_shape[-2], input_shape[-1]) if self.dim_ordering == 'th' else (input_shape[-3], input_shape[-2])
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c == 'a':
                    nb_channels = 2 * self.R_stack_sizes[l]
                else:
                    nb_channels = self.stack_sizes[l] * 2 + self.R_stack_sizes[l]
                    if l < self.nb_layers - 1:
                        nb_channels += self.R_stack_sizes[l+1]
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)
                if self.dim_ordering == 'tf': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

        self.initial_weights = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.states = [None] * self.nb_layers*3

        if self.extrap_start_time is not None:
            self.t_extrap = tf.variable(self.extrap_start_time, int)

    def step(self, a, states):
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]

        if self.extrap_start_time is not None:
            t = states[-1]
            a = tf.switch(t >= self.t_extrap, states[-2], a)  # if past self.extrap_start_time, the previous prediction will be treated as the actual

        c = []
        r = []
        e = []

        for l in reversed(range(self.nb_layers)):
            inputs = [r_tm1[l], e_tm1[l]]
            if l < self.nb_layers - 1:
                inputs.append(r_up)

            inputs = tf.concatenate(inputs, axis=self.channel_axis)
            i = self.conv_layers['i'][l].call(inputs)
            f = self.conv_layers['f'][l].call(inputs)
            o = self.conv_layers['o'][l].call(inputs)
            _c = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
            _r = o * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            if l > 0:
                r_up = self.upsample.call(_r)

        for l in range(self.nb_layers):
            ahat = self.conv_layers['ahat'][l].call(r[l])
            if l == 0:
                ahat = tf.minimum(ahat, self.pixel_max)
                frame_prediction = ahat

            # compute errors
            e_up = self.error_activation(ahat - a)
            e_down = self.error_activation(a - ahat)

            e.append(K.concatenate((e_up, e_down), axis=self.channel_axis))

            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]

            if l < self.nb_layers - 1:
                a = self.conv_layers['a'][l].call(e[l])
                a = self.pool.call(a)  # target for next layer

        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e
        if self.extrap_start_time is not None:
            states += [frame_prediction, t + 1]
        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'A_activation': self.A_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        base_config = super(PredNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
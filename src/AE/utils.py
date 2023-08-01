import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from keras.utils import conv_utils


def preprocessing(factor, sf, mask):
    """ Perfom all preprocessing steps.
        Args:
        factor: int
        sf: np.ndarray
        mask: np.ndarray
        Returns: np.ndarray, np.ndarray
        """

    # Downsampling
    downsampled_sf = downsampling(factor, sf)

    # Masking
    masked_sf = apply_mask(downsampled_sf, mask)

    # Scaling masked sound field
    scaled_sf = scale(masked_sf)

    # Upsampling scaled sound field and mask
    irregular_sf, mask = upsampling(factor, scaled_sf, mask)

    return irregular_sf, mask


def downsampling(dw_factor, input_sfs):
    """ Downsamples sound fields given a downsampling factor.
        Args:
        dw_factor: int
        input_sfs: np.ndarray
        Returns: np.ndarray
        """
    return input_sfs[:, 0:input_sfs.shape[1]:dw_factor, 0:input_sfs.shape[2]:dw_factor, :]


def apply_mask(input_sfs, masks):
    """ Apply masks to sound fields.
        Args:
        input_sfs: np.ndarray
        masks: np.ndarray
        Returns: np.ndarray
        """

    masked_sfs = []
    for sf, mk in zip(input_sfs, masks):
        aux_sf = copy.deepcopy(sf)
        aux_sf[mk==0] = 0
        for i in range(sf.shape[2]):
            aux_max = aux_sf[:, :, i].max()
            sf[:, :, i][mk[:, :, i]==0] = aux_max
        masked_sfs.append(sf)
    return np.asarray(masked_sfs)

def scale(input_sfs):
    """ Scale data in range 0-1.
        Args:
        input_sfs: np.ndarray
        Returns: np.ndarray
        """

    scaled_sf = []
    for sf in input_sfs:
        for i in range(sf.shape[2]):
            aux_max = sf[:, :, i].max()
            aux_min = sf[:, :, i].min()
            if aux_max == aux_min:
                sf[:, :, i] = 1
            else:
                sf[:, :, i] = (sf[:, :, i]-aux_min)/(aux_max-aux_min)
        scaled_sf.append(sf)
    return np.asarray(scaled_sf)

def upsampling(up_factor, input_sfs, masks):
    """ Upsamples sound fields and masks given a upsampling factor.
        Args:
        up_factor: int
        input_sfs: np.ndarray
        masks: np.ndarray
        Returns: np.ndarray, np.ndarray
        """

    batch_sf_up = []
    batch_mask_up = []

    for sf, mask in zip(input_sfs, masks): #for each sample in the batch size
        sf_up = []
        mask_up = []
        sf = np.swapaxes(sf, 2, 0)
        mask = np.swapaxes(mask, 2, 0)
        for sf_slice in sf:
            positions = np.repeat(range(1, sf_slice.shape[1]), up_factor-1) #positions in sf slice to put 1
            sf_slice_up = np.insert(sf_slice, obj=positions,values=np.ones(len(positions)), axis=1)
            sf_slice_up = np.transpose(np.insert(np.transpose(sf_slice_up),obj=positions,values=np.ones(len(positions)), axis=1))
            sf_slice_up = np.pad(sf_slice_up, (0,up_factor-1),  mode='constant', constant_values=1)
            sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=0)
            sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=1)
            sf_up.append(sf_slice_up)

        mask_slice = mask[0, :, :]
        positions = np.repeat(range(1, mask_slice.shape[1]), up_factor-1) #positions in mask slice to put 0
        mask_slice_up = np.insert(mask_slice, obj=positions,values=np.zeros(len(positions)), axis=1)
        mask_slice_up = np.transpose(np.insert(np.transpose(mask_slice_up),obj=positions,values=np.zeros(len(positions)), axis=1))
        mask_slice_up = np.pad(mask_slice_up, (0,up_factor-1),  mode='constant')
        mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=0)
        mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=1)
        mask_slice_up = mask_slice_up[np.newaxis, :]
        mask_up = np.repeat(mask_slice_up, mask.shape[0], axis=0)


        batch_sf_up.append(sf_up)
        batch_mask_up.append(mask_up)

    batch_sf_up = np.asarray(batch_sf_up)
    batch_sf_up = np.swapaxes(batch_sf_up, 3, 1)

    batch_mask_up = np.asarray(batch_mask_up)
    batch_mask_up = np.swapaxes(batch_mask_up, 3, 1)

    return batch_sf_up, batch_mask_up

def postprocessing(pred_sf, measured_sf, freq_num, pattern, factor):
    """ Perfoms all postprocessing steps.
        Args:
        pred_sf: np.ndarray
        measured_sf: np.ndarray
        freq_num: int
        pattern: np.ndarray
        factor: int
        Returns: np.ndarray
        """

    # Use linear regression to compute the rescaling parameters

    measured_sf_slice = copy.deepcopy(measured_sf[0, :, :, freq_num])

    # Downsampling pred_sf to compute regression using same positions.
    pred_sf_dw = downsampling(factor, pred_sf)

    x = np.asarray(pred_sf_dw[0, :, :, freq_num].flatten()[pattern])
    y = np.asarray(measured_sf_slice.flatten()[pattern])

    A = np.vstack([x, np.ones(len(x))]).T

    # compute regression coefficients
    m, c = np.linalg.lstsq(A, y, rcond=-1)[0]

    # rescale values
    reconstructed_sf_slice = pred_sf[0, :, :, freq_num]*m + c

    return reconstructed_sf_slice

# Code forked from https://github.com/mvoelk/keras_layers/

def conv_init_relu(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:3])
    v = v / (fan_in**0.5) * 2**0.5
    return K.constant(v, dtype=dtype)

class Conv2DBaseLayer(Layer):
    """Basic Conv2D class from which other layers inherit.
    """
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 #data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        super(Conv2DBaseLayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.rank = rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def get_config(self):
        config = super(Conv2DBaseLayer, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        })
        return config

class PartialConv2D(Conv2DBaseLayer):
    """2D Partial Convolution layer for sparse input data.

    # Arguments
        They are the same as for the normal Conv2D layer.
        binary: Boolean flag, whether the sparsity is propagated as binary
            mask or as float values.

    # Input shape
        features: 4D tensor with shape (batch_size, rows, cols, channels)
        mask: 4D tensor with shape (batch_size, rows, cols, channels)
            If the shape is (batch_size, rows, cols, 1), the mask is repeated
            for each channel. If no mask is provided, all input elements
            unequal to zero are considered as valid.

    # Example
        x, m = PartialConv2D(32, 3, padding='same')(x)
        x = Activation('relu')(x)
        x, m = PartialConv2D(32, 3, padding='same')([x,m])
        x = Activation('relu')(x)

    # Notes
        In contrast to Sparse Convolution, Partial Convolution propagates
        the sparsity for each channel separately. This makes it possible
        to concatenate the features and the masks from different branches
        in architecture.

    # References
        [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
        [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)
    """

    def __init__(self, filters, kernel_size,
                 kernel_initializer=conv_init_relu,
                 binary=True,
                 weightnorm=False,
                 eps=1e-6,
                 **kwargs):

        super(PartialConv2D, self).__init__(kernel_size, kernel_initializer=kernel_initializer, **kwargs)

        self.filters = filters
        self.binary = binary
        self.weightnorm = weightnorm
        self.eps = eps

    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
            mask_shape = input_shape[1]
            self.mask_shape = mask_shape
        else:
            feature_shape = input_shape
            self.mask_shape = feature_shape

        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)

        self.mask_kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.mask_kernel = tf.ones(self.mask_kernel_shape)
        self.mask_fan_in = tf.reduce_prod(self.mask_kernel_shape[:3])

        if self.weightnorm:
            self.wn_g = self.add_weight(name='wn_g',
                                        shape=(self.filters,),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        super(PartialConv2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
            mask = inputs[1]
            # if mask has only one channel, repeat
            if self.mask_shape[-1] == 1:
                mask = tf.repeat(mask, tf.shape(features)[-1], axis=-1)
        else:
            # if no mask is provided, get it from the features
            features = inputs
            mask = tf.where(tf.equal(features, 0), 0.0, 1.0)

        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.kernel), (0, 1, 2)) + self.eps)
            kernel = self.kernel / norm * self.wn_g
        else:
            kernel = self.kernel

        mask_kernel = self.mask_kernel

        features = tf.multiply(features, mask)
        features = K.conv2d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)

        norm = K.conv2d(mask, mask_kernel,
                        strides=self.strides,
                        padding=self.padding,
                        dilation_rate=self.dilation_rate)

        mask_fan_in = tf.cast(self.mask_fan_in, 'float32')

        if self.binary:
            mask = tf.where(tf.greater(norm, 0), 1.0, 0.0)
        else:
            mask = norm / mask_fan_in

        ratio = tf.where(tf.equal(norm, 0), 0.0, mask_fan_in / norm)

        features = tf.multiply(features, ratio)

        if self.use_bias:
            features = tf.add(features, self.bias)

        if self.activation is not None:
            features = self.activation(features)

        return [features, mask]

    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        space = feature_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        feature_shape = [feature_shape[0], *new_space, self.filters]
        mask_shape = [feature_shape[0], *new_space, self.filters]

        return [feature_shape, mask_shape]

    def get_config(self):
        config = super(PartialConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'binary': self.binary,
            'weightnorm': self.weightnorm,
            'eps': self.eps,
        })
        return config


# x = tf.random.normal((1,21,21,1))
#
# conv = PartialConv2D(32, 5, strides= 3, padding='same',
#                                   name='encoder_partialconv_')
#
# xout, mask = conv([x,x])
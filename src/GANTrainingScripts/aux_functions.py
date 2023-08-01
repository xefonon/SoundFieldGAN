import sys
sys.path.append('../')
sys.path.append('./')
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, Conv2DTranspose, BatchNormalization, Flatten
from tensorflow.keras.initializers import RandomNormal
from scipy.special import sph_harm
# fix relative import issue for sf_reconstruction_utils
from .sf_reconstruction_utils import single_measurement_sim
# from sf_reconstruction_utils import single_measurement_sim
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import yaml
from glob import glob
# physical_devices = tf.config.list_physical_devices('GPU')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)
def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)
def config_from_yaml(yamlFilePath, no_description = True):
    def changer(config):
        for attr, value in vars(config).items():
            setattr(config, attr, value.value)

    with open(yamlFilePath) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    config = dict2obj(dataMap)
    if no_description:
        changer(config)
    return config

def print_training_stats(epoch, num_epochs, steps_per_epoch, loss, i):
    keys = list(loss.keys())
    print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
          f"\033[93m[Batch {i%steps_per_epoch + 1}/{steps_per_epoch}]\033[0m"
          f" {keys[0]} = {loss[keys[0]]:.5f},"
          f" {keys[1]} = {loss[keys[1]]:.5f}", end='')

def normalize_complex(x_cmplx):
    real_data = np.real(x_cmplx)
    imag_data = np.imag(x_cmplx)

    real_data = (real_data - real_data.mean()) / real_data.std()
    imag_data = (imag_data - imag_data.mean()) / imag_data.std()

    x_norm = real_data + 1j * imag_data # values between 0-1
    return x_norm

def pressure_to_spherical_harmonics(pressure_values, grid, lmax):
    """
    Convert pressure measurements on a sphere to spherical harmonic coefficients.

    Args:
        pressure_values: An array of shape (n_points,) containing pressure measurements on the sphere.
        lmax: The maximum degree of the spherical harmonics expansion.

    Returns:
        A numpy array of shape (lmax+1)**2 containing the spherical harmonic coefficients.
    """
    # Construct the grid of spherical coordinates
    theta, phi, _ = cart2sph(grid[:, 0], grid[:, 1], grid[:, 2])

    # Compute the spherical harmonics for each degree and order up to lmax
    Ylm = np.zeros(((lmax + 1) ** 2, len(theta.flatten())), dtype=np.complex128)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Ylm[idx, :] = sph_harm(m, l, phi.flatten(), theta.flatten())
            idx += 1

    # Compute the coefficients by performing a least-squares fit
    Ylm = tf.constant(Ylm, dtype=tf.complex128)
    p = tf.constant(pressure_values.reshape(len(theta.flatten()), 1), dtype=tf.complex128)
    coefficients, _ = tf.linalg.lstsq(Ylm, p)

    # Convert the coefficients to a numpy array and return them
    coefficients = tf.reshape(coefficients, (lmax + 1) ** 2)
    return coefficients.numpy()


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def l2_normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.
    .. math::
        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}
    with :math:`\alpha \in [-pi, pi], \beta \in [0, \pi], r \geq 0`
    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates
    Returns
    -------
    theta : float or `numpy.ndarray`
            Azimuth angle in radians
    phi : float or `numpy.ndarray`
            Colatitude angle in radians (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi, r

def sph2cart(alpha, beta, r):
    """Spherical to cartesian coordinate transform.

    .. math::

        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return x, y, z

def plot_array_pressure(p_array, array_grid, ax=None, plane = False, norm = None, z_label = False):
    if ax is None:
        if z_label:
            ax = plt.axes(projection='3d')
        else:
            ax = plt.axes()

    cmp = plt.get_cmap("RdBu")
    if norm is None:
        vmin = p_array.real.min()
        vmax = p_array.real.max()
    else:
        vmin, vmax = norm
    if z_label:
        sc = ax.scatter(array_grid[0], array_grid[1], array_grid[2], c=p_array.real,
                        cmap=cmp, alpha=1., s=50, vmin = vmin, vmax = vmax)
    else:
        sc = ax.scatter(array_grid[0], array_grid[1], c=p_array.real,
                        cmap=cmp, alpha=1., s=50, vmin = vmin, vmax = vmax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if z_label:
        ax.set_zlabel('z [m]')
        ax.view_init(45, 45)

        if plane:
            ax.set_box_aspect((1,1,1))
        else:
            ax.set_box_aspect((array_grid[0].max(), array_grid[1].max(), array_grid[2].max()))
    return ax, sc

def fib_sphere(num_points, radius=1):
    radius = tf.cast(radius, dtype=tf.float32)
    ga = (3 - tf.math.sqrt(5.)) * math.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * tf.range(num_points, dtype=tf.float32)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = tf.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = tf.math.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * tf.sin(theta)
    x = alpha * tf.cos(theta)

    x_batch = tf.tensordot(radius, x, 0)
    y_batch = tf.tensordot(radius, y, 0)
    z_batch = tf.tensordot(radius, z, 0)

    # expand the dimensions of each coordinate
    shape = x_batch.shape.as_list()
    if len(shape) < 3:
        x_batch = tf.expand_dims(x_batch, 0)
        y_batch = tf.expand_dims(y_batch, 0)
        z_batch = tf.expand_dims(z_batch, 0)

    # Stack the x, y, z coordinates into a single array
    grid = tf.stack([x_batch, y_batch, z_batch], axis=-1)
    if len(grid.shape.as_list()) > 3:
        grid = tf.squeeze(grid, axis=0)
    xyz = tf.transpose(grid, perm=[0, 2, 1])
    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch[27, :], y_batch[27, :], z_batch[27, :], s = 3)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()

    return xyz

class Complex2DTConv(Layer):

    def __init__(self, filters, kernel_size, strides=2, padding='same',
                 kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                 name='Cmplx_1DConv', activation="relu",
                 kernel_constraint=None, **kwargs):
        # self.filters = filters // 2 # allocate half the features to real, half to imaginary
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        if activation == 'lrelu':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = tf.keras.activations.get(activation)
        self.kernel_constraint = kernel_constraint
        super(Complex2DTConv, self).__init__(name=name, **kwargs)

        self.conv2dT_real_real = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides,
                                                 padding=self.padding,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name=name + '_real_real',
                                                 activation=None,
                                                 kernel_constraint=self.kernel_constraint,
                                                 **kwargs)
        self.conv2dT_real_imag = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides,
                                                 padding=self.padding,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name=name + '_real_imag',
                                                 activation=None,
                                                 kernel_constraint=self.kernel_constraint,
                                                 **kwargs)
        self.conv2dT_imag_imag = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides,
                                                 padding=self.padding,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name=name + '_imag_imag',
                                                 activation=None,
                                                 kernel_constraint=self.kernel_constraint,
                                                 **kwargs)
        self.conv2dT_imag_real = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides,
                                                 padding=self.padding,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name=name + '_imag_real',
                                                 activation=None,
                                                 kernel_constraint=self.kernel_constraint,
                                                 **kwargs)

    def call(self, input_tensor):
        x_real = tf.math.real(input_tensor)
        x_imag = tf.math.imag(input_tensor)

        x_real_real = self.conv2dT_real_real(x_real)
        x_real_imag = self.conv2dT_real_imag(x_real)
        x_imag_imag = self.conv2dT_imag_imag(x_imag)
        x_imag_real = self.conv2dT_imag_real(x_imag)

        x_real_out = x_real_real - x_imag_imag
        x_imag_out = x_imag_real + x_real_imag

        if self.activation is not None:
            x_real_out = self.activation(x_real_out)
            x_imag_out = self.activation(x_imag_out)

        x_out = tf.complex(x_real_out, x_imag_out)
        return x_out


class Complex2DConv(Layer):

    def __init__(self, filters, kernel_size, strides=2, padding='same',
                 kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                 name='Cmplx_1DConv', activation="relu",
                 kernel_constraint=None, **kwargs):
        self.filters = filters // 2  # allocate half the features to real, half to imaginary
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        if activation == 'lrelu':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = tf.keras.activations.get(activation)
        self.kernel_constraint = kernel_constraint

        super(Complex2DConv, self).__init__(name=name, **kwargs)

        self.conv2d_real_real = Conv2D(self.filters, self.kernel_size, strides=self.strides,
                                       padding=self.padding,
                                       kernel_initializer=self.kernel_initializer,
                                       name=name + '_real_real',
                                       activation=None,
                                       kernel_constraint=self.kernel_constraint,
                                       **kwargs)
        self.conv2d_real_imag = Conv2D(self.filters, self.kernel_size, strides=self.strides,
                                       padding=self.padding,
                                       kernel_initializer=self.kernel_initializer,
                                       name=name + '_real_imag',
                                       activation=None,
                                       kernel_constraint=self.kernel_constraint,
                                       **kwargs)
        self.conv2d_imag_imag = Conv2D(self.filters, self.kernel_size, strides=self.strides,
                                       padding=self.padding,
                                       kernel_initializer=self.kernel_initializer,
                                       name=name + '_imag_imag',
                                       activation=None,
                                       kernel_constraint=self.kernel_constraint,
                                       **kwargs)
        self.conv2d_imag_real = Conv2D(self.filters, self.kernel_size, strides=self.strides,
                                       padding=self.padding,
                                       kernel_initializer=self.kernel_initializer,
                                       name=name + '_imag_real',
                                       activation=None,
                                       kernel_constraint=self.kernel_constraint,
                                       **kwargs)

    def call(self, input_tensor):
        x_real = tf.math.real(input_tensor)
        x_imag = tf.math.imag(input_tensor)

        x_real_real = self.conv2d_real_real(x_real)
        x_real_imag = self.conv2d_real_imag(x_real)
        x_imag_imag = self.conv2d_imag_imag(x_imag)
        x_imag_real = self.conv2d_imag_real(x_imag)

        x_real_out = x_real_real - x_imag_imag
        x_imag_out = x_imag_real + x_real_imag

        if self.activation is not None:
            x_real_out = self.activation(x_real_out)
            x_imag_out = self.activation(x_imag_out)

        x_out = tf.complex(x_real_out, x_imag_out)
        return x_out


class ComplexDense(Layer):
    def __init__(self, units, use_bias=True,
                 kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                 name='Cmplx_Dense', activation="relu",
                 kernel_constraint=None, **kwargs):
        self.units = units #// 2  # allocate half the features to real, half to imaginary
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        if activation == 'lrelu':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = tf.keras.activations.get(activation)
        self.kernel_constraint = kernel_constraint

        super(ComplexDense, self).__init__(name=name, **kwargs)

        self.dense_real_real = Dense(self.units, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     name=name + '_real_real',
                                     activation=None,
                                     kernel_constraint=self.kernel_constraint,
                                     **kwargs)
        self.dense_real_imag = Dense(self.units, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     name=name + '_real_imag',
                                     activation=None,
                                     kernel_constraint=self.kernel_constraint,
                                     **kwargs)
        self.dense_imag_imag = Dense(self.units, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     name=name + '_imag_imag',
                                     activation=None,
                                     kernel_constraint=self.kernel_constraint,
                                     **kwargs)
        self.dense_imag_real = Dense(self.units, use_bias=self.use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     name=name + '_imag_real',
                                     activation=None,
                                     kernel_constraint=self.kernel_constraint,
                                     **kwargs)

    def call(self, input_tensor):
        x_real = tf.math.real(input_tensor)
        x_imag = tf.math.imag(input_tensor)

        x_real_real = self.dense_real_real(x_real)
        x_real_imag = self.dense_real_imag(x_real)
        x_imag_imag = self.dense_imag_imag(x_imag)
        x_imag_real = self.dense_imag_real(x_imag)

        x_real_out = x_real_real - x_imag_imag
        x_imag_out = x_imag_real + x_real_imag

        if self.activation is not None:
            x_real_out = self.activation(x_real_out)
            x_imag_out = self.activation(x_imag_out)

        x_out = tf.complex(x_real_out, x_imag_out)
        return x_out

class ComplexBatchNormalization(Layer):
    def __init__(self, axis=-1, momentum=0.1, epsilon=1e-5, name = '', **kwargs):
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        super(ComplexBatchNormalization, self).__init__(name=name, **kwargs)
        self.bn_r = BatchNormalization(self.axis, self.momentum, self.epsilon)
        self.bn_i = BatchNormalization(self.axis, self.momentum, self.epsilon)

    def call(self, input_tensor):
        x_real = tf.math.real(input_tensor)
        x_imag = tf.math.imag(input_tensor)

        x_real = self.bn_r(x_real)
        x_imag = self.bn_i(x_imag)
        x_out = tf.complex(x_real, x_imag)
        return x_out


class ComplexFlatten(Layer):
    def __init__(self, name = '', **kwargs):

        super(ComplexFlatten, self).__init__(name=name, **kwargs)
        self.flatten_r = Flatten()
        self.flatten_i = Flatten()

    def call(self, input_tensor):
        # tf.print(f"inputs at ComplexFlatten are {inputs.dtype}")
        x_real = tf.math.real(input_tensor)
        x_imag = tf.math.imag(input_tensor)

        real_flat = self.flatten_r(x_real)
        imag_flat = self.flatten_r(x_imag)
        return tf.complex(real_flat, imag_flat)  # Keep input dtype


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)

def instance_norm(inputs,
                  center=True,
                  scale=True,
                  epsilon=1e-6,
                  activation_fn=None,
                  param_initializers=None,
                  reuse=None,
                  outputs_collections=None,
                  trainable=True,
                  data_format= 'NHWC',
                  scope=None):
  """Functional interface for the instance normalization layer.
  Reference: https://arxiv.org/abs/1607.08022.
    "Instance Normalization: The Missing Ingredient for Fast Stylization"
    Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `tf.nn.relu`), this can
      be disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    scope: Optional scope for `variable_scope`.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  inputs = tf.convert_to_tensor(value=inputs)
  inputs_shape = inputs.shape
  inputs_rank = inputs.shape.ndims
  DATA_FORMAT_NCHW = 'NCHW'
  DATA_FORMAT_NHWC = 'NHWC'

  if inputs_rank is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  with tf.compat.v1.variable_scope(
      scope, 'InstanceNorm', [inputs], reuse=reuse):
    if data_format == DATA_FORMAT_NCHW:
      reduction_axis = 1
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, tf.compat.dimension_value(inputs_shape[1])] +
          [1 for _ in range(2, inputs_rank)])
    else:
      reduction_axis = inputs_rank - 1
      params_shape_broadcast = None
    moments_axes = list(range(inputs_rank))
    del moments_axes[reduction_axis]
    del moments_axes[0]
    params_shape = inputs_shape[reduction_axis:reduction_axis + 1]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}
    if center:
      beta_initializer = param_initializers.get(
          'beta', tf.compat.v1.initializers.zeros())
      beta = tf.compat.v1.get_variable(
          name='beta',
          shape=params_shape,
          dtype=dtype,
          initializer=beta_initializer,
          trainable=trainable)
      if params_shape_broadcast:
        beta = tf.reshape(beta, params_shape_broadcast)
    if scale:
      gamma_initializer = param_initializers.get(
          'gamma', tf.compat.v1.initializers.ones())
      gamma = tf.compat.v1.get_variable(
          name='gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=gamma_initializer,
          trainable=trainable)
      if params_shape_broadcast:
        gamma = tf.reshape(gamma, params_shape_broadcast)

    # Calculate the moments (instance activations).
    mean, variance = tf.nn.moments(x=inputs, axes=moments_axes, keepdims=True)

    # Compute instance normalization.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon, name='instancenorm')
    if activation_fn is not None:
      outputs = activation_fn(outputs)

    if outputs_collections:
      tf.compat.v1.add_to_collections(outputs_collections, outputs)

    return outputs
def get_latent_vector(batch_size, latent_dim):
    return tf.random.normal(shape=(batch_size, latent_dim))

def array_to_complex(x):
    return tf.complex(x[..., 0], x[..., 1])

def generate_random_pressure_fields(frq, G, Nfields = 5, plot = True, conditionalG = False,
                                    normal_prior = True, normalize = True, grid_dim = 21):
    pressure = []
    f_vec = np.fft.rfftfreq(16384, 1 / 16000)
    f_ind = np.argmin(f_vec < frq)
    f = f_vec[f_ind]
    x,y = np.meshgrid(np.linspace(-.5, .5, grid_dim), np.linspace(-0.5, 0.5, grid_dim))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    grid = np.stack([x, y, z], axis=1).T
    # change grid type to np.float32
    grid = np.float32(grid)
    H, _ = tf_sensing_mat(f, G.output.shape[1], grid)
    pm_train_all = []
    pwcoeffs_train_all = []
    for ii in range(Nfields):
        pm_train, pwcoeffs_train, pw_pos = single_measurement_sim(frq, n_plane_waves= 4096, snr = 30, grid_dim= grid_dim,
                                                          normal_prior = normal_prior, normalize= normalize)
        pm_train_all.append(pm_train.numpy().squeeze(0))
        pwcoeffs_train_all.append(pwcoeffs_train.numpy().squeeze(0))
    if plot:
        fig, ax = plt.subplots(4, Nfields, figsize=(12,12))

    maxcoef = np.max(abs(np.array(pwcoeffs_train_all)))
    pw_pos[0] = np.rad2deg(pw_pos[0])
    pw_pos[1] = np.rad2deg(pw_pos[1])
    norm = (np.array(pm_train_all).real.min(), np.array(pm_train_all).real.max())
    for i in range(Nfields):
        if conditionalG:
            # get frequency as tensorflow vector
            frq_vec = tf.constant([frq], dtype=tf.float32)
            coeffs = G([get_latent_vector(1, 128), frq_vec])
        else:
            coeffs = G(get_latent_vector(1, 128))
        coeffs = array_to_complex(coeffs)
        fake_sound_fields = tf.einsum('ijk, ik -> ij', H, coeffs)
        pressure.append(fake_sound_fields.numpy().squeeze(0))
        coeffs = coeffs.numpy().squeeze(0)
        maxcoef = np.maximum(maxcoef, abs(coeffs).max())
        if plot:
            ax[0,i], sc = plot_array_pressure(fake_sound_fields.numpy().squeeze(0), grid, ax=ax[0,i], norm=norm)
            ax[0,i].set_title('G - f = {} Hz'.format(int(f)))
            ax[0,i].set_xlabel('x [m]')
            ax[0,i].set_ylabel('y [m]')
            ax[0,i].set_aspect('equal')
            # ax[1, i].stem(abs(coeffs),  linefmt='grey', markerfmt=' ')
            # plot scatter plot with marker face corresponding to magnitude

            sc3 = ax[1, i].scatter(pw_pos[0], pw_pos[1], c=abs(coeffs), cmap='viridis', marker='o', s=2,
                                   vmin = np.abs(pwcoeffs_train_all).min(), vmax = np.abs(pwcoeffs_train_all).max())
            ax[1,i].set_xlabel(r'azi [$^\circ$]')
            ax[1,i].set_ylabel(r'elev [$^\circ$]')
            # make marker face color none
            # ax[1,i].set_ylim([0, maxcoef])
            # True fields
            ax[2,i], sc2 = plot_array_pressure(pm_train_all[i], grid, ax=ax[2,i], norm=norm)
            ax[2,i].set_title('True - f = {} Hz'.format(int(f)))
            ax[2,i].set_xlabel('x [m]')
            ax[2,i].set_ylabel('y [m]')
            ax[2,i].set_aspect('equal')
            # ax[3, i].stem(abs(pwcoeffs_train_all[i]), linefmt='grey', markerfmt=' ')
            sc4 = ax[3, i].scatter(pw_pos[0], pw_pos[1], c=abs(pwcoeffs_train_all[i]), cmap='viridis', marker='o', s=2,
                                   vmin = np.abs(pwcoeffs_train_all).min(), vmax = np.abs(pwcoeffs_train_all).max())
            # add colorbar to ax[3, i]

            ax[3,i].set_xlabel(r'azi [$^\circ$]')
            ax[3,i].set_ylabel(r'elev [$^\circ$]')
            # make marker face color none
            # ax[3,i].set_ylim([0, maxcoef])


    if plot:
        # make space for colorbar
        fig.subplots_adjust(right=0.8)
        fig.colorbar(sc, ax=ax[0,i])
        fig.colorbar(sc2, ax=ax[2,i])
        fig.colorbar(sc3, ax=ax[1,i])
        fig.colorbar(sc4, ax=ax[3,i])
        # fig.show()
    # print std

    return fig, pressure, grid

def reference_grid(steps, xmin = -.7, xmax = .7, z = 0):
    x = tf.linspace(xmin, xmax, steps)
    y = tf.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = tf.meshgrid(x, y)
    Z = z*tf.ones(X.shape)
    return X,Y,Z


def wavenumber(f, n_PW, c = 343.):
    k = 2*math.pi*f/c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

def random_wavenumber(f, n_Pw, c = 343):
    k = 2*np.pi*f/c
    k_grid = tf.random.normal((3, n_Pw))
    k_grid = k*k_grid/tf.linalg.norm(k_grid)
    return k_grid

def get_sensing_mat(f, n_pw, X, Y, Z, k_samp=None):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw)
        # k_samp = random_wavenumber(f, n_pw)

    kx, ky, kz = k_samp
    if tf.experimental.numpy.ndim(kx) < 2:
        kx = tf.expand_dims(kx, 0)
        ky = tf.expand_dims(ky, 0)
        kz = tf.expand_dims(kz, 0)
    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z, mesh = True)
    column_norm = tf.linalg.norm(H, axis = 2, keepdims = True)
    H = H/column_norm
    return H, k_out

@tf.function
def tf_sensing_mat(f, n_pw, grid, k_samp=None, c=343):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c)
    H = build_sensing_mat(k_samp, grid)
    # column_norm = tf.linalg.norm(H, axis=1, keepdims=True)
    # H = H / column_norm
    return H, k_samp

def build_sensing_mat(k, grid):
    """ Build the sensing matrix for a given set of plane waves and microphones

    Args:
        k (tf.Tensor): 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
        grid (tf.Tensor): 3 x n_microphones tensor, where n_microphones is the number of microphones

    Returns:
        tf.Tensor: n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
            each column corresponds to a plane wave
        """
    # k is a 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
    # grid is a 3 x n_microphones tensor, where n_microphones is the number of microphones
    # Returns a n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
    # each column corresponds to a plane wave
    # Compute the dot product between k and x
    dot_product = tf.einsum('oij,ik->ojk', k, grid)

    # Compute the sensing matrix using the dot product and the exponential function
    sensing_matrix = tf.math.exp(-tf.complex(0., dot_product))

    # Transpose the sensing matrix to get the desired shape
    return tf.transpose(sensing_matrix, perm=[0,2,1])

def create_sfs_from_meshRIR_set(data_dir, source_pos = 2, sample_rate = 16000, nfft = 8192):
    import re
    from librosa import resample
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    rir_files = glob(data_dir + '/ir*.npy')
    rir_files.sort(key=natural_keys)
    grid_file = glob(data_dir + 'pos_mic.npy')
    all_freq_responses = []
    for file in rir_files:
        rirs_ = np.load(file)
        rir = rirs_[source_pos]
        rir = resample(rir, 48000, sample_rate)
        if len(rir) < int(2*nfft):
            rir = np.pad(rir, (0, int(2*nfft) - len(rir)))
        else:
            rir = rir[:int(2*nfft)]
        freq_resp = np.fft.rfft(rir, int(2*nfft) )
        all_freq_responses.append(freq_resp)
    grid = np.load(grid_file[0])
    return np.array(all_freq_responses), grid.T

def create_MeshRIR_frequency_response_dict(data_dir):
    freq_responses_src_pos = {}
    for ii in range(32):
        freq_resp, grid = create_sfs_from_meshRIR_set(data_dir, ii)
        freq_responses_src_pos[f'position_{ii}'] = freq_resp
    freq_responses_src_pos['grid'] = grid
    np.savez('MeshRIR_FResponses', freq_responses_src_pos)
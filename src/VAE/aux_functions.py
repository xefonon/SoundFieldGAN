import numpy as np
import tensorflow as tf
import math
import tensorflow_probability as tfp

def get_centre_freq_octave_bands(bands=10, octaves=True):
    # Calculate Third Octave Bands (base 2)
    fcentre = 10 ** 3 * (2 ** (np.arange(-18, 13) / 3))
    if octaves:
        fcentre = fcentre[::3]
    return np.round(fcentre, 1).tolist()[3:(bands + 3)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_mask(shape_, subsample_ratio=0.2, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    if shape_.ndims > 3:
        batchsize, dim1, dim2, chan = shape_
    else:
        dim1, dim2, chan = shape_
        batchsize = 1
    mask = tfp.distributions.Bernoulli(probs=subsample_ratio).sample((batchsize, dim1 * dim2))
    mask_reshaped = tf.reshape(mask, (batchsize, dim1, dim2, 1))
    if chan > 1:
        mask_reshaped = tf.concat([mask_reshaped, mask_reshaped], axis=-1)
    mask_reshaped = tf.cast(mask_reshaped, tf.float32)
    return mask_reshaped

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


def build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=False):
    if mesh:
        H = tf.math.exp(-tf.complex(0., tf.einsum('ij,k -> ijk', kx, tf.reshape(X, [-1])) + tf.einsum('ij,k -> ijk', ky, tf.reshape(Y, [-1])) + tf.einsum('ij,k -> ijk', kz, tf.reshape(Z, [-1]))))
    else:
        H = tf.math.exp(-tf.complex(0., tf.einsum('ij,k -> ijk', kx, X) + tf.einsum('ij,k -> ijk', ky,Y) + tf.einsum('ij,k -> ijk', kz, Z)))
    return tf.transpose(H, perm = [0,2,1])/len(kx)

def wavenumber(f, n_PW, c = 343):
    k = 2*math.pi*f/c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

def fib_sphere(num_points, radius = 1):
    radius = tf.cast(radius, dtype = tf.float32)
    ga = (3 - tf.math.sqrt(5.)) * math.pi # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * tf.range(num_points, dtype = tf.float32)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = tf.linspace(1/num_points-1, 1-1/num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = tf.math.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * tf.sin(theta)
    x = alpha * tf.cos(theta)

    x_batch = tf.tensordot(radius, x, 0)
    y_batch = tf.tensordot(radius, y, 0)
    z_batch = tf.tensordot(radius, z, 0)

    return [x_batch,y_batch,z_batch]

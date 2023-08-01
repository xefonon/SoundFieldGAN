"""
Functions used for training generative
models for sound field reconstruction
"""

import numpy as np
import tensorflow as tf
import math
import json
import yaml
from glob import glob
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib as mpl

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


def config_from_yaml(yamlFilePath, no_description=True):
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
          f"\033[93m[Batch {i % steps_per_epoch + 1}/{steps_per_epoch}]\033[0m"
          f" {keys[0]} = {loss[keys[0]]:.5f},", end='')


def normalize_complex(x_cmplx):
    real_data = np.real(x_cmplx)
    imag_data = np.imag(x_cmplx)

    real_data = (real_data - real_data.mean()) / real_data.std()
    imag_data = (imag_data - imag_data.mean()) / imag_data.std()

    x_norm = real_data + 1j * imag_data  # values between 0-1
    return x_norm


def modal_overlap(Trev, V, c, freq):
    return 12. * tf.math.log(10.) * V * freq ** 2 / (Trev * c ** 3)


def number_plane_waves(Trev, V, c, freq):
    """
    Number of plane waves according to oblique modes in a room
    e.g. F. Jacobsen "Fundamentals of general linear acoustics" pp. 137-140
    Parameters
    ----------
    Trev
    V
    c
    freq

    Returns
    -------

    """
    M = modal_overlap(Trev, V, c, freq)
    return 8 * M


def cmplx_to_array(cmplx):
    real = tf.math.real(cmplx)
    imag = tf.math.imag(cmplx)
    arr = tf.concat([tf.expand_dims(real, -1), tf.expand_dims(imag, -1)], axis=-1)
    return arr


def array_to_complex(arr):
    cmplx = arr[..., 0] + 1j * arr[..., 1]
    return cmplx


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def l2_normalize(v, eps=1e-12):
    """l2 normalize the input vector."""
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

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch[27, :], y_batch[27, :], z_batch[27, :], s = 3)
    # plt.show()
    return [x_batch, y_batch, z_batch]

def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim=None, tex=False, cmap=None, normalise=True,
            colorbar = False, cbar_label = '', cbar_loc = 'bottom'):
    """
    Plot spatial soundfield normalised amplitude
    --------------------------------------------
    Args:
        P : Pressure in meshgrid [X,Y]
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)
    """
    # plot_settings()

    N_interp = 1500
    if normalise:
        Pvec = P / np.max(abs(P))
    else:
        Pvec = P
    res = complex(0, N_interp)
    Xc, Yc = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    Pmesh = griddata(points, Pvec, (Xc, Yc), method='cubic', rescale=True)
    if cmap is None:
        cmap = 'coolwarm'
    # P = P / np.max(abs(P))
    X = Xc.flatten()
    Y = Yc.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * X.ptp() / Pmesh.size
    dy = 0.5 * Y.ptp() / Pmesh.size
    if ax is None:
        _, ax = plt.subplots()  # create figure and axes
    im = ax.imshow(Pmesh.real, cmap=cmap, origin='upper',
                   extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    if clim is not None:
        lm1, lm2 = clim
        im.set_clim(lm1, lm2)
    if colorbar:
        if cbar_loc != 'bottom':
            shrink = 1.
            orientation = 'vertical'
        else:
            shrink = 1.
            orientation = 'horizontal'

        cbar = plt.colorbar(im, ax = ax, location=cbar_loc,
                            shrink=shrink)
        # cbar.ax.get_yaxis().labelpad = 15
        titlesize = int(1. * mpl.rcParams['axes.titlesize'])
        # cbar.ax.set_title(cbar_label, fontsize = titlesize)
        cbar.set_label(cbar_label, fontsize = titlesize)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        if f is None:
            ax.set_title(name)
            print(f)
        else:
            ax.set_title(name + 'f : {} Hz'.format(f))
            print(f)
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax, im

def reference_grid(steps, xmin=-.7, xmax=.7, z=0):
    x = tf.linspace(xmin, xmax, steps)
    y = tf.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = tf.meshgrid(x, y)
    Z = z * tf.ones(X.shape)
    return np.asarray([X, Y, Z])


def wavenumber(f, n_PW, c=343):
    k = 2 * math.pi * f / c
    k_grid = fib_sphere(n_PW, k)
    return k_grid


def random_wavenumber(f, n_Pw, c=343):
    k = 2 * np.pi * f / c
    k_grid = tf.random.normal((3, n_Pw))
    k_grid = k * k_grid / tf.linalg.norm(k_grid)
    return k_grid


def get_sensing_mat(f, n_pw, X, Y, Z, k_samp=None, c=None):
    # Basis functions for coefficients
    if c is None:
        c = 343.
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c)

    kx, ky, kz = k_samp
    if tf.experimental.numpy.ndim(kx) < 2:
        kx = tf.expand_dims(kx, 0)
        ky = tf.expand_dims(ky, 0)
        kz = tf.expand_dims(kz, 0)
    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=True)
    column_norm = tf.linalg.norm(abs(H), axis=2, keepdims=True)
    H = H / tf.complex(column_norm, 0.)
    return H, k_out


def build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=False):
    if mesh:
        H = tf.math.exp(-tf.complex(0., tf.einsum('ij,k -> ijk', kx, tf.reshape(X, [-1])) + tf.einsum('ij,k -> ijk', ky,
                                                                                                      tf.reshape(Y, [
                                                                                                          -1])) + tf.einsum(
            'ij,k -> ijk', kz, tf.reshape(Z, [-1]))))
    else:
        H = tf.math.exp(-tf.complex(0., tf.einsum('ij,k -> ijk', kx, X) + tf.einsum('ij,k -> ijk', ky, Y) + tf.einsum(
            'ij,k -> ijk', kz, Z)))
    return tf.transpose(H, perm=[0, 2, 1]) / len(kx)


def create_sfs_from_meshRIR_set(data_dir, source_pos=2, sample_rate=16000, nfft=8192):
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
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    rir_files = glob(data_dir + '/ir*.npy')
    rir_files.sort(key=natural_keys)
    grid_file = glob(data_dir + 'pos_mic.npy')
    all_freq_responses = []
    for file in rir_files:
        rirs_ = np.load(file)
        rir = rirs_[source_pos]
        rir = resample(rir, 48000, sample_rate)
        if len(rir) < int(2 * nfft):
            rir = np.pad(rir, (0, int(2 * nfft) - len(rir)))
        else:
            rir = rir[:int(2 * nfft)]
        freq_resp = np.fft.rfft(rir, int(2 * nfft))
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


def create_transfer_learning_dataset(data_dir,
                                     batch_size=16,
                                     epochs=100,
                                     return_unbatched=True,
                                     n_plane_waves=1000
                                     ):
    rirs = np.load(data_dir + '/MeshRIR_set.npz')
    grid_mic = np.float32(rirs['mic_pos']).T
    grid_source = rirs['src_pos'].T
    f = np.float32(rirs['freq_vec'])
    responses = rirs['spatio_temporal_responses']
    freq = np.tile(f, (responses.shape[0], 1))
    responses = np.transpose(responses, (0, 2, 1))
    freq = np.transpose(freq, (1, 0))
    responses = responses.reshape(-1, responses.shape[-1])
    freq = freq.reshape(-1, 1)
    # shuffle once
    n_data = len(responses)
    p = np.random.permutation(len(responses))
    responses = responses[p]
    responses = tf.reshape(responses, [n_data, grid_mic.shape[0], grid_mic.shape[1]])
    freq = freq[p]
    grids = tf.repeat(grid_mic[None, ...], len(freq), axis=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((responses, grids, None, freq))
    if return_unbatched:
        return train_dataset, tf.convert_to_tensor(grid_mic), n_data
    else:

        train_dataset = train_dataset.repeat(epochs
                                             ).shuffle(buffer_size=10000
                                                       ).batch(batch_size=batch_size,
                                                               drop_remainder=True).prefetch(2)

        return train_dataset, tf.convert_to_tensor(grid_mic), n_data


def create_sound_fields_dataset(sampling_rate=16000,
                                N_fft_size=16384,
                                avg_snr=30,
                                n_plane_waves=3000,
                                grid_dimension=21,
                                batch_size=16,
                                epochs=100,
                                n_fields=80000,
                                freq_vector=None,
                                augment=False,
                                config=None):
    if freq_vector is None:
        f = np.fft.rfftfreq(N_fft_size, 1 / sampling_rate)
    else:
        f = freq_vector
    ds_series = tf.data.Dataset.from_generator(
        simulate_measurements,
        args=[f, n_plane_waves, avg_snr, 1, grid_dimension,
              n_fields, config.use_gaussian_prior, augment, config.real_data_dir
              ],
        output_types=(tf.float32, tf.float32, tf.complex64, tf.float32),
        output_shapes=((grid_dimension, grid_dimension, 2),
                       (3, np.prod((grid_dimension, grid_dimension))),
                       (np.prod((grid_dimension, grid_dimension)), n_plane_waves),
                       (1,))

    )  # output_shapes=([21, 21], [3, 21*21], [21*21, 2000]))
    ds_series_batch = ds_series.repeat(epochs).batch(batch_size).prefetch(2 * batch_size)
    return ds_series_batch


def read_real_dataset(data_dir, return_subset=False):
    rirs = np.load(str(data_dir) + '/MeshRIR_set.npz')
    grid_mic = np.float32(rirs['mic_pos']).T
    grid_source = rirs['src_pos'].T
    f = np.float32(rirs['freq_vec'])
    responses = rirs['spatio_temporal_responses']
    freq = np.tile(f, (responses.shape[0], responses.shape[1], 1))
    responses = np.transpose(responses, (0, 2, 1))
    freq = np.transpose(freq, (0, 2, 1))
    responses = responses.reshape(-1, responses.shape[-1])
    freq = freq.reshape(-1, freq.shape[-1])[..., 0]
    # shuffle once
    n_data = len(responses)
    p = np.random.permutation(len(responses))
    responses = responses[p]
    # responses = np.reshape(responses, [n_data, 21, 21])
    freq = freq[p]
    indices = np.random.choice(np.arange(n_data), 10, replace=False)
    if return_subset:
        return responses[indices], freq[indices], grid_mic
    else:
        return responses, freq, grid_mic


def simulate_measurements(frequency_vector,
                          n_plane_waves=3000,
                          snr=None,
                          batchsize=1,
                          grid_dim=21,
                          n_fields=80000,
                          normal_prior=False,
                          augment=False,
                          real_data_dir='./',
                          deterministic=False,
                          complex_net=None):
    n_plane_waves = int(n_plane_waves)
    if deterministic:
        n_fields = len(frequency_vector)
    if augment:
        try:
            real_data_dir = real_data_dir.decode("utf-8")
        except:
            real_data_dir = real_data_dir
        real_responses, real_freq, real_grid_mic = read_real_dataset(data_dir=real_data_dir)
        nreal_fields = len(real_responses)
        n_fields += nreal_fields
    else:
        nreal_fields = 10000
    for ii, jj in zip(range(n_fields), range(nreal_fields)):
        if ii % 2 == 0 and augment:
            grid = real_grid_mic
            frequency = real_freq[jj]
            H = sensing_mat_transfer_learning(grid, frequency, temperature=17.1, n_plane_waves=n_plane_waves)
            H = tf.tile(H, tf.constant([batchsize, 1, 1]))
            pm = real_responses[jj]
            dim1, dim2 = 21, 21
        else:
            if deterministic:
                if ii > len(frequency_vector) - 1:
                    frequency = np.random.choice(frequency_vector, size=1)
                else:
                    frequency = frequency_vector[ii]
            else:
                frequency = np.random.choice(frequency_vector, size=1)
            # modal overlap -> number of plane waves
            Trev = np.random.uniform(0.08, 2.1)
            V = np.random.uniform(25, 120)
            T = 20 + 1 * np.random.randn()  # + jitter
            c = speed_of_sound(T)
            n_active_waves = int(tf.round(number_plane_waves(Trev, V, c, frequency)).numpy())
            if n_active_waves > n_plane_waves:
                n_active_waves = n_plane_waves
            mask = get_random_np_boolean_mask(n_active_waves, n_plane_waves)
            grid = reference_grid(grid_dim, xmin=-.5, xmax=.5)
            _, dim1, dim2 = grid.shape
            grid = tf.reshape(grid, [3, np.prod((dim1, dim2))])

            H, k = get_sensing_mat(frequency,
                                   n_plane_waves,
                                   grid[0],
                                   grid[1],
                                   grid[2],
                                   c=c)

            H = tf.tile(H, tf.constant([batchsize, 1, 1]))

            if not normal_prior:
                pw_phase = tf.random.uniform(shape=(batchsize, n_plane_waves), minval=0, maxval=2 * math.pi)

                pw_mag = tf.random.uniform(shape=(batchsize, n_plane_waves), minval=0, maxval=1)
                # pw_mag = tf.random.gamma(shape=(batchsize, n_plane_waves), alpha = 0.5, beta = 1 )

                pw_mag = tf.where(mask, pw_mag, tf.zeros_like(pw_mag))  # mask magnitude
                pwcoeff = tf.complex(pw_mag, 0.) * tf.exp(tf.complex(0., pw_phase))
            else:
                pwcoeff = coeff_priors(shape=(batchsize, n_plane_waves),
                                       sparse=True)  # normally or Laplace distributed hierarchical
            pm = tf.einsum('ijk, ik -> ij', H, pwcoeff)
            if snr is not None:
                pm = adjustSNR(pm, snrdB=snr + np.random.uniform(-5, 5), td=False)
        pm = pm / tf.complex(tf.reduce_max(tf.abs(pm)), 0.)
        # pm, _ = tf.linalg.normalize(pm)
        pm = tf.reshape(pm, [dim1, dim2])
        pm = cmplx_to_array(tf.squeeze(pm))
        freq = tf.convert_to_tensor(frequency)
        if freq.ndim < 1:
            freq = tf.expand_dims(freq, axis=0)
        yield pm, grid, tf.squeeze(H), freq


def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * tf.sqrt(273.15 + T)
    return c


def sensing_mat_transfer_learning(grid, freq, temperature=17.1, n_plane_waves=4000):
    c = speed_of_sound(temperature)

    grid = tf.cast(grid, tf.float32)
    H, k = get_sensing_mat(tf.cast(freq, tf.float32),
                           n_plane_waves,
                           grid[0],
                           grid[1],
                           grid[2],
                           c=c
                           )
    return H


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_centre_freq_octave_bands(bands=10, octaves=True):
    # Calculate Third Octave Bands (base 2)
    fcentre = 10 ** 3 * (2 ** (np.arange(-18, 13) / 3))
    if octaves:
        fcentre = fcentre[::3]
    return np.round(fcentre, 1).tolist()[3:(bands + 3)]


def sample_from_generator(generator, grid, grid_nd=21, fs=16000, Nfft=16384, H=None, z_latent_vec=None,
                          latent_dim=128,
                          n_bands=10, config=None):
    freq = np.fft.rfftfreq(Nfft, 1 / fs)
    frq = get_centre_freq_octave_bands(bands=n_bands)
    new_freqs = [find_nearest(freq, f) for f in frq]

    new_freqs = tf.convert_to_tensor(new_freqs)
    get_latent = lambda shape: tf.complex(tf.random.normal(shape, mean=0, stddev=.5),
                                          tf.random.normal(shape, mean=0, stddev=.5))

    if z_latent_vec is None:
        z = get_latent((n_bands, latent_dim))
    else:
        z = z_latent_vec
    fake_coefficients = generator(z, training=False)
    if H is None:
        H, _ = get_sensing_mat(new_freqs,
                               generator.output.shape[1],
                               grid[0],
                               grid[1],
                               grid[2])

    fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients)
    fake_sound_fields = tf.reshape(fake_sound_fields, (n_bands, grid_nd, grid_nd))
    return fake_sound_fields, new_freqs


def pad_boundaries(input_, zeros=11):
    if zeros % 2 == 0:
        zeros_left = zeros // 2
        zeros_right = zeros // 2
    else:
        zeros_left = zeros // 2 + 1
        zeros_right = zeros // 2
    return tf.pad(input_, tf.constant([[0, 0], [zeros_left, zeros_right], [zeros_left, zeros_right], [0, 0]]))


def get_mask(shape_, subsample_ratio=0.2, seed=None, flatten=False):
    if seed is not None:
        tf.random.set_seed(seed)
    if shape_.ndims > 3:
        batchsize, dim1, dim2, chan = shape_
    else:
        dim1, dim2, chan = shape_
        batchsize = 1
    mask = tfp.distributions.Bernoulli(probs=subsample_ratio).sample((shape_))
    mask_reshaped = tf.reshape(mask, (batchsize, dim1, dim2, 1))
    if chan > 1:
        mask_reshaped = tf.concat([mask_reshaped, mask_reshaped], axis=-1)
    mask_reshaped = tf.cast(mask_reshaped, tf.float32)
    if flatten:
        mask_reshaped = tf.reshape(mask_reshaped, [-1])
    return mask_reshaped


def mask_pressure_field(input_, subsample_ratio=0.2, mask=None):
    if mask is None:
        mask = get_mask(input_.shape, subsample_ratio=subsample_ratio)
    # masked = tf.boolean_mask(input_, mask)
    masked = mask * input_
    return masked, mask


def downsample_tf(input_, down_sample_factor=2):
    batch_size, dim, _, chan = input_.shape

    return tf.strided_slice(input_, [0, 0, 0, 0], [batch_size, dim, dim, 1],
                            [1, down_sample_factor, down_sample_factor, 1])


def upsample_zeros(input_, factor=2, output_shape=None):
    batch_size, height, width, n_channels = tf.shape(input_)  # might not work in graph mode
    if output_shape is None:
        output_shape = [batch_size, factor * height, factor * width, 1]
    upsampled_input = tf.nn.conv2d_transpose(input_, tf.ones([1, 1, 1, 1]), output_shape, strides=factor,
                                             padding='VALID')
    return upsampled_input


def preprocess_chain(input_, mask, factor=2, pad_size=21):
    inputsize = input_.shape[1]
    input_shape = input_.shape
    # Downsampling
    # input_ = downsample_tf(input_, down_sample_factor=factor) # downsample soundfield
    # mask = downsample_tf(mask, down_sample_factor=factor) # downsample mask
    # Masking
    masked_input, _ = mask_pressure_field(input_, mask=mask)
    jitter = tf.random.normal(masked_input.shape, mean=0, stddev=1e-6)
    # pressure is already scaled at this point
    masked_input = tf.where(masked_input == 0, jitter, masked_input)
    # upsampling
    # upsampled_masked_input = upsample_zeros(masked_input, factor = 2, output_shape= input_shape)
    # upsampled_mask = upsample_zeros(mask, factor = 2, output_shape= input_shape)

    # if pad_size != inputsize:
    #     zeros = pad_size - inputsize
    #     upsampled_masked_input = pad_boundaries(upsampled_masked_input, zeros)
    #     upsampled_mask = pad_boundaries(upsampled_mask, zeros)
    #
    return masked_input, mask


def sample_from_real_data(fs=16000, Nfft=16384, n_waves=4096, snr=30, grid_dim=21, n_bands=10, config=None):
    complex_to_array = lambda x: np.concatenate((x[..., None].real, x[..., None].imag), axis=-1)

    freq = np.fft.rfftfreq(Nfft, 1 / fs)
    frq = get_centre_freq_octave_bands(bands=n_bands)
    new_freqs = [find_nearest(freq, f) for f in frq]

    data_gen = simulate_measurements(new_freqs,
                                     n_plane_waves=n_waves,
                                     snr=snr,
                                     batchsize=1,
                                     grid_dim=grid_dim,
                                     deterministic=True,
                                     normal_prior=config.use_gaussian_prior,
                                     augment=False
                                     )
    ps = []
    Hs = []
    for data in data_gen:
        pm, _, Hm, frq_tensor = data
        pm = pm / np.max(abs(pm))
        ps.append(pm)
        Hs.append(Hm)

    ps = np.stack(ps)
    Hs = np.stack(Hs)
    if config is not None:
        if config.augmented_dset:
            realdata, frequencies, grid_mic = read_real_dataset(config.real_data_dir, return_subset=True)
            for i, response in enumerate(realdata):
                realdata[i], _ = tf.linalg.normalize(response)
            ps = np.concatenate((ps, complex_to_array(realdata.reshape((10, 21, 21)))), axis=0)
            if frequencies.ndim > 1:
                frequencies = np.squeeze(frequencies, axis=-1)
            new_freqs += list(frequencies)
    return ps, Hs, new_freqs


def coeff_priors(shape=(500,), sparse=False):
    if len(shape) < 2:
        waves = shape[0]
    else:
        batchsize = shape[0]
        waves = shape[1]
    mu_r = tfp.distributions.Normal(0, 10.).sample(waves)
    tau_r = tfp.distributions.LogNormal(1., 1.).sample(waves)
    mu_i = tfp.distributions.Normal(0, 10.).sample(waves)
    tau_i = tfp.distributions.LogNormal(1., 1.).sample(waves)
    if not sparse:
        coeffs_r = tfp.distributions.Normal(mu_r, tau_r).sample(batchsize)
        coeffs_i = tfp.distributions.Normal(mu_i, tau_i).sample(batchsize)
    else:
        coeffs_r = tfp.distributions.Laplace(mu_r, tau_r).sample(batchsize)
        coeffs_i = tfp.distributions.Laplace(mu_i, tau_i).sample(batchsize)
    return tf.complex(coeffs_r, coeffs_i)


def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    td : boolean, optional
        Indicates whether noise is added in time domain or not.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from signal
    ndim = sig.ndim
    if ndim > 2:
        dims = [-2, -1]
    else:
        dims = -1
    mean, _ = tf.nn.moments(sig, axes=dims)
    if ndim > 2:
        sig_zero_mean = sig - mean[..., tf.newaxis, tf.newaxis]
    else:
        sig_zero_mean = sig - mean[..., tf.newaxis]

    _, var = tf.nn.moments(sig_zero_mean, axes=dims)
    if ndim > 2:
        psig = var[..., tf.newaxis, tf.newaxis]
    else:
        psig = var[..., tf.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        noise = tf.sqrt(pnoise) * tf.random.normal(sig.shape)
    else:
        # complex valued white noise
        real_noise = tf.random.normal(mean=0, stddev=np.sqrt(2) / 2, shape=sig.shape)
        imag_noise = tf.random.normal(mean=0, stddev=np.sqrt(2) / 2, shape=sig.shape)
        noise = tf.complex(real_noise, imag_noise)
        noise_mag = tf.sqrt(pnoise) * tf.complex(tf.abs(noise), 0.)
        noise = noise_mag * tf.exp(tf.complex(0., tf.math.angle(noise)))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise


class Sparam(tf.keras.layers.Layer):
    def __init__(self, name="injective", **kwargs):
        super(Sparam, self).__init__(trainable=True, name=name, **kwargs)

    def build(self, input_shape):
        self.s_sq = self.add_weight(name="alpha",
                                    shape=[1],
                                    initializer=tf.keras.initializers.zeros,
                                    dtype=tf.float32,
                                    trainable=True)

    def call(self, x):
        neg = -tf.square(self.s_sq) * x
        return neg


class Injective_Constraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self):
        self.s_sq = Sparam()

    def __call__(self, w):
        _, _, wx, wy = w.shape
        wdiff = (wy - tf.math.floor(wy / 2)) % 2

        wlen1 = tf.math.floor(wy / 2)
        wlen2 = tf.math.floor(wy / 2) + wdiff
        wlen1 = tf.cast(wlen1, tf.int32)
        wlen2 = tf.cast(wlen2, tf.int32)

        wpart = w[..., :wlen1]
        wpart2 = self.s_sq(w[..., :wlen2])
        wnew = tf.concat([wpart, wpart2], axis=-1)
        return wnew

    def get_config(self):
        return {'s_sq': self.s_sq}


class ConstantTensorConstraint(tf.keras.constraints.Constraint):
    """Constrains tensors to `t`."""

    def __init__(self, t):
        self.t = t

    def __call__(self, w):
        return self.t

    def get_config(self):
        return {'t': self.t}


def get_random_np_boolean_mask(n_true_elements, total_n_elements):
    assert total_n_elements >= n_true_elements
    a = np.zeros(total_n_elements, dtype=int)
    a[:n_true_elements] = 1
    np.random.shuffle(a)
    return a.astype(bool)


def cmplx_to_array(cmplx):
    real = tf.math.real(cmplx)
    imag = tf.math.imag(cmplx)
    arr = tf.concat([tf.expand_dims(real, -1), tf.expand_dims(imag, -1)], axis=-1)
    return arr

# imports
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import tensorflow_probability as tfp
from scipy.special import sph_harm


def sample_points_in_sphere(npoints, radius=1):
    vec = tf.random.normal((3, npoints))
    vec /= tf.linalg.norm(vec, axis=0)
    vec *= radius
    return vec


def _random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")


def plot_array_pressure(p_array, array_grid, ax=None, plane=False, norm=None, z_label=False):
    """
    Plot pressure on array

    Parameters
    ----------
    p_array : Tensor
        Pressure on array.
    array_grid : Tensor
        Array grid.
    ax : Axes3D, optional
        Axes to plot on. The default is None.
    plane : bool, optional
        Plot in 2D. The default is False.
    norm : tuple, optional
        Normalization. The default is None.
    z_label : bool, optional
        Plot z label. The default is False.

    Returns
    -------
    ax : Axes3D
        Axes with plot.
    sc : colorbar object
        Colorbar object.

    """
    from matplotlib.colors import Normalize
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
                        cmap=cmp, alpha=1., s=100, vmin=vmin, vmax=vmax)
    else:
        sc = ax.scatter(array_grid[0], array_grid[1], c=p_array.real,
                        cmap=cmp, alpha=1., s=100, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if z_label:
        ax.set_zlabel('z [m]')
        ax.view_init(45, 45)

        if plane:
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_box_aspect((array_grid[0].max(), array_grid[1].max(), array_grid[2].max()))
    return ax, sc

# @tf.function(autograph=False, experimental_compile=False)
def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    noise : Vector or Tensor, optional
        Noise Tensor. The default is None.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from signal
    ndim = (len(sig.shape.as_list()))
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
    # pdb.set_trace()
    pnoise = psig / tf.cast(snr_lin, psig.dtype)

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
def pressure_to_spherical_harmonics(pressure_values, Ybasis):
    """
    Convert pressure measurements on a sphere to spherical harmonic coefficients.

    Args:
        pressure_values: An array of shape (n_points,) containing pressure measurements on the sphere.
        lmax: The maximum degree of the spherical harmonics expansion.

    Returns:
        A numpy array of shape (lmax+1)**2 containing the spherical harmonic coefficients.
    """
    # Compute the coefficients by performing a least-squares fit
    coefficients = spatialFT(pressure_values, spherical_harmonic_bases = Ybasis, leastsq_fit = True)
    # Convert the coefficients to a numpy array and return them
    # coefficients = tf.reshape(coefficients, (lmax + 1) ** 2)
    return coefficients

def mnArrays(nMax):
    """Generate degrees n and orders m up to nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    Returns
    -------
    m : (int), array_like
        0, -1, 0, 1, -2, -1, 0, 1, 2, ... , -nMax ..., nMax
    n : (int), array_like
        0, 1, 1, 1, 2, 2, 2, 2, 2, ... nMax, nMax, nMax
    """
    # Degree n = 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    degs = np.arange(nMax + 1)
    n = np.repeat(degs, degs * 2 + 1)

    # Order m = 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
    # http://oeis.org/A196199
    elementNumber = np.arange((nMax + 1) ** 2) + 1
    t = np.floor(np.sqrt(elementNumber - 1)).astype(int)
    m = elementNumber - t * t - t - 1

    return m, n
def sph_harm_all(nMax, grid, kind="complex"):
    """Compute all spherical harmonic coefficients up to degree nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    az: (float), array_like
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float), array_like
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    Returns
    -------
    y_mn : (complex float) or (float), array_like
        Spherical harmonics of degrees n [0 ... nMax] and all corresponding
        orders m [-n ... n], sampled at [az, co]. dim1 corresponds to az/co
        pairs, dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1,
        -2/2, -1/2 ...
    """
    az, co, _ = cart2sph(grid[0], grid[1], grid[2])
    m, n = mnArrays(nMax)
    mA, azA = tf.meshgrid(m.astype(np.float32), az)
    nA, coA = tf.meshgrid(n.astype(np.float32), co)
    return sph_harmonics(mA, nA, azA, coA, kind=kind)

def sph_harmonics(m, n, az, co, kind="complex"):
    """Compute spherical harmonics.
    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n
    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0
    az : (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float)
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type according to complex [7]_ or
        real definition [8]_ [Default: 'complex']
    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at theta = az,
        phi = co
    References
    ----------
    .. [7] `scipy.special.sph_harm()`
    .. [8] Zotter, F. (2009). Analysis and Synthesis of Sound-Radiation with
        Spherical Arrays University of Music and Performing Arts Graz, Austria,
        192 pages.
    """
    # SAFETY CHECKS
    kind = kind.lower()
    if kind not in ["complex", "real"]:
        raise ValueError("Invalid kind: Choose either complex or real.")
    m = np.atleast_1d(m)

    Y = tf.convert_to_tensor(sph_harm(m, n, az, co))
    if kind == "complex":
        return Y
    else:  # kind == 'real'
        mg0 = m > 0
        ml0 = m < 0
        Y[mg0] = np.float_power(-1.0, m)[mg0] * np.sqrt(2) * np.real(Y[mg0])
        Y[ml0] = np.sqrt(2) * np.imag(Y[ml0])
        return tf.convert_to_tensor(np.real(Y))
def spatialFT(data, position_grid = None, order_max=10, kind="complex",
              spherical_harmonic_bases=None, weight=None,
              leastsq_fit=False):
    """Perform spatial Fourier transform.
    Parameters
    ----------
    data : array_like
        Data to be transformed, with signals in rows and frequency bins in
        columns
    position_grid : array_like cartesian coordinates of spatial sampling points [3, Npoints], optional
    order_max : int, optional
        Maximum transform order [Default: 10]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]
    Returns
    -------
    Pnm : array_like
        Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in
        columns
    Notes
    -----
    In case no weights in spatial sampling grid are given, the pseudo inverse
    of the SH bases is computed according to Eq. 3.34 in [5]_.
    References
    ----------
    .. [5] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
        (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg,
        2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8
    """

    if spherical_harmonic_bases is None:
        if position_grid is None:
            raise ValueError("No spatial sampling grid given.")
        azi, elev, r = cart2sph(position_grid[0], position_grid[1], position_grid[2])
        spherical_harmonic_bases = tf.convert_to_tensor(sph_harm_all(
            order_max, azi, elev, kind=kind
        ), dtype=tf.complex64 if kind == 'complex' else tf.float32)
    if leastsq_fit:
        if data.shape[0] != spherical_harmonic_bases.shape[1]:
            data = tf.transpose(data)
        coeffs =  tf.linalg.lstsq(spherical_harmonic_bases, data, fast=False)
        if coeffs.shape[1] == 1:
            coeffs = tf.squeeze(coeffs)
        return coeffs
    else:
        if weight is None:
            # calculate pseudo inverse in case no spatial sampling point weights
            # are given
            spherical_harmonics_weighted = tf.linalg.pinv(spherical_harmonic_bases)
        else:
            # apply spatial sampling point weights in case they are given
            spherical_harmonics_weighted = tf.math.conj(spherical_harmonic_bases).numpy().T * (
                    4 * np.pi * weight
            )
            spherical_harmonics_weighted = tf.convert_to_tensor(spherical_harmonics_weighted,
                                                                dtype=spherical_harmonic_bases.dtype)

            return tf.matmul(spherical_harmonics_weighted, data)


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
    if x_batch.shape[0] != 1:
        x_batch = tf.expand_dims(x_batch, 0)
        y_batch = tf.expand_dims(y_batch, 0)
        z_batch = tf.expand_dims(z_batch, 0)

    # pdb.set_trace()
    # Stack the x, y, z coordinates into a single array
    xyz = tf.transpose(tf.stack([x_batch, y_batch, z_batch], axis=-1), perm=[0, 2, 1])
    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch[27, :], y_batch[27, :], z_batch[27, :], s = 3)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()

    return xyz

def fib_sphere_numpy(num_points, radius=1):
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle
    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)
    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)
    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)
    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)
    x_batch = np.tensordot(radius, x, 0)
    y_batch = np.tensordot(radius, y, 0)
    z_batch = np.tensordot(radius, z, 0)
    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 20)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()
    return np.array([x_batch, y_batch, z_batch])


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


def wavenumber(f, n_PW, c=343):
    k = 2 * math.pi * f / c
    k_grid = fib_sphere(n_PW, k)
    return k_grid


def get_random_np_boolean_mask(n_true_elements, total_n_elements):
    assert total_n_elements >= n_true_elements
    a = np.zeros(total_n_elements, dtype=int)
    a[:n_true_elements] = 1
    np.random.shuffle(a)
    return a.astype(bool)

def apply_random_mask(input_tensor, number_of_nonzeros):
    """
    Applies a random mask to a tensor such that a certain number of elements
    are set to zero for each batch in n_batches.

    Args:
    - input_tensor: A Tensor of shape (n_elements,)
    - number_of_nonzeros: The number of non-zero elements to retain

    Returns:
    - A Tensor of shape (n_elements,) with the same values as input_tensor, but with a certain number of elements set to zero.
    """
    input_tensor = tf.squeeze(input_tensor, 0)
    n_elements = tf.shape(input_tensor)[0]
    indices = tf.random.shuffle(tf.range(n_elements))
    indices = indices[:number_of_nonzeros]
    indices = tf.cast(indices, dtype=tf.int64)
    indices = tf.sort(indices, axis=0)
    mask = tf.sparse.SparseTensor(indices=tf.expand_dims(indices, axis=1), values=tf.ones(tf.shape(indices)[0], dtype=tf.float32), dense_shape=[n_elements])
    mask = tf.sparse.to_dense(mask)
    if mask.dtype != input_tensor.dtype:
        mask = tf.cast(mask, dtype=input_tensor.dtype)
    new_tensor = input_tensor*mask
    return tf.expand_dims(new_tensor, 0)

def get_sensing_matrix(k, x):
    # k is a 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
    # x is a 3 x n_microphones tensor, where n_microphones is the number of microphones
    # Returns a n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
    # each column corresponds to a plane wave

    # Expand dimensions of k to make it broadcastable with x
    k_expanded = tf.expand_dims(k, axis=1)

    # Compute the dot product between k and x
    sensing_matrix = tf.reduce_sum(k_expanded * x, axis=0)

    return sensing_matrix


@tf.function
def tf_sensing_mat(f, n_pw, grid, k_samp=None, c=343):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c)
    H = build_sensing_mat(k_samp, grid)
    # column_norm = tf.linalg.norm(H, axis=1, keepdims=True)
    # H = H / column_norm
    return H, k_samp


def get_sensing_mat(f, n_pw, X, Y, Z, k_samp=None, c=343, mesh=True):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c)
        # k_samp = random_wavenumber(f, n_pw)

    kx, ky, kz = k_samp
    if tf.experimental.numpy.ndim(kx) < 2:
        kx = tf.expand_dims(kx, 0)
        ky = tf.expand_dims(ky, 0)
        kz = tf.expand_dims(kz, 0)
    elif tf.experimental.numpy.ndim(kx) > 2:
        kx = tf.squeeze(kx, 1)
        ky = tf.squeeze(ky, 1)
        kz = tf.squeeze(kz, 1)

    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=mesh)
    column_norm = tf.linalg.norm(H, axis=2, keepdims=True)
    H = H / column_norm
    return H, k_out


def reference_grid(steps, rmin=-.7, rmax=.7, z_=0, plane = 'xy'):
    if plane == 'xy':
        x = tf.linspace(rmin, rmax, steps)
        y = tf.linspace(rmin, rmax, steps)
        X, Y = tf.meshgrid(x, y)
        Z = z_ * tf.ones(X.shape)
    elif plane == 'xz':
        x = tf.linspace(rmin, rmax, steps)
        z = tf.linspace(rmin, rmax, steps)
        X, Z = tf.meshgrid(x, z)
        Y = z_ * tf.ones(X.shape)
    elif plane == 'yz':
        y = tf.linspace(rmin, rmax, steps)
        z = tf.linspace(rmin, rmax, steps)
        Y, Z = tf.meshgrid(y, z)
        X = z_ * tf.ones(Y.shape)
    else:
        raise ValueError('plane must be one of xy, xz, yz')
    return tf.convert_to_tensor([X, Y, Z])


def build_sensing_mat(k, grid):
    """
    Builds the sensing matrix for a given set of plane waves and microphones

    Args:
    - k: A 3 x n_plane_waves tensor, where n_plane_waves is the number of plane waves
    - grid: A 3 x n_microphones tensor, where n_microphones is the number of microphones

    Returns:
    - A n_microphones x n_plane_waves tensor, where each row corresponds to a microphone and
    each column corresponds to a plane wave
    """
    dot_product = tf.einsum('oij,ik->ojk', k, grid)

    # Compute the sensing matrix using the dot product and the exponential function
    sensing_matrix = tf.math.exp(-tf.complex(0., dot_product))

    # Transpose the sensing matrix to get the desired shape
    return tf.transpose(sensing_matrix, perm=[0,2,1])


def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
    return a_oo / np.abs(a_oo).max()


def cmplx_to_array(cmplx):
    real = tf.math.real(cmplx)
    imag = tf.math.imag(cmplx)
    arr = tf.concat([tf.expand_dims(real, -1), tf.expand_dims(imag, -1)], axis=-1)
    return arr


import tensorflow as tf

@tf.function
def mixture_of_gaussians(N, n_plane_waves, seed=None):
    # Set random seed
    if seed is not None:
        tf.random.set_seed(seed)

    # Define the weight of the zero component
    weight_zero = (N - n_plane_waves) / N
    weight_non_zero = 1 - weight_zero

    # Define the Gaussian components
    zero_component = tfp.distributions.Normal(loc=0., scale=1e-6)
    non_zero_component = tfp.distributions.Normal(loc=0., scale=1.)

    # Define the mixture model
    mixture = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=[weight_zero, weight_non_zero]),
        components=[zero_component, non_zero_component])

    # Generate samples
    real_samples = mixture.sample(sample_shape=N)
    imag_samples = mixture.sample(sample_shape=N)
    samples = tf.complex(real_samples, imag_samples)
    return tf.expand_dims(samples, 0)

def coeff_priors(shape=(500,), sparse=False):
    if len(shape) == 1:
        batchsize = 1
        waves = shape[0]
    else:
        batchsize = shape[0]
        waves = shape[1]
    sigma_mu_r = np.random.uniform(0, 1)
    sigma_mu_i = np.random.uniform(0, 1)
    mu_r = tfp.distributions.Normal(0, sigma_mu_r).sample(waves)
    tau_r = tfp.distributions.Gamma(concentration=0.05, rate=2).sample(waves)
    mu_i = tfp.distributions.Normal(0, sigma_mu_i).sample(waves)
    tau_i =tfp.distributions.Gamma(concentration=0.05, rate=2).sample(waves)
    if not sparse:
        coeffs_r = tfp.distributions.Normal(mu_r, tau_r).sample(batchsize)
        coeffs_i = tfp.distributions.Normal(mu_i, tau_i).sample(batchsize)
    else:
        coeffs_r = tfp.distributions.Laplace(mu_r, tau_r).sample(batchsize)
        coeffs_i = tfp.distributions.Laplace(mu_i, tau_i).sample(batchsize)
    return tf.complex(coeffs_r, coeffs_i)


def sensing_mat_transfer_learning(grid, freq, temperature=17.1, n_plane_waves=4000):
    c = speed_of_sound(temperature)

    grid = tf.cast(grid, tf.float32)
    H, k = get_sensing_mat(tf.cast(freq, tf.float32),
                           n_plane_waves,
                           grid[0],
                           grid[1],
                           grid[2],
                           c=c,
                           mesh=True)
    return H


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


def normalize_with_moments(x, axes=[1], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
    return x_normed, mean, tf.sqrt(variance)

def normalize_pressure(x, axis = 1):
    """ Unit norm normalization of pressure field """
    norm = tf.norm(x, axis=axis, keepdims=True)
    normalised_x = x / norm
    return normalised_x, norm

def von_mises_plane_waves(k, mean_direction = [1., 0., 0.],
                          concentration = 0., n_plane_waves=3000, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    vm = tfp.distributions.VonMises(loc=mean_direction, concentration=concentration)
    samples = vm.sample(n_plane_waves)
    norms = tf.linalg.norm(samples, axis=-1, keepdims=True)
    k_samples = k * samples / norms

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(k_samples[:, 0], k_samples[:, 1], k_samples[:, 2], s=3)
    ax.view_init(30, 30)
    plt.show()

def simulate_measurements(frequency_vector,
                          n_plane_waves=3000,
                          snr_lims=None,
                          batchsize=1,
                          grid_dim=21,
                          n_fields=80000,
                          normal_prior=False,
                          augment=False,
                          real_data_dir='./',
                          deterministic=False,
                          normalize=True,
                          grid_sphere=None,
                          Ybasis=None):
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
        # real_grid_mic = real_grid_mic.reshape(3, 21,21)
        n_fields += nreal_fields
    else:
        nreal_fields = 10000
    if grid_sphere is None:
        grid_sphere = tf.squeeze(fib_sphere(1000), 0)
    # if Ybasis is None:
    #     Ybasis = sph_harm_all(nMax= 14, grid=grid_sphere)
    grid_xy = reference_grid(grid_dim, rmin=-.5, rmax=.5, plane='xy')
    grid_xz = reference_grid(grid_dim, rmin=-.5, rmax=.5, plane='xz')
    grid_yz = reference_grid(grid_dim, rmin=-.5, rmax=.5, plane='yz')
    grids = [grid_xy, grid_xz, grid_yz]
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
            Trev = np.random.uniform(0.1, .4)
            V = np.random.uniform(70, 140)
            T = np.random.uniform(15, 25)
            c = speed_of_sound(T)
            n_active_waves = int(tf.round(number_plane_waves(Trev, V, c, frequency)).numpy())
            if n_active_waves > n_plane_waves:
                n_active_waves = n_plane_waves
            # mask = get_random_np_boolean_mask(n_active_waves, n_plane_waves)


            H_sphere, k = tf_sensing_mat(frequency,
                                            n_plane_waves,
                                            grid_sphere,
                                            c=c)
            H_sphere = tf.tile(H_sphere, tf.constant([batchsize, 1, 1]))

            _, dim1, dim2 = grid_xy.shape
            grid = tf.reshape(grids[np.random.choice([0,1,2])], [3, np.prod((dim1, dim2))])
            H, k = tf_sensing_mat(frequency,
                                  n_plane_waves,
                                  grid,
                                  c=c)

            H = tf.tile(H, tf.constant([batchsize, 1, 1]))

            if not normal_prior:
                pw_phase = tf.random.uniform(shape=(batchsize, n_plane_waves), minval=0, maxval=2 * math.pi)
                set_waves_to_zero = tfp.distributions.Binomial(total_count=1, probs =  min(1,n_active_waves/n_plane_waves)).sample((batchsize, n_plane_waves))
                if tf.reduce_sum(set_waves_to_zero) == 0:
                    set_waves_to_zero = tf.random.uniform(shape=(batchsize, n_plane_waves), minval=0, maxval= .1)
                pw_mag = set_waves_to_zero*tf.random.truncated_normal(shape=(batchsize, n_plane_waves), mean=0.0, stddev=.25)
                # pw_mag = np.random.rayleigh(scale=0.4, size=(batchsize, n_plane_waves))
                # pw_mag = tfp.random.rayleigh(scale=0.2, shape=(batchsize, n_plane_waves))
                # pw_mag = apply_random_mask(pw_mag, n_active_waves)
                pwcoeff = tf.complex(pw_mag, 0.) * tf.exp(tf.complex(0., pw_phase))
                pwcoeff /= n_plane_waves
            else:
                # pwcoeff = mixture_of_gaussians( n_plane_waves, n_active_waves//2) # /2 because complex valued
                pwcoeff = coeff_priors((batchsize, n_plane_waves))
                # pwcoeff = apply_random_mask(pwcoeff, n_active_waves) # normally or Laplace distributed hierarchical
                pwcoeff /= n_plane_waves
            # if normalize:
                # pwcoeff = 0.01*pwcoeff / tf.cast(tf.reduce_max(tf.abs(pwcoeff)), tf.complex64)
            pm = tf.einsum('ijk, ik -> ij', H, pwcoeff)
            # pm_sphere = tf.einsum('ijk, ik -> ij', H_sphere, pwcoeff)
            if snr_lims is not None:
                snr = np.random.uniform(snr_lims[0], snr_lims[1])
                pm = adjustSNR(pm, snrdB=snr, td=False)
                # pm_sphere = adjustSNR(pm_sphere, snrdB=snr, td=False)

        if normalize:
            pm, norm = normalize_pressure(pm, axis= 1)
            # pm_sphere = pm_sphere/norm
            pwcoeff = pwcoeff/norm
        #     pm, mu, std = normalize_with_moments(pm, axes=1)
        #     pm_sphere = (pm_sphere - mu)/std
        #     pwcoeff = pwcoeff/std
        sph_harm_coeffs = 0.
        # sph_harm_coeffs = pressure_to_spherical_harmonics(pm_sphere,Ybasis)
        pm = tf.reshape(pm, [dim1, dim2])
        pm = cmplx_to_array(tf.squeeze(pm))
        freq = tf.convert_to_tensor(frequency)
        if freq.ndim < 1:
            freq = tf.expand_dims(freq, axis=0)
        yield pm, grid, tf.squeeze(H), tf.squeeze(H_sphere), freq, tf.reshape(pwcoeff, (n_plane_waves,)), sph_harm_coeffs

def single_measurement_sim(frequency, n_plane_waves=3000, snr=None, grid_dim=21, normal_prior=False, normalize=True):
    n_plane_waves = int(n_plane_waves)
    frequency = float(frequency)
    grid = reference_grid(grid_dim, rmin=-.5, rmax=.5, z_=np.random.uniform(-3.5, 3.5))
    _, dim1, dim2 = grid.shape
    grid = tf.reshape(grid, [3, np.prod((dim1, dim2))])
    Trev = np.random.uniform(0.1, .4)
    V = np.random.uniform(70, 140)
    T = np.random.uniform(15, 25)
    c = speed_of_sound(T)
    n_active_waves = int(tf.round(number_plane_waves(Trev, V, c, frequency)).numpy())
    if n_active_waves > n_plane_waves:
        n_active_waves = n_plane_waves
    H, k = tf_sensing_mat(frequency, n_plane_waves, grid, c=c)
    if not normal_prior:
        pw_phase = tf.random.uniform(shape=(n_plane_waves,), minval=0, maxval=2 * math.pi)
        set_waves_to_zero = tfp.distributions.Binomial(total_count=1,
                                                       probs=min(1, n_active_waves / n_plane_waves)).sample((n_plane_waves,))
        if tf.reduce_sum(set_waves_to_zero) == 0:
            set_waves_to_zero = tf.random.uniform(shape=(n_plane_waves,), minval=0, maxval=.1)
        pw_mag = set_waves_to_zero * tf.random.truncated_normal(shape=(n_plane_waves,), mean=0.0, stddev=.25)
        # pw_mag = tf.random.truncated_normal(shape=(n_plane_waves,), mean=0.0, stddev=.25)
        pwcoeff = tf.complex(pw_mag, 0.) * tf.exp(tf.complex(0., pw_phase))
    else:
        # pwcoeff = mixture_of_gaussians(n_plane_waves, n_active_waves // 2)  # /2 because complex valued
        pwcoeff = coeff_priors((n_plane_waves,))
    if len(pwcoeff.shape.as_list()) < 2:
        pwcoeff = tf.expand_dims(pwcoeff, 0)
    # if normalize:
    #     pwcoeff = 0.01*pwcoeff / tf.cast(tf.reduce_max(tf.abs(pwcoeff)), tf.complex64)
    pw_pos = tf.squeeze(fib_sphere(n_plane_waves))
    pw_azi, pw_elev, _ = cart2sph(pw_pos[0], pw_pos[1], pw_pos[2])
    pm = tf.einsum('ijk, ik -> ij', H, pwcoeff)
    if snr is not None:
        pm = adjustSNR(pm, snrdB=snr + np.random.uniform(-5, 5), td=False)
    if normalize:
        pm, norm = normalize_pressure(pm, axis=1)
        pwcoeff = pwcoeff / norm
    #     if len(pm.shape.as_list()) < 2:
    #         pm = tf.expand_dims(pm, 0)
    #     # normalize with moments
    #     pm, mu, std = normalize_with_moments(pm, axes=1)
    #     pwcoeff = pwcoeff / std

    return pm, pwcoeff, np.array([pw_azi, pw_elev])

def array_to_complex(arr):
    cmplx = arr[..., 0] + 1j * arr[..., 1]
    return cmplx


def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def distance_between(s, r):
    """Distance of all combinations of locations in s and r
    Args:
        s (ndarray [3, N]): cartesian coordinates of s
        r (ndarray [3, M]): cartesian coordinates of r
    Returns:
        ndarray [M, N]: distances in meters
    """
    return np.linalg.norm(s[:, None, :] - r[:, :, None], axis=0)


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
    c = 20.05 * np.sqrt(273.15 + T)
    return c


def spatial_coherence_mat(frq, array_pos, T=None):
    if T is None:
        T = 20
    c = speed_of_sound(T)
    kvec = 2 * np.pi * frq / c
    # K = len(kvec)
    # euclidean distances between source and receivers
    r = distance_between(array_pos, array_pos)

    # stupid equation from paper again:
    # phi = (r[..., np.newaxis]/c) * (np.pi * fs * kvec) / K
    # more appropriate equation:
    phi = r[..., np.newaxis] * kvec
    sp_coherence = np.sinc(phi)
    return sp_coherence


def diffuse_kernel(freq_vec, array_pos, sigma):
    cov = sigma ** 2 * spatial_coherence_mat(freq_vec, array_pos)
    return cov


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


def circularly_symm_cov(frq, array_pos, sigma):
    """
    https://en.wikipedia.org/wiki/Complex_normal_distribution
    """
    cov_all = diffuse_kernel(frq, array_pos, sigma)
    # imaginary part of covariance is zero
    realcov = cov_all
    imcov = np.zeros_like(realcov)
    c_cov_real = 0.5 * np.hstack((realcov, imcov))
    c_cov_imag = 0.5 * np.hstack((imcov, realcov))
    covariance_mat = np.vstack((c_cov_real, c_cov_imag))
    return covariance_mat

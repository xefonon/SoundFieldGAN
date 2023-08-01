import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
# import tensorflow_probability as tfp
tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


def array_to_complex(arr):
    cmplx = tf.complex(arr[..., 0], arr[..., 1])
    return cmplx

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

def fib_sphere(num_points, radius=1):
    radius = tf.cast(radius, dtype=tf.float32)
    ga = (3 - tf.math.sqrt(5.)) * np.pi  # golden angle

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

def wavenumber(f, n_PW, c = 343.):
    k = 2*np.pi*f/c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

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

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

def locator(self):
    return plt.MultipleLocator(self.number / self.denominator)

def formatter(self):
    return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


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

def cmplx_to_array(cmplx):
    real = tf.math.real(cmplx)
    imag = tf.math.imag(cmplx)
    arr = tf.concat([real, imag], axis=-1)
    return arr

def scatter_pol_plot(theta, phi, c, ax = None):
    if ax is None:
        ax = plt.gca()
    sc = ax.scatter(theta, phi, c = c)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r'$\phi$')
    return ax, sc


def cos_sim(a, b):
    return abs(np.vdot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b))

def get_latent_z(batchsize = 1):
    z = tf.random.normal([batchsize, 128])
    return tf.Variable(z/tf.math.maximum(tf.norm(z, ord = 2), 1), name='z_batch')

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def get_optimizer(learning_rate, opt_type, decay_steps = None):
    if decay_steps is not None:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, 0.9
        )
    if opt_type == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate)
    if opt_type == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, momentum = 0.9)
    elif opt_type == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate)
    elif opt_type == 'adam':
        return tf.keras.optimizers.Adam(learning_rate,beta_1= 0.5)
    elif opt_type == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate)
    else:
        raise Exception('Optimizer ' + opt_type + ' not supported')


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

def plot_sf(P, X, Y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim= None, tex=False):
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
    cmap = 'coolwarm'
    if f is None:
        f = ''
    if P.ndim < 2:
        P = P.reshape(X.shape)
    # P = P / np.max(abs(P))
    x = X.flatten()
    y = Y.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * x.ptp() / P.size
    dy = 0.5 * y.ptp() / P.size
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(np.real(P), cmap=cmap, origin='lower',
                   extent=[x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy])
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    # lm1, lm2 = clim
    # im.set_clim(lm1, lm2)
    cbar = plt.colorbar(im)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Normalised SPL [dB]', rotation=270)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name + ' - f : {} Hz'.format(f))
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax

def stack_real_imag_H(mat):

    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack
def stack_reference_measurements(measurements, reference_measurements, grid, grid_ref, radius = 0.5):
    rng = np.random.RandomState(1234)
    npoints = 5
    x_ref, y_ref, z_ref = grid_ref
    # number of interior points for zero-cross of bessel functions
    mask = np.argwhere(x_ref.ravel() ** 2 + y_ref.ravel() ** 2 <= radius ** 2)
    interp_ind = rng.choice(mask.shape[0], size=npoints, replace=False)
    interp_ind = np.squeeze(mask[interp_ind])
    grid = np.concatenate((grid, grid_ref[:, interp_ind]), axis=-1)
    measurements = np.concatenate((measurements, reference_measurements[interp_ind]))
    return grid, measurements

def get_measurement_vector(data_dir, dataset = 'ISM_distributed', add_ref_measurements = True):
    import os
    filepath = os.path.join(data_dir, dataset + '.npz')
    with np.load(filepath) as data:
        rirs_measured = data['array_data']
        rirs_reference = data['reference_data']
        grid_measured = data['grids_sphere']
        grid_reference = data['grid_reference']
    if rirs_measured.ndim > 2:
        rirs_measured = np.squeeze(rirs_measured)
        rirs_reference= np.squeeze(rirs_reference)
        grid_measured= np.squeeze(grid_measured)
        grid_reference= np.squeeze(grid_reference)
    if add_ref_measurements:
        grid_measured, rirs_measured = stack_reference_measurements(rirs_measured,
                                                                    rirs_reference,
                                                                    grid_measured,
                                                                    grid_reference,
                                                                    radius = 0.5)
    return rirs_measured, rirs_reference, grid_measured, grid_reference

def mask_measurements(pmeasured, grid, subsample_ratio = 0.1, frequency = 250,
                      decimate = True,
                      seed = None,
                      fs = 16000):
    rng = np.random.RandomState(seed)
    pmeasured /= np.max(abs(pmeasured))
    Nfft = pmeasured.shape[-1]
    Pmeasured = np.fft.rfft(pmeasured, n = Nfft, axis = -1)
    freq = np.fft.rfftfreq( n = Nfft, d = 1/fs)
    ind = np.argmax(freq > frequency )
    Pm = Pmeasured[:, ind]
    if decimate:
        subsample_ind = rng.randint(low = 0, high = len(Pm), size = int(subsample_ratio*len(Pm)))
        Pm = Pm[subsample_ind]
        gridm = grid[:, subsample_ind]
    else:
        gridm = grid
    return Pm, gridm

def nmse(y_true, y_predicted, db = True):
    M = len(y_true)
    nmse_ = 1/M * np.sum(abs(y_true - y_predicted)**2)/np.sum(abs(y_true))**2
    if db:
        nmse_ = np.log10(nmse_)
    return nmse_

def nmse2(y_meas, y_predicted, axis=(-1,), db = True):
    y_meas = y_meas.ravel()
    y_predicted = y_predicted.ravel()
    M = 1
    for i in axis:
        # number of items
        M *= y_meas.shape[i]
    num = (1/ M)*np.sum(np.linalg.norm(y_meas - y_predicted, 2, axis)**2)
    denom = np.linalg.norm(y_meas, 2, axis)**2
    nmse = num/denom
    if db:
        nmse = 10*np.log10(nmse)

    return nmse

def get_Gram_mat(H,  plot = False):
    Gram = tf.linalg.adjoint(H)@H
    column_norm = tf.math.sqrt(tf.reduce_sum(Gram, axis = -1))

    if plot:
        fig, ax = plt.subplots(1,1)
        sc = ax.imshow(abs(Gram/column_norm))
        fig.colorbar(sc)
        fig.show()

def norm_of_columns(A, p=2):
    """Vector p-norm of each column of a matrix.
    Parameters
    ----------
    A : array_like
        Input matrix.
    p : int, optional
        p-th norm.
    Returns
    -------
    array_like
        p-norm of each column of A.
    """
    _, N = A.shape
    return np.asarray([np.linalg.norm(A[:, j], ord=p) for j in range(N)])

def MutualCoherence(A):
    """Mutual coherence of columns of A.

    Parameters
    ----------
    A : array_like
        Input matrix.
    p : int, optional
        p-th norm.

    Returns
    -------
    array_like
        Mutual coherence of columns of A.
    """
    A = np.asmatrix(A)
    _, N = A.shape
    A = A * np.asmatrix(np.diag(1/norm_of_columns(A)))
    Gram_A = A.H*A
    for j in range(N):
        Gram_A[j, j] = 0
    return np.max(np.abs(Gram_A)), Gram_A

def columnwise_coherence(A, k_vec, plot = False, ax = None):
    A = np.asmatrix(A)
    _, N = A.shape

    A = A * np.asmatrix(np.diag(1/norm_of_columns(A)))
    Gram_A = abs(A.H*A)
    col = np.asarray(Gram_A[:,0])[:,0]
    coh = []
    for j in range(N):
        coh.append(np.corrcoef(col, np.asarray(Gram_A[:,j])[:,0])[0,1])

    if plot:
        cart_k = np.squeeze(k_vec)
        k_theta, k_phi, k = cart2sph(cart_k[0], cart_k[1], cart_k[2])
        ax, sc = scatter_pol_plot(k_theta, k_phi, coh, ax = ax )
        return coh, ax, sc
    else:
        return coh

def babel_mat(A):
    A = np.asmatrix(A)
    _, N = A.shape
    A = A * np.asmatrix(np.diag(1/norm_of_columns(A)))
    Gram_A = A.H*A
    for j in range(N):
        Gram_A[j, j] = 0

    Gram_A = np.sort(abs(Gram_A), axis=1)  # sort rows
    Gram_A = Gram_A[:, ::-1]  # in descending order
    Gram_A = Gram_A[:, 1:]  # skip the first column of 1s (diagonal elements)
    Gram_A = Gram_A.cumsum(axis=1)  # cumsum rows
    mu1 = Gram_A.max(axis=0)

    return mu1, Gram_A



def reference_grid(steps, xmin = -.7, xmax = .7, z = 0):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    Z = z*np.ones(X.shape)
    return X,Y,Z

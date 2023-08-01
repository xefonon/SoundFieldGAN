import tensorflow as tf
from tensorflow.keras.models import load_model  # noqa
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow_probability as tfp
from sklearn import linear_model

sys.path.append('./')
from tqdm import trange
from src.CSGAN_auxfun import get_optimizer, plot_array_pressure, \
    nmse, cos_sim, tf_sensing_mat
from src.algorithms import CSGAN, infer_sf, planewaveGAN
import os

# set seeds for reproduceability
seed = 4321
tf.random.set_seed(seed)
np.random.seed(seed)
def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def nmse(y_true, y_predicted, db=True):
    nmse_ = np.mean(abs(y_true - y_predicted) ** 2) / np.mean(abs(y_true) ** 2)
    if db:
        nmse_ = 10 * np.log10(nmse_)
    return nmse_


def mac_similarity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def linear_regression_rescale(x_hat, x):
    # rescale x_hat to the scale of x, using linear regression and scipy stats
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, x_hat)
    x_hat = slope * x_hat + intercept

    return x_hat


def normalize_pressure(p, normalization='l2norm', epsilon=1e-8):
    assert p.ndim == 1
    if normalization == 'maxabs':
        return p / np.max(abs(p)), np.max(abs(p))
    if normalization == 'l2norm':
        return p / np.linalg.norm(p), np.linalg.norm(p)
    if normalization == 'standardization':
        mu = p.mean()
        var = p.var()
        pnorm = (p - mu) / np.sqrt(var + epsilon)
        return pnorm, mu, var


def get_measurement_vector(path):
    data_ = np.load(path)
    keys = [k for k in data_.keys()]
    data_dict = {}
    grid_mic = np.float32(data_['mic_pos']).T
    grid_source = data_['src_pos'].T
    f = np.float32(data_['freq_vec'])
    responses = data_[keys[-1]]
    data_dict['grid_mic'] = grid_mic
    data_dict['grid_source'] = grid_source
    data_dict['f'] = f
    data_dict['responses'] = responses
    return data_dict

def get_spherical_measurement_vector(path):
    data_ = np.load(path)
    data_dict = {}
    grid_mic = np.float32(data_['grids_sphere']).T
    grid_reference = np.float32(data_['grid_reference']).T
    rirs_sphere = data_['array_data']
    rirs_reference = data_['reference_data']
    f = np.fft.rfftfreq(rirs_sphere.shape[-1], d=1/16000)
    data_dict['grid_mic'] = grid_mic
    data_dict['grid_reference'] = grid_reference
    data_dict['f'] = f
    data_dict['rirs_reference'] = rirs_reference
    data_dict['rirs_sphere'] = rirs_sphere
    return data_dict
def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.
    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-2, 1e-8, 50),
                                   fit_intercept=True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=True)

    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk


def get_measurement_vector_manuel(path):
    data_ = np.load(path)
    data_dict = {}
    grid_mic = np.float32(data_['grid']).T
    f = np.float32(data_['freq'])
    responses = data_['responses'].T
    responses = np.expand_dims(responses, 0)
    data_dict['grid_mic'] = grid_mic
    data_dict['f'] = f
    data_dict['responses'] = responses
    return data_dict


def get_mask(shape_, subsample_ratio=0.2, seed=None, flatten=True):
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


def fib_sphere(num_points, radius=1):
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
    return [x_batch, y_batch, z_batch]


def wavenumber(f, n_PW, c=343, two_dim=True):
    k = 2 * np.pi * f / c
    k_grid = fib_sphere(n_PW, k)
    return k_grid


def build_sensing_mat(k_sampled, sensor_grid):
    kx, ky, kz = k_sampled
    X, Y, Z = sensor_grid
    H = np.exp(-1j * (np.einsum('ij,k -> ijk', kx, X) + np.einsum('ij,k -> ijk', ky, Y) +
                      np.einsum('ij,k -> ijk', kz, Z)))
    return H.squeeze(0).T


def plane_wave_sensing_matrix(f, sensor_grid, n_pw=1600, k_samp=None, c=343., normalise=False,
                              two_dim=False):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c,
                            two_dim=two_dim)
    kx, ky, kz = k_samp
    if np.ndim(kx) < 2:
        kx = np.expand_dims(kx, 0)
        ky = np.expand_dims(ky, 0)
        kz = np.expand_dims(kz, 0)
    elif np.ndim(kx) > 2:
        kx = np.squeeze(kx, 1)
        ky = np.squeeze(ky, 1)
        kz = np.squeeze(kz, 1)
    k_out = [kx, ky, kz]
    H = build_sensing_mat(k_out, sensor_grid)
    if normalise:
        column_norm = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / column_norm
    return H, k_out


def get_latent_vector(dim=128):
    return tf.random.normal([1, dim])


def get_latent_vector_batch(batch_size=1, dim=128):
    return tf.random.normal([batch_size, dim])


def array_to_complex(arr):
    cmplx = tf.complex(arr[..., 0], arr[..., 1])
    return cmplx


def generate_random_pressure_fields(frq, Nfields=5, model_direc='./Generator_model', plot=True):
    pressure = []
    f_vec = np.fft.rfftfreq(16384, 1 / 16000)
    f_ind = np.argmin(f_vec < frq)
    f = f_vec[f_ind]
    x, y = np.meshgrid(np.linspace(-.5, .5, 30), np.linspace(-0.5, 0.5, 30))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    grid = np.stack([x, y, z], axis=1).T
    # change grid type to np.float32
    grid = np.float32(grid)
    G = load_model(model_direc)

    H, _ = tf_sensing_mat(f, G.output.shape[1], grid)

    if plot:
        fig, ax = plt.subplots(2, Nfields, figsize=(11, 3.5))
    for i in range(Nfields):
        if len(G.inputs) == 2:
            coeffs = G([get_latent_vector(), tf.constant(f, shape=[1, 1])])
        else:
            coeffs = G(get_latent_vector())
        coeffs = array_to_complex(coeffs)
        fake_sound_fields = tf.einsum('ijk, ik -> ij', H, coeffs)
        pressure.append(fake_sound_fields.numpy().squeeze(0))
        if plot:
            ax[0, i], sc = plot_array_pressure(fake_sound_fields.numpy().squeeze(0), grid, ax=ax[0, i])
            # ax[0, i].set_title('Pressure field, f = {:.1} Hz'.format(f))
            ax[0, i].set_xlabel('x [m]')
            ax[0, i].set_ylabel('y [m]')
            ax[0, i].set_aspect('equal')
            ax[1, i].stem(abs(coeffs.numpy().squeeze(0)))
    if plot:
        # make space for colorbar
        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(sc, ax=ax[0, i])
        # add colorbar label
        cbar.set_label('Pressure [Pa]', rotation=270, labelpad=15)
        fig.suptitle('Pressure fields and magnitude of coefficients for f = {:.1f} Hz'.format(f))
        fig.show()
    return pressure, grid


def ridge_freq_inference(freqs, datadict, settings, subsample_ratio=0.2, src_indx=28, plot_mic=False):
    """Ridge regression for pressure reconstruction."""
    pinference = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    responses = datadict['responses'][src_indx]
    responses_td = np.fft.irfft(responses)
    responses = np.fft.rfft(responses_td)
    ptrue_planar_shape = tf.zeros((21, 21, 1)).shape
    mask = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234)
    nmics = np.count_nonzero(mask.numpy())
    unmasked_indices = np.argwhere(mask != 0.)
    c = settings['c']
    grid_measured = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
    grid = datadict['grid_mic']
    print("\nNumber of mics used as data: ", nmics)
    if plot_mic:
        # plot the mics used for inference as 'pixels'
        mask_plot = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234, flatten=False)
        mask_plot = mask_plot.numpy().squeeze(-1).squeeze(0)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(mask_plot, cmap='gray')
        # add marker to every mic (where pixel == 1) in image
        for i in range(mask_plot.shape[0]):
            for j in range(mask_plot.shape[1]):
                if mask_plot[i, j] == 1:
                    ax.scatter(j, i, c='r', s=20, marker = 'x')

        # set x and y tick labels to be like the grid
        ticks = np.arange(0, 21, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        xticklabels = np.linspace(grid[0].min(), grid[0].max(), 21).round(2)
        yticklabels = np.linspace(grid[1].min(), grid[1].max(), 21).round(2)
        # make strings
        xticklabels = np.array([str(x) for x in xticklabels])
        yticklabels = np.array([str(y) for y in yticklabels])
        # set every other tick label to be blank
        xticklabels[1::2] = ''
        yticklabels[1::2] = ''
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Measurement aperture')
        fig.show()

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        H, k = plane_wave_sensing_matrix(f_rec, n_pw=4000,
                                         sensor_grid=grid_measured,
                                         c=c, normalise=False)

        Href, _ = plane_wave_sensing_matrix(f_rec,
                                            k_samp=k,
                                            n_pw=4000,
                                            sensor_grid=grid,
                                            c=c, normalise=False)

        ptrue = responses[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = ptrue * mask.numpy()
        pm = pm[unmasked_indices].squeeze(-1)
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)

        coeffs, _ = Ridge_regression(H,pm)

        p_pred = np.einsum('jk, k -> j', Href, coeffs)
        if settings["normalize"]:
            nmse_ = nmse(ptrue/norm, p_pred)
            mac_ = cos_sim(ptrue/norm, p_pred)
            pinference.append(p_pred * norm)
            ptrues.append(ptrue)

        else:
            nmse_ = nmse(ptrue, p_pred)
            mac_ = cos_sim(ptrue, p_pred)
            pinference.append(p_pred)
            ptrues.append(ptrue)

        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_), refresh=True)

    return np.array(pinference), np.array(ptrues), nmics, mask.numpy()
def ridge_spherical_freq_inference(freqs, datadict, settings):
    """Ridge regression for pressure reconstruction."""
    pinference = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    reference_responses = datadict['rirs_reference']
    spherical_responses = datadict['rirs_sphere']
    spherical_responses_fd = np.fft.rfft(spherical_responses)
    reference_responses_fd = np.fft.rfft(reference_responses)
    c = settings['c']
    grid_measured = datadict['grid_mic'].T
    grid = datadict['grid_reference'].T
    temp_grid = grid_measured - grid_measured.mean(axis=0)
    # get radius of the sphere
    r = np.sqrt(temp_grid[0]**2 + temp_grid[1]**2).max()

    # add 5 random mics within the radius of the spherical grid (grid_measured) from grid to grid_measured
    # use rng to make sure the same mics are added to the grid_measured
    rng = np.random.default_rng(42)
    indices = np.argwhere(grid[0]**2 + grid[1]**2 <= (r*0.9)**2)
    indices = rng.choice(indices.squeeze(), 5, replace=False)
    grid_measured = np.concatenate((grid_measured, grid[:, indices]), axis=-1)
    spherical_responses_fd = np.concatenate((spherical_responses_fd, reference_responses_fd[indices]), axis=0)
    nmics = spherical_responses_fd.shape[0]
    print("\nNumber of mics: ", nmics)
    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        H, k = plane_wave_sensing_matrix(f_rec, n_pw=4000,
                                         sensor_grid=grid_measured,
                                         c=c, normalise=False)

        Href, _ = plane_wave_sensing_matrix(f_rec,
                                            k_samp=k,
                                            n_pw=4000,
                                            sensor_grid=grid,
                                            c=c, normalise=False)

        ptrue = reference_responses_fd[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = spherical_responses_fd[:, f_indx]
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)
        # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
        coeffs, _ = Ridge_regression(H,pm)

        p_pred = np.einsum('jk, k -> j', Href, coeffs)
        if settings["normalize"]:
            nmse_ = nmse(ptrue/norm, p_pred)
            mac_ = cos_sim(ptrue/norm, p_pred)
            pinference.append(p_pred * norm)
            ptrues.append(ptrue)

        else:
            nmse_ = nmse(ptrue, p_pred)
            mac_ = cos_sim(ptrue, p_pred)
            pinference.append(p_pred)
            ptrues.append(ptrue)

        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))

    return np.array(pinference), np.array(ptrues), nmics, grid

def spherical_array_freq_inference(freqs, datadict, settings, model_direc = './Generator_model'):
    pinference_1 = []
    pinference_2 = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    reference_responses = datadict['rirs_reference']
    spherical_responses = datadict['rirs_sphere']
    spherical_responses_fd = np.fft.rfft(spherical_responses)
    reference_responses_fd = np.fft.rfft(reference_responses)
    c = settings['c']
    grid = datadict['grid_reference'].T
    grid_measured = datadict['grid_mic'].T
    grid = datadict['grid_reference'].T
    temp_grid = grid_measured - grid_measured.mean(axis=0)
    # get radius of the sphere
    r = np.sqrt(temp_grid[0]**2 + temp_grid[1]**2).max()

    # add 5 random mics within the radius of the spherical grid (grid_measured) from grid to grid_measured
    # use rng to make sure the same mics are added to the grid_measured
    rng = np.random.default_rng(42)
    indices = np.argwhere(grid[0]**2 + grid[1]**2 <= (r*0.9)**2)
    indices = rng.choice(indices.squeeze(), 5, replace=False)
    grid_measured = np.concatenate((grid_measured, grid[:, indices]), axis=-1)
    spherical_responses_fd = np.concatenate((spherical_responses_fd, reference_responses_fd[indices]), axis=0)
    nmics = spherical_responses_fd.shape[0]
    print("\nNumber of mics: ", nmics)

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        ptrue_norm = reference_responses_fd[:, f_indx]
        pm = spherical_responses_fd[:, f_indx]
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)
        else:
            norm = 1
        pm = tf.convert_to_tensor(pm.reshape(-1, 1))
        if pm.dtype == tf.complex128:
            pm = tf.cast(pm, tf.complex64)
        G = load_model(model_direc)

        # z optimizer paramaters
        # decay_steps = settings['adapt_gan_iters'] * .1
        # decay_rate = 0.96
        # learning_rate_fn1 = tf.keras.optimizers.schedules.ExponentialDecay(
        #     settings["csganlr"], decay_steps, decay_rate)
        # optz = get_optimizer(learning_rate_fn1, 'adam')
        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_measured, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, grid, sparsegen=False)
        # x optimizer paramaters
        decay_steps = settings['adapt_gan_iters'] * .2
        decay_rate = 0.9
        learning_rate_fn2 = tf.keras.optimizers.schedules.ExponentialDecay(
            settings["adapt_lr"], decay_steps, decay_rate)
        opt_x = get_optimizer(learning_rate_fn2, 'adam')
        # opt_x = get_optimizer(settings["adapt_lr"], 'adam')

        if settings["optimize_weights"]:
            opt_w = get_optimizer(0.0001, 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_measured, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True,
                                use_MAC_loss=True,
                                pref = ptrue_norm/ norm,
                                grid_ref= grid)

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, grid, ptych=True)

        if settings["normalize"]:
            nmse_ = nmse(ptrue_norm / norm, Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm / norm, Ppred3.numpy())
            pinference_1.append(norm * Ppred.numpy())
            pinference_2.append(norm * Ppred3.numpy())
        else:
            nmse_ = nmse(ptrue_norm , Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm , Ppred3.numpy())
            pinference_1.append(Ppred.numpy())
            pinference_2.append(Ppred3.numpy())
        ptrues.append(ptrue_norm)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))
    return np.array(pinference_1), np.array(pinference_2), np.array(ptrues), nmics, grid

def frequency_inference(freqs, datadict, settings, subsample_ratio=0.2,
                        model_direc='./Generator_model', src_indx=28):
    # nmses = []
    # macs = []
    pinference_1 = []
    pinference_2 = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    responses = datadict['responses'][src_indx]
    # responses /= np.max(abs(responses))
    responses_td = np.fft.irfft(responses)
    # responses_td /= np.max(abs(responses_td))
    responses = np.fft.rfft(responses_td)
    ptrue_planar_shape = tf.zeros((21, 21, 1)).shape
    # ptrue_planar_shape = tf.zeros((69, 69, 1)).shape
    mask = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234)
    nmics = np.count_nonzero(mask.numpy())

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        ptrue_norm = responses[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = ptrue_norm * mask.numpy()
        unmasked_indices = np.argwhere(pm != 0.)
        pm = pm[unmasked_indices]
        grid_meas = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm.squeeze(-1))
        pm = tf.convert_to_tensor(pm.reshape(-1, 1))
        if pm.dtype == tf.complex128:
            pm = tf.cast(pm, tf.complex64)
        G = load_model(model_direc)

        # z optimizer paramaters
        # decay_steps = settings['adapt_gan_iters'] * .1
        # decay_rate = 0.96
        # learning_rate_fn1 = tf.keras.optimizers.schedules.ExponentialDecay(
        #     settings["csganlr"], decay_steps, decay_rate)
        # optz = get_optimizer(learning_rate_fn1, 'adam')
        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_meas, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, datadict['grid_mic'], sparsegen=False)
        # x optimizer paramaters
        decay_steps = settings['adapt_gan_iters'] * .2
        decay_rate = 0.9
        learning_rate_fn2 = tf.keras.optimizers.schedules.ExponentialDecay(
            settings["adapt_lr"], decay_steps, decay_rate)
        opt_x = get_optimizer(learning_rate_fn2, 'adam')
        # opt_x = get_optimizer(settings["adapt_lr"], 'adam')

        if settings["optimize_weights"]:
            opt_w = get_optimizer(0.0001, 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_meas, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True,
                                use_MAC_loss=True,
                                pref=ptrue_norm / norm,
                                grid_ref=datadict['grid_mic'])

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, datadict['grid_mic'], ptych=True)

        if settings["normalize"]:
            nmse_ = nmse(ptrue_norm / norm, Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm / norm, Ppred3.numpy())
            pinference_1.append(norm * Ppred.numpy())
            pinference_2.append(norm * Ppred3.numpy())
        else:
            nmse_ = nmse(ptrue_norm , Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm , Ppred3.numpy())
            pinference_1.append(Ppred.numpy())
            pinference_2.append(Ppred3.numpy())
        ptrues.append(ptrue_norm)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))
    return np.array(pinference_1), np.array(pinference_2), np.array(ptrues), nmics, mask.numpy()

def measurements_inference(ratios, datadict, settings, frequency=250.):
    nmses = []
    macs = []
    pinference = []
    ptrues = []
    t = trange(len(ratios), desc='Reconstructing...', leave=True)
    nmics_list = []
    for i in t:
        r = ratios[i]
        f_indx = np.argmin(datadict['f'] < frequency)
        ptrue = datadict['responses'][0, :, f_indx]
        f_rec = datadict['f'][f_indx]
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar_shape = tf.reshape(ptrue, (21, 21, 1)).shape
        mask = get_mask(ptrue_planar_shape, subsample_ratio=r, seed=1234)

        pm = ptrue * mask.numpy()
        unmasked_indices = np.argwhere(pm != 0.)
        nmics = np.count_nonzero(mask[..., 0].numpy())
        pm = pm[unmasked_indices]
        # normalize the pressure
        pm, norm = normalize_pressure(pm, normalization='l2norm')
        grid_meas = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
        pm = tf.convert_to_tensor(pm)
        G = load_model(model_direc)

        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_meas, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, grid_meas, sparsegen=False)

        opt_x = get_optimizer(settings["adapt_lr"], 'adam')
        if settings["optimize_weights"]:
            opt_w = get_optimizer(settings['weights_lr'], 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_meas, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True)

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, datadict['grid_mic'], ptych=True)

        nmse_ = nmse(ptrue, norm * Ppred3.numpy())
        mac_ = mac_similarity(ptrue, norm * Ppred3.numpy())
        nmses.append(nmse_)
        nmics_list.append(nmics)
        macs.append(mac_)
        pinference.append(norm * Ppred3.numpy())
        ptrues.append(ptrue)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(frequency, mac_, nmse_))
    return np.array(pinference), np.array(ptrues), np.array(nmses), np.array(macs), np.array(nmics_list)

def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def nmse(y_true, y_predicted, db=True):
    nmse_ = np.mean(abs(y_true - y_predicted) ** 2) / np.mean(abs(y_true) ** 2)
    if db:
        nmse_ = 10 * np.log10(nmse_)
    return nmse_


def mac_similarity(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def linear_regression_rescale(x_hat, x):
    # rescale x_hat to the scale of x, using linear regression and scipy stats
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, x_hat)
    x_hat = slope * x_hat + intercept

    return x_hat


def normalize_pressure(p, normalization='l2norm', epsilon=1e-8):
    assert p.ndim == 1
    if normalization == 'maxabs':
        return p / np.max(abs(p)), np.max(abs(p))
    if normalization == 'l2norm':
        return p / np.linalg.norm(p), np.linalg.norm(p)
    if normalization == 'standardization':
        mu = p.mean()
        var = p.var()
        pnorm = (p - mu) / np.sqrt(var + epsilon)
        return pnorm, mu, var


def get_measurement_vector(path):
    data_ = np.load(path)
    keys = [k for k in data_.keys()]
    data_dict = {}
    grid_mic = np.float32(data_['mic_pos']).T
    grid_source = data_['src_pos'].T
    f = np.float32(data_['freq_vec'])
    responses = data_[keys[-1]]
    data_dict['grid_mic'] = grid_mic
    data_dict['grid_source'] = grid_source
    data_dict['f'] = f
    data_dict['responses'] = responses
    return data_dict

def get_spherical_measurement_vector(path):
    data_ = np.load(path)
    data_dict = {}
    grid_mic = np.float32(data_['grids_sphere']).T
    grid_reference = np.float32(data_['grid_reference']).T
    rirs_sphere = data_['array_data']
    rirs_reference = data_['reference_data']
    f = np.fft.rfftfreq(rirs_sphere.shape[-1], d=1/16000)
    data_dict['grid_mic'] = grid_mic
    data_dict['grid_reference'] = grid_reference
    data_dict['f'] = f
    data_dict['rirs_reference'] = rirs_reference
    data_dict['rirs_sphere'] = rirs_sphere
    return data_dict
def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.
    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-2, 1e-8, 50),
                                   fit_intercept=True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=True)

    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk


def get_measurement_vector_manuel(path):
    data_ = np.load(path)
    data_dict = {}
    grid_mic = np.float32(data_['grid']).T
    f = np.float32(data_['freq'])
    responses = data_['responses'].T
    responses = np.expand_dims(responses, 0)
    data_dict['grid_mic'] = grid_mic
    data_dict['f'] = f
    data_dict['responses'] = responses
    return data_dict


def get_mask(shape_, subsample_ratio=0.2, seed=None, flatten=True):
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


def fib_sphere(num_points, radius=1):
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
    return [x_batch, y_batch, z_batch]


def wavenumber(f, n_PW, c=343, two_dim=True):
    k = 2 * np.pi * f / c
    k_grid = fib_sphere(n_PW, k)
    return k_grid


def build_sensing_mat(k_sampled, sensor_grid):
    kx, ky, kz = k_sampled
    X, Y, Z = sensor_grid
    H = np.exp(-1j * (np.einsum('ij,k -> ijk', kx, X) + np.einsum('ij,k -> ijk', ky, Y) +
                      np.einsum('ij,k -> ijk', kz, Z)))
    return H.squeeze(0).T


def plane_wave_sensing_matrix(f, sensor_grid, n_pw=1600, k_samp=None, c=343., normalise=False,
                              two_dim=False):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c,
                            two_dim=two_dim)
    kx, ky, kz = k_samp
    if np.ndim(kx) < 2:
        kx = np.expand_dims(kx, 0)
        ky = np.expand_dims(ky, 0)
        kz = np.expand_dims(kz, 0)
    elif np.ndim(kx) > 2:
        kx = np.squeeze(kx, 1)
        ky = np.squeeze(ky, 1)
        kz = np.squeeze(kz, 1)
    k_out = [kx, ky, kz]
    H = build_sensing_mat(k_out, sensor_grid)
    if normalise:
        column_norm = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / column_norm
    return H, k_out


def get_latent_vector(dim=128):
    return tf.random.normal([1, dim])


def get_latent_vector_batch(batch_size=1, dim=128):
    return tf.random.normal([batch_size, dim])


def array_to_complex(arr):
    cmplx = tf.complex(arr[..., 0], arr[..., 1])
    return cmplx


def generate_random_pressure_fields(frq, Nfields=5, model_direc='./Generator_model', plot=True):
    pressure = []
    f_vec = np.fft.rfftfreq(16384, 1 / 16000)
    f_ind = np.argmin(f_vec < frq)
    f = f_vec[f_ind]
    x, y = np.meshgrid(np.linspace(-.5, .5, 30), np.linspace(-0.5, 0.5, 30))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    grid = np.stack([x, y, z], axis=1).T
    # change grid type to np.float32
    grid = np.float32(grid)
    G = load_model(model_direc)

    H, _ = tf_sensing_mat(f, G.output.shape[1], grid)

    if plot:
        fig, ax = plt.subplots(2, Nfields, figsize=(11, 3.5))
    for i in range(Nfields):
        if len(G.inputs) == 2:
            coeffs = G([get_latent_vector(), tf.constant(f, shape=[1, 1])])
        else:
            coeffs = G(get_latent_vector())
        coeffs = array_to_complex(coeffs)
        fake_sound_fields = tf.einsum('ijk, ik -> ij', H, coeffs)
        pressure.append(fake_sound_fields.numpy().squeeze(0))
        if plot:
            ax[0, i], sc = plot_array_pressure(fake_sound_fields.numpy().squeeze(0), grid, ax=ax[0, i])
            # ax[0, i].set_title('Pressure field, f = {:.1} Hz'.format(f))
            ax[0, i].set_xlabel('x [m]')
            ax[0, i].set_ylabel('y [m]')
            ax[0, i].set_aspect('equal')
            ax[1, i].stem(abs(coeffs.numpy().squeeze(0)))
    if plot:
        # make space for colorbar
        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(sc, ax=ax[0, i])
        # add colorbar label
        cbar.set_label('Pressure [Pa]', rotation=270, labelpad=15)
        fig.suptitle('Pressure fields and magnitude of coefficients for f = {:.1f} Hz'.format(f))
        fig.show()
    return pressure, grid


def ridge_freq_inference(freqs, datadict, settings, subsample_ratio=0.2, src_indx=28, plot_mic=False):
    """Ridge regression for pressure reconstruction."""
    pinference = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    responses = datadict['responses'][src_indx]
    responses_td = np.fft.irfft(responses)
    responses = np.fft.rfft(responses_td)
    ptrue_planar_shape = tf.zeros((21, 21, 1)).shape
    mask = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234)
    nmics = np.count_nonzero(mask.numpy())
    unmasked_indices = np.argwhere(mask != 0.)
    c = settings['c']
    grid_measured = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
    grid = datadict['grid_mic']
    print("\nNumber of mics used as data: ", nmics)
    if plot_mic:
        # plot the mics used for inference as 'pixels'
        mask_plot = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234, flatten=False)
        mask_plot = mask_plot.numpy().squeeze(-1).squeeze(0)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(mask_plot, cmap='gray')
        # add marker to every mic (where pixel == 1) in image
        for i in range(mask_plot.shape[0]):
            for j in range(mask_plot.shape[1]):
                if mask_plot[i, j] == 1:
                    ax.scatter(j, i, c='r', s=20, marker = 'x')

        # set x and y tick labels to be like the grid
        ticks = np.arange(0, 21, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        xticklabels = np.linspace(grid[0].min(), grid[0].max(), 21).round(2)
        yticklabels = np.linspace(grid[1].min(), grid[1].max(), 21).round(2)
        # make strings
        xticklabels = np.array([str(x) for x in xticklabels])
        yticklabels = np.array([str(y) for y in yticklabels])
        # set every other tick label to be blank
        xticklabels[1::2] = ''
        yticklabels[1::2] = ''
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Measurement aperture')
        fig.show()

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        H, k = plane_wave_sensing_matrix(f_rec, n_pw=4000,
                                         sensor_grid=grid_measured,
                                         c=c, normalise=False)

        Href, _ = plane_wave_sensing_matrix(f_rec,
                                            k_samp=k,
                                            n_pw=4000,
                                            sensor_grid=grid,
                                            c=c, normalise=False)

        ptrue = responses[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = ptrue * mask.numpy()
        pm = pm[unmasked_indices].squeeze(-1)
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)

        coeffs, _ = Ridge_regression(H,pm)

        p_pred = np.einsum('jk, k -> j', Href, coeffs)
        if settings["normalize"]:
            nmse_ = nmse(ptrue/norm, p_pred)
            mac_ = cos_sim(ptrue/norm, p_pred)
            pinference.append(p_pred * norm)
            ptrues.append(ptrue)

        else:
            nmse_ = nmse(ptrue, p_pred)
            mac_ = cos_sim(ptrue, p_pred)
            pinference.append(p_pred)
            ptrues.append(ptrue)

        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_), refresh=True)

    return np.array(pinference), np.array(ptrues), nmics, mask.numpy()
def ridge_spherical_freq_inference(freqs, datadict, settings):
    """Ridge regression for pressure reconstruction."""
    pinference = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    reference_responses = datadict['rirs_reference']
    spherical_responses = datadict['rirs_sphere']
    spherical_responses_fd = np.fft.rfft(spherical_responses)
    reference_responses_fd = np.fft.rfft(reference_responses)
    c = settings['c']
    grid_measured = datadict['grid_mic'].T
    grid = datadict['grid_reference'].T
    temp_grid = grid_measured - grid_measured.mean(axis=0)
    # get radius of the sphere
    r = np.sqrt(temp_grid[0]**2 + temp_grid[1]**2).max()

    # add 5 random mics within the radius of the spherical grid (grid_measured) from grid to grid_measured
    # use rng to make sure the same mics are added to the grid_measured
    rng = np.random.default_rng(42)
    indices = np.argwhere(grid[0]**2 + grid[1]**2 <= (r*0.9)**2)
    indices = rng.choice(indices.squeeze(), 5, replace=False)
    grid_measured = np.concatenate((grid_measured, grid[:, indices]), axis=-1)
    spherical_responses_fd = np.concatenate((spherical_responses_fd, reference_responses_fd[indices]), axis=0)
    nmics = spherical_responses_fd.shape[0]
    print("\nNumber of mics: ", nmics)
    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        H, k = plane_wave_sensing_matrix(f_rec, n_pw=4000,
                                         sensor_grid=grid_measured,
                                         c=c, normalise=False)

        Href, _ = plane_wave_sensing_matrix(f_rec,
                                            k_samp=k,
                                            n_pw=4000,
                                            sensor_grid=grid,
                                            c=c, normalise=False)

        ptrue = reference_responses_fd[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = spherical_responses_fd[:, f_indx]
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)
        # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
        coeffs, _ = Ridge_regression(H,pm)

        p_pred = np.einsum('jk, k -> j', Href, coeffs)
        if settings["normalize"]:
            nmse_ = nmse(ptrue/norm, p_pred)
            mac_ = cos_sim(ptrue/norm, p_pred)
            pinference.append(p_pred * norm)
            ptrues.append(ptrue)

        else:
            nmse_ = nmse(ptrue, p_pred)
            mac_ = cos_sim(ptrue, p_pred)
            pinference.append(p_pred)
            ptrues.append(ptrue)

        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))

    return np.array(pinference), np.array(ptrues), nmics, grid

def spherical_array_freq_inference(freqs, datadict, settings, model_direc = './Generator_model'):
    pinference_1 = []
    pinference_2 = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    reference_responses = datadict['rirs_reference']
    spherical_responses = datadict['rirs_sphere']
    spherical_responses_fd = np.fft.rfft(spherical_responses)
    reference_responses_fd = np.fft.rfft(reference_responses)
    c = settings['c']
    grid = datadict['grid_reference'].T
    grid_measured = datadict['grid_mic'].T
    grid = datadict['grid_reference'].T
    temp_grid = grid_measured - grid_measured.mean(axis=0)
    # get radius of the sphere
    r = np.sqrt(temp_grid[0]**2 + temp_grid[1]**2).max()

    # add 5 random mics within the radius of the spherical grid (grid_measured) from grid to grid_measured
    # use rng to make sure the same mics are added to the grid_measured
    rng = np.random.default_rng(42)
    indices = np.argwhere(grid[0]**2 + grid[1]**2 <= (r*0.9)**2)
    indices = rng.choice(indices.squeeze(), 5, replace=False)
    grid_measured = np.concatenate((grid_measured, grid[:, indices]), axis=-1)
    spherical_responses_fd = np.concatenate((spherical_responses_fd, reference_responses_fd[indices]), axis=0)
    nmics = spherical_responses_fd.shape[0]
    print("\nNumber of mics: ", nmics)

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        ptrue_norm = reference_responses_fd[:, f_indx]
        pm = spherical_responses_fd[:, f_indx]
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm)
        else:
            norm = 1
        pm = tf.convert_to_tensor(pm.reshape(-1, 1))
        if pm.dtype == tf.complex128:
            pm = tf.cast(pm, tf.complex64)
        G = load_model(model_direc)

        # z optimizer paramaters
        # decay_steps = settings['adapt_gan_iters'] * .1
        # decay_rate = 0.96
        # learning_rate_fn1 = tf.keras.optimizers.schedules.ExponentialDecay(
        #     settings["csganlr"], decay_steps, decay_rate)
        # optz = get_optimizer(learning_rate_fn1, 'adam')
        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_measured, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, grid, sparsegen=False)
        # x optimizer paramaters
        decay_steps = settings['adapt_gan_iters'] * .2
        decay_rate = 0.9
        learning_rate_fn2 = tf.keras.optimizers.schedules.ExponentialDecay(
            settings["adapt_lr"], decay_steps, decay_rate)
        opt_x = get_optimizer(learning_rate_fn2, 'adam')
        # opt_x = get_optimizer(settings["adapt_lr"], 'adam')

        if settings["optimize_weights"]:
            opt_w = get_optimizer(0.0001, 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_measured, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True,
                                use_MAC_loss=True,
                                pref = ptrue_norm/ norm,
                                grid_ref= grid)

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, grid, ptych=True)

        if settings["normalize"]:
            nmse_ = nmse(ptrue_norm / norm, Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm / norm, Ppred3.numpy())
            pinference_1.append(norm * Ppred.numpy())
            pinference_2.append(norm * Ppred3.numpy())
        else:
            nmse_ = nmse(ptrue_norm , Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm , Ppred3.numpy())
            pinference_1.append(Ppred.numpy())
            pinference_2.append(Ppred3.numpy())
        ptrues.append(ptrue_norm)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))
    return np.array(pinference_1), np.array(pinference_2), np.array(ptrues), nmics, grid

def frequency_inference(freqs, datadict, settings, subsample_ratio=0.2,
                        model_direc='./Generator_model', src_indx=28):
    # nmses = []
    # macs = []
    pinference_1 = []
    pinference_2 = []
    ptrues = []
    t = trange(len(freqs), desc='Reconstructing...', leave=True)
    responses = datadict['responses'][src_indx]
    # responses /= np.max(abs(responses))
    responses_td = np.fft.irfft(responses)
    # responses_td /= np.max(abs(responses_td))
    responses = np.fft.rfft(responses_td)
    ptrue_planar_shape = tf.zeros((21, 21, 1)).shape
    # ptrue_planar_shape = tf.zeros((69, 69, 1)).shape
    mask = get_mask(ptrue_planar_shape, subsample_ratio=subsample_ratio, seed=1234)
    nmics = np.count_nonzero(mask.numpy())

    for i in t:
        f = freqs[i]
        f_indx = np.argmin(datadict['f'] < f)
        f_rec = datadict['f'][f_indx]
        ptrue_norm = responses[:, f_indx]
        # for sake of using exactly the same mask as autoencoder:
        pm = ptrue_norm * mask.numpy()
        unmasked_indices = np.argwhere(pm != 0.)
        pm = pm[unmasked_indices]
        grid_meas = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
        if settings["normalize"]:
            # pm, _ = normalize_pressure(pm.squeeze(-1), normalization = 'maxabs')
            # pm, mu, var = normalize_pressure(pm.squeeze(-1), normalization = 'standardization')
            pm, norm = normalize_pressure(pm.squeeze(-1))
        pm = tf.convert_to_tensor(pm.reshape(-1, 1))
        if pm.dtype == tf.complex128:
            pm = tf.cast(pm, tf.complex64)
        G = load_model(model_direc)

        # z optimizer paramaters
        # decay_steps = settings['adapt_gan_iters'] * .1
        # decay_rate = 0.96
        # learning_rate_fn1 = tf.keras.optimizers.schedules.ExponentialDecay(
        #     settings["csganlr"], decay_steps, decay_rate)
        # optz = get_optimizer(learning_rate_fn1, 'adam')
        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_meas, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, datadict['grid_mic'], sparsegen=False)
        # x optimizer paramaters
        decay_steps = settings['adapt_gan_iters'] * .2
        decay_rate = 0.9
        learning_rate_fn2 = tf.keras.optimizers.schedules.ExponentialDecay(
            settings["adapt_lr"], decay_steps, decay_rate)
        opt_x = get_optimizer(learning_rate_fn2, 'adam')
        # opt_x = get_optimizer(settings["adapt_lr"], 'adam')

        if settings["optimize_weights"]:
            opt_w = get_optimizer(0.0001, 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_meas, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True,
                                use_MAC_loss=True,
                                pref=ptrue_norm / norm,
                                grid_ref=datadict['grid_mic'])

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, datadict['grid_mic'], ptych=True)

        if settings["normalize"]:
            nmse_ = nmse(ptrue_norm / norm, Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm / norm, Ppred3.numpy())
            pinference_1.append(norm * Ppred.numpy())
            pinference_2.append(norm * Ppred3.numpy())
        else:
            nmse_ = nmse(ptrue_norm , Ppred3.numpy())
            mac_ = mac_similarity(ptrue_norm , Ppred3.numpy())
            pinference_1.append(Ppred.numpy())
            pinference_2.append(Ppred3.numpy())
        ptrues.append(ptrue_norm)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(f, mac_, nmse_))
    return np.array(pinference_1), np.array(pinference_2), np.array(ptrues), nmics, mask.numpy()

def measurements_inference(ratios, datadict, settings, frequency=250.):
    nmses = []
    macs = []
    pinference = []
    ptrues = []
    t = trange(len(ratios), desc='Reconstructing...', leave=True)
    nmics_list = []
    for i in t:
        r = ratios[i]
        f_indx = np.argmin(datadict['f'] < frequency)
        ptrue = datadict['responses'][0, :, f_indx]
        f_rec = datadict['f'][f_indx]
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar_shape = tf.reshape(ptrue, (21, 21, 1)).shape
        mask = get_mask(ptrue_planar_shape, subsample_ratio=r, seed=1234)

        pm = ptrue * mask.numpy()
        unmasked_indices = np.argwhere(pm != 0.)
        nmics = np.count_nonzero(mask[..., 0].numpy())
        pm = pm[unmasked_indices]
        # normalize the pressure
        pm, norm = normalize_pressure(pm, normalization='l2norm')
        grid_meas = np.squeeze(datadict['grid_mic'][:, unmasked_indices], axis=-1)
        pm = tf.convert_to_tensor(pm)
        G = load_model(model_direc)

        optz = get_optimizer(settings["csganlr"], 'adam')

        Gvars = CSGAN(G, optz, pm, grid_meas, f_rec, n_resets=settings['csgan_resets'],
                      lambda_z=settings['csgan_lambda'], max_itr=settings['csgan_iters'],
                      complex_net=settings['complex_net'], conditional_net=settings['conditional_net'],
                      verbose=True)
        Ppred, _ = infer_sf(G, Gvars, f_rec, grid_meas, sparsegen=False)

        opt_x = get_optimizer(settings["adapt_lr"], 'adam')
        if settings["optimize_weights"]:
            opt_w = get_optimizer(settings['weights_lr'], 'adam', decay_steps=int(settings['adapt_gan_iters'] * .9))
        else:
            opt_w = None
        # opt_w = None

        G.trainable = True

        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, pm, grid_meas, f_rec, Gvars['Ginputs'][0],
                                lambda_=settings['adapt_gan_lambda'], k_vec=Gvars['wavenumber_vec'],
                                max_itr=settings['adapt_gan_iters'],
                                complex_net=settings['complex_net'],
                                conditional_net=settings['conditional_net'],
                                optimizer_theta=opt_w,
                                verbose=True)

        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, datadict['grid_mic'], ptych=True)

        nmse_ = nmse(ptrue, norm * Ppred3.numpy())
        mac_ = mac_similarity(ptrue, norm * Ppred3.numpy())
        nmses.append(nmse_)
        nmics_list.append(nmics)
        macs.append(mac_)
        pinference.append(norm * Ppred3.numpy())
        ptrues.append(ptrue)
        t.set_description("Frequency: {} Hz MAC: {:.4e} NMSE: {:.4e}".format(frequency, mac_, nmse_))
    return np.array(pinference), np.array(ptrues), np.array(nmses), np.array(macs), np.array(nmics_list)

# %%
# CONSTANTS
results_dir = './Results'
# model = 'Real_FC_GAN_data_augmentation'
# model = 'Ragan_latest'
model = 'Ragan_best'
# model = 'Real_Conv_GAN'
# model = 'Real_Conv'

# """ planar data """
# use pathlib to set the path to the data (/home/xen/PhD/Repos/pwgan/data/Inference files/MeshRIR_set.npz
datapath = './Transfer_learning_data/MeshRIR_set.npz'
# datapath = '../Transfer_learning_data/MeshRIR_set.npz'
planar = True

""" spherical data """
# datapath = './CSGAN/data_folder/real_data.npz'
# planar = False
# datapath = './data_folder/real_data.npz'

settings = {
    "csgan_iters": 1000,
    "csgan_resets": 1,
    "csgan_lambda": 0.001,
    "conditional_net": False,
    "complex_net": False,
    "adapt_gan_iters": 1000 if planar else 3000,
    "adapt_gan_lambda": 5e-2,
    "csganlr": 0.001,
    "weights_lr": 0.0001,
    "adapt_lr": 0.002,
    "c": 343.,
    "optimize_weights": False,
    "normalize": True
}
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_dir = os.path.join(results_dir, model)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_path = './CSGAN/Generator Models'
# model_path = './Generator Models'
model_direc = os.path.join(model_path, model)
model_direc = os.path.join(os.getcwd(), model_direc)

if planar:
    datadict = get_measurement_vector(datapath)
else:
    datadict = get_spherical_measurement_vector(datapath)
ratios = np.linspace(0.05, .25, 25)

fc = [125., 250.,500., 1000., 2000.]
# fc = [500.]

true_fcs = []
fvec = np.fft.rfftfreq(16384, 1 / 16000)
for ffc in fc:
    find = np.argmin(fvec < ffc)
    true_fcs.append(fvec[find])
# %% plot random pressure fields
nfields = 5
pp, _ = generate_random_pressure_fields(fc[2], nfields, model_direc=model_direc, plot=True)
# %%
# ptrue,pinference, name_, mac, Nmics = measurements_inference(ratios, datadict, settings, frequency = 552.)

# %% Ridge regression
if planar:
    p_inference_ridge, ptrue_ridge, Nmics, mask = ridge_freq_inference(true_fcs, datadict, settings,
                                                                       subsample_ratio=0.15, src_indx=7, plot_mic= True)
else:
    p_inference_ridge, ptrue_ridge, Nmics, grid = ridge_spherical_freq_inference(true_fcs, datadict, settings)

# %%
if planar:
    pinference_csgm, ptrue_ptych, ptrue, Nmics, mask = frequency_inference(true_fcs,
                                                                           datadict,
                                                                           settings,
                                                                           model_direc=model_direc,
                                                                           subsample_ratio=0.15,
                                                                           src_indx=7)
else:
    pinference_csgm, ptrue_ptych, ptrue, Nmics, grid = spherical_array_freq_inference(true_fcs,
                                                                                     datadict,
                                                                                     settings,
                                                                                     model_direc)

# %%ptrue_ptych plot sound fields
plt_indx = 3
if planar:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0], _ = plot_array_pressure(ptrue[plt_indx], array_grid=datadict['grid_mic'], ax=ax[0])
    ax[0].set_title('True pressure')
    ax[1], _ = plot_array_pressure(ptrue_ptych[plt_indx], array_grid=datadict['grid_mic'], ax=ax[1])
    ax[1].set_title('PWGAN pressure')
    ax[2], _ = plot_array_pressure(p_inference_ridge[plt_indx], array_grid=datadict['grid_mic'], ax=ax[2])
    ax[2].set_title('Titkhonov pressure')
    fig.show()
else:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0], _ = plot_array_pressure(ptrue[plt_indx], array_grid=grid, ax=ax[0])
    ax[0].set_title('True pressure')
    ax[1], _ = plot_array_pressure(ptrue_ptych[plt_indx], array_grid=grid, ax=ax[1])
    ax[1].set_title('PWGAN pressure')
    ax[2], _ = plot_array_pressure(p_inference_ridge[plt_indx], array_grid=grid, ax=ax[2])
    ax[2].set_title('Titkhonov pressure')
    fig.show()
# %% print errors
# reshape vectors
ptrue_plt = ptrue[plt_indx]
ptrue_ptych_plt = ptrue_ptych[plt_indx]
pinference_csgm_plt = pinference_csgm[plt_indx]
p_inference_ridge_plt = p_inference_ridge[plt_indx]
ptrue_ridge = ptrue_ridge.reshape(-1)
print(f"--CSGM-- \nMAC: {mac_similarity(ptrue, pinference_csgm).round(3)}, NMSE: {nmse(ptrue, pinference_csgm).round(3)}")
print(f"--PWGAN-- \nMAC: {mac_similarity(ptrue, ptrue_ptych).round(3)}, NMSE: {nmse(ptrue, ptrue_ptych).round(3)}")
print(f"--Ridge-- \nMAC: {mac_similarity(ptrue_ridge, p_inference_ridge).round(3)}, NMSE: {nmse(ptrue_ridge, p_inference_ridge).round(3)}")
# %%
# get all nmse and mac and plot
nmse_ridge = []
mac_ridge = []
nmse_csgm = []
mac_csgm = []
nmse_pwgan = []
mac_pwgan = []
for i in range(len(ptrue)):
    nmse_ridge.append(nmse(ptrue_ridge[i], p_inference_ridge[i]))
    mac_ridge.append(mac_similarity(ptrue_ridge[i], p_inference_ridge[i]))
    nmse_csgm.append(nmse(ptrue[i], pinference_csgm[i]))
    mac_csgm.append(mac_similarity(ptrue[i], pinference_csgm[i]))
    nmse_pwgan.append(nmse(ptrue[i], ptrue_ptych[i]))
    mac_pwgan.append(mac_similarity(ptrue[i], ptrue_ptych[i]))
# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(fc,nmse_ridge, label='Ridge')
ax[0].plot(fc,nmse_csgm, label='CSGM')
ax[0].plot(fc,nmse_pwgan, label='PWGAN')
ax[0].set_title('NMSE')
ax[0].legend()
ax[1].plot(fc,mac_ridge, label='Ridge')
ax[1].plot(fc,mac_csgm, label='CSGM')
ax[1].plot(fc,mac_pwgan, label='PWGAN')
ax[1].set_title('MAC')
ax[1].legend()
fig.show()
# %%

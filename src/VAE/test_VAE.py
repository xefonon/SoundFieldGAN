import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from aux_functions import find_nearest, get_centre_freq_octave_bands, get_sensing_mat
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from aux_functions import get_mask

def MAC_loss(p, p_hat, returnMAC=False):
    p = tf.cast(p, p_hat.dtype)

    numer = tf.abs(tf.experimental.numpy.vdot(p, p_hat))
    denom = tf.abs(tf.linalg.norm(p, ord=2) * tf.linalg.norm(p_hat, ord=2))

    MACLoss = tf.divide(numer, denom)
    if returnMAC:
        return MACLoss
    else:
        return (1 - MACLoss)

def tf_normalize_pressure(p, normalization = 'l2norm', epsilon=1e-8):
    # assert p.ndim == 1
    if normalization == 'maxabs':
        return p/tf.reduce_max(tf.abs(p)), tf.reduce_max(abs(p))
    if normalization == 'l2norm':
        return p/tf.linalg.norm(p), tf.linalg.norm(p)
    if normalization == 'standardization':
        mu = p.mean()
        var = p.var()
        pnorm = (p - mu)/np.sqrt(var + epsilon)
        return pnorm, (mu, var)

def plot_array_pressure(p_array, array_grid, ax=None, plane = False, norm = None, z_label = False):
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
                        cmap=cmp, alpha=1., s=20, vmin = vmin, vmax = vmax)
    else:
        sc = ax.scatter(array_grid[0], array_grid[1], c=p_array.real,
                        cmap=cmp, alpha=1., s=20, vmin = vmin, vmax = vmax)
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

def nmse(y_true, y_predicted, db = True):
    M = len(y_true)
    nmse_ = 1/M * np.sum(abs(y_true - y_predicted)**2)/np.sum(abs(y_true))**2
    if db:
        nmse_ = np.log10(nmse_)
    return nmse_

def tf_nmse(y_true, y_predicted):
    y_true = tf.cast(y_true, y_predicted.dtype)
    M = y_true.shape[-1]
    nmse_ = 1/M * tf.math.reduce_sum(tf.abs(y_true - y_predicted)**2)/tf.math.reduce_sum(tf.abs(y_true))**2
    return nmse_

def cos_sim(a, b):
    return abs(np.vdot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b))

def infer_sf(generator, inputs, frq, grid, sparsegen=False, ptych=False):
    # if not isinstance(frq, np.ndarray):
    Ginputs = inputs['Ginputs']
    k_vec = inputs['wavenumber_vec']

    frq = np.atleast_1d(frq)
    pred_coefficients = generator(Ginputs, training=False)
    if sparsegen:
        vsparse = inputs['v_sparse']
        if pred_coefficients.dtype == tf.complex64:
            pred_coefficients_cmplx = pred_coefficients + vsparse
        else:
            pred_coefficients_cmplx = array_to_complex(pred_coefficients + vsparse)
    elif ptych:
        pred_coefficients_cmplx = inputs['coefficients']
    else:
        if pred_coefficients.dtype == tf.complex64:
            pred_coefficients_cmplx = pred_coefficients
        else:
            pred_coefficients_cmplx = array_to_complex(pred_coefficients)

    H, k = get_sensing_mat(frq, generator.output.shape[1], grid[0], grid[1], grid[2], k_samp=k_vec)
    # H = tf.transpose(H)
    pred_sound_fields = tf.einsum('ijk, ik -> ij', H, pred_coefficients_cmplx)
    pred_sound_fields = tf.squeeze(pred_sound_fields)
    return pred_sound_fields, H

def mask_measurements(pmeasured, grid, subsample_ratio = 0.1, frequency = 250, decimate = True, seed = None):
    rng = np.random.RandomState(seed)
    Pmeasured = np.fft.rfft(pmeasured, n = 16384, axis = -1)
    fs = 16000 # sampling rate
    freq = np.fft.rfftfreq( n = 16384, d = 1/fs)
    ind = np.argmax(freq > frequency )
    Pm = Pmeasured[:, ind]
    if decimate:
        subsample_ind = rng.randint(low = 0, high = len(Pm), size = int(subsample_ratio*len(Pm)))
        Pm = Pm[subsample_ind]
        gridm = grid[:, subsample_ind]
    else:
        gridm = grid
    return Pm, gridm

def normalize_pressure(p, normalization = 'l2norm', epsilon=1e-8):
    assert p.ndim == 1
    if normalization == 'maxabs':
        return p/np.max(abs(p)), np.max(abs(p))
    if normalization == 'l2norm':
        return p/np.linalg.norm(p), np.linalg.norm(p)
    if normalization == 'standardization':
        mu = p.mean()
        var = p.var()
        pnorm = (p - mu)/np.sqrt(var + epsilon)
        return pnorm, (mu, var)

def array_to_complex(arr):
    cmplx = tf.complex(arr[..., 0], arr[..., 1])
    return cmplx

def nmse(y_true, y_predicted, db=True):
    nmse_ = np.mean(abs(y_true - y_predicted) ** 2) / np.mean(abs(y_true)** 2)
    if db:
        nmse_ = 10*np.log10(nmse_)
    return nmse_

def mac_similarity(a,b):
    return abs(a.T.conj() @ b)**2 / ((a.T.conj()@a) * (b.T.conj()@b))

def measurements_ratio_inference(model_direc, ratios, datadict, frequency=250.):
    nmses = []
    macs = []
    cossim = []
    pinference = []
    ptrues = []
    t = trange(len(ratios), desc='Reconstructing...', leave=True)
    nmics_list = []
    G = load_model(model_direc)

    G.encoder.trainable = False
    G.decoder.trainable = True
    G = tf.keras.Sequential([tf.keras.Input(shape=(16,)), G.decoder])

    for i in t:
        r = ratios[i]
        f_indx = np.argmin(datadict['f'] < frequency)
        ptrue = datadict['responses'][0, :, f_indx]
        # f_rec = datadict['f'][f_indx]
        ptrue_norm, _ = normalize_pressure(ptrue, normalization= 'l2norm')
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar = ptrue_norm.reshape(21, 21, 1)
        input_shape = tf.convert_to_tensor(ptrue_planar).shape
        mask = get_mask(input_shape, subsample_ratio=r, seed=1234)

        pm = ptrue_planar * mask.numpy()

        pm = pm.flatten()

        unmasked_indices = np.argwhere(pm != 0.)
        measurementgrid = datadict['grid_mic'][:, unmasked_indices]
        pm = pm[unmasked_indices].T
        nmics = np.count_nonzero(mask[...,0].numpy())
        pm, _ = normalize_pressure(pm.squeeze(0), normalization= 'l2norm')

        pm = tf.convert_to_tensor(pm)

        optz = tf.keras.optimizers.Adam(learning_rate=0.1)

        Gvars = CSGAN(G, optz, pm, measurementgrid, frequency, n_resets=1, lambda_z=0.00001, max_itr=1500)
        H, k = get_sensing_mat(frequency,
                               G.output.shape[1],
                               datadict['grid_mic'][0],
                               datadict['grid_mic'][1],
                               datadict['grid_mic'][2])
        predicted_coefficients = G([Gvars['zhat']])
        predicted_coefficients = tf.complex(predicted_coefficients[..., 0], predicted_coefficients[..., 1])
        ppred = tf.einsum('ijk, ik -> ij', H, predicted_coefficients)
        ppred = tf.squeeze(ppred)
        ppred_norm, _ = normalize_pressure(ppred.numpy(), normalization= 'l2norm')
        nmse_ = nmse(abs(ptrue_norm),abs( ppred_norm))
        mac_ = mac_similarity(abs(ptrue_norm), abs(ppred_norm.flatten()))
        cosine_sim = cosine(abs(ptrue_norm), abs(ppred_norm.flatten()))
        nmses.append(nmse_)
        macs.append(mac_)
        cossim.append(cosine_sim)
        pinference.append( abs(ppred_norm.flatten()))
        ptrues.append(abs(ptrue_norm))
        nmics_list.append(nmics)
        t.set_description("Frequency: {} Hz Nmics : {} MAC: {:.4e} NMSE: {:.4e}".format(frequency, nmics,
                                                                                        mac_, nmse_))
    return np.array(pinference), np.array(ptrues), \
           np.array(nmses), np.array(macs), np.array(cossim), np.array(nmics_list)

def frequency_sweep_inference(model_direc, frequencies, datadict, subsample_ratio = 0.3):
    nmses = []
    macs = []
    cossim = []
    pinference = []
    ptrues = []
    t = trange(len(frequencies), desc='Reconstructing...', leave=True)
    G = load_model(model_direc)

    G.encoder.trainable = False
    G.decoder.trainable = True
    G = tf.keras.Sequential([tf.keras.Input(shape=(16,)), G.decoder])

    for i in t:
        f = frequencies[i]
        f_indx = np.argmin(datadict['f'] < f)
        ptrue = datadict['responses'][0, :, f_indx]
        # f_rec = datadict['f'][f_indx]
        ptrue_norm, _ = normalize_pressure(ptrue, normalization= 'l2norm')
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar = ptrue_norm.reshape(21, 21, 1)
        input_shape = tf.convert_to_tensor(ptrue_planar).shape
        mask = get_mask(input_shape, subsample_ratio=subsample_ratio, seed=1234)

        pm = ptrue_planar * mask.numpy()

        pm = pm.flatten()

        unmasked_indices = np.argwhere(pm != 0.)
        measurementgrid = datadict['grid_mic'][:, unmasked_indices]
        pm = pm[unmasked_indices].T
        nmics = np.count_nonzero(mask[...,0].numpy())
        pm, _ = normalize_pressure(pm.squeeze(0), normalization= 'l2norm')

        pm = tf.convert_to_tensor(pm)

        optz = tf.keras.optimizers.Adam(learning_rate=0.1)

        Gvars = CSGAN(G, optz, pm, measurementgrid, f, n_resets=1, lambda_z=0.00001, max_itr=1500)
        H, k = get_sensing_mat(f,
                               G.output.shape[1],
                               datadict['grid_mic'][0],
                               datadict['grid_mic'][1],
                               datadict['grid_mic'][2])
        predicted_coefficients = G([Gvars['zhat']])
        predicted_coefficients = tf.complex(predicted_coefficients[..., 0], predicted_coefficients[..., 1])
        ppred = tf.einsum('ijk, ik -> ij', H, predicted_coefficients)
        ppred = tf.squeeze(ppred)
        ppred_norm, _ = normalize_pressure(ppred.numpy(), normalization= 'l2norm')
        nmse_ = nmse(abs(ptrue_norm),abs( ppred_norm))
        mac_ = mac_similarity(abs(ptrue_norm), abs(ppred_norm.flatten()))
        cosine_sim = cosine(abs(ptrue_norm), abs(ppred_norm.flatten()))
        nmses.append(nmse_)
        macs.append(mac_)
        cossim.append(cosine_sim)
        pinference.append( abs(ppred_norm.flatten()))
        ptrues.append(abs(ptrue_norm))

        t.set_description("Frequency: {} Hz Nmics : {} MAC: {:.4e} NMSE: {:.4e}".format(f, nmics,
                                                                                        mac_, nmse_))
    return np.array(pinference), np.array(ptrues), \
           np.array(nmses), np.array(macs), np.array(cossim), nmics

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


def get_latent(shape = (1,16)):
    return 100*tf.random.normal(shape)
def CSGAN(generator,
          optimizer,
          Pmeasured,
          grid_measured,
          frq,
          n_resets=5,
          max_itr=1000,
          lambda_z=0.1):
    # Instatiate (frequency) label for generator input
    freq = tf.constant(np.atleast_2d(frq))
    loss_best = 1e10
    frq = np.atleast_1d(frq)

    H, k = get_sensing_mat(frq,
                           generator.output.shape[1],
                           grid_measured[0],
                           grid_measured[1],
                           grid_measured[2])

    for ii in range(n_resets):
        z_batch = get_latent((1, 16))
        z_batch = tf.Variable(z_batch, name='latent_z')
        t = trange(max_itr, desc='Loss', leave=True)
        for i in t:
            with tf.GradientTape() as tape:
                tape.watch(z_batch)
                # H = tf.transpose(H)
                fake_coefficients = generator(z_batch, training=False)
                fake_coefficients = array_to_complex(fake_coefficients)
                fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients)
                # fake_sound_fields, _ = tf_normalize_pressure(fake_sound_fields)
                # znorm = tf.cast(tf.norm(z_batch, ord = 2), dtype = tf.complex64)
                misfit = tf.reduce_mean(tf.abs(Pmeasured - fake_sound_fields) ** 2)
                znorm = tf.norm(z_batch, ord=2)
                # znorm =  total_var_norm(fake_coefficients_cmplx, 0)
                znorm = tf.cast(znorm, dtype=misfit.dtype)

                # MACLoss = MAC_loss(Pmeasured, fake_sound_fields)
                # MACLoss = tf.cast(MACLoss, znorm.dtype)
                misfit = tf.reduce_mean(tf.abs(Pmeasured - fake_sound_fields)**2)
                NMSE = tf_nmse(Pmeasured, fake_sound_fields)
                # NMSE = tf.cast(nmse, znorm.dtype)
                # loss = misfit + 100*nmse +MACLoss #+ lambda_z * znorm  # + 1*MACLoss
                loss = misfit + lambda_z * znorm  # + 1*MACLoss

                t.set_description("iter: {} loss: {:.4f}  nmse: {:.4f} || z ||_2 : {:.4f} ".format(i, misfit.numpy(),
                                                                                                   np.log10(NMSE.numpy()),
                                                                                                   znorm.numpy()),
                                  refresh=True)
            if znorm < .9:
                continue
            gradients = tape.gradient(loss, [z_batch])
            optimizer.apply_gradients(zip(gradients, [z_batch]))
        if tf.abs(loss) < loss_best:
            i_best = ii
            zhat = tf.identity(z_batch)
            loss_best = tf.abs(loss)
    print(f"best iter: {i_best}")
    outputs = {}
    inputs = [zhat]
    outputs['Ginputs'] = inputs
    outputs['zhat'] = zhat
    outputs['wavenumber_vec'] = k
    outputs['H'] = H
    return outputs

#%%
model = './VAE model'
model_path = './'
data_dir = './data_folder'
datapath = '../Transfer_learning_data/MeshRIR_set.npz'
dataset = 'real_data'
f_rec = 552.
results_dir = './Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_dir = os.path.join(results_dir, model)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_direc = model

datadict = get_measurement_vector(datapath)

ratios = np.linspace(0.05, .5, 25)
# ratios = np.linspace(.8, .9, 25)
# pinference_mag, ptrue_mag, nmse_, mac, cossim, Nmics = measurements_ratio_inference(model_direc,
#                                                                               ratios,
#                                                                               datadict,
#                                                                               frequency = f_rec)
#
#
# np.savez_compressed('./VAE_magnitude_inference_vs_mics.npz',
#                     pinference_mag = pinference_mag,
#                     ptrue_mag = ptrue_mag,
#                     nmse_ = nmse_,
#                     mac = mac,
#                     cossim = cossim,
#                     Nmics = Nmics
#                     )
# csgan_data = np.load('../CSGAN/Results/inference_meas_no_weigtht_opt_lambda_5e-1.npz')


fc = [16., 20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250.,
      315., 400., 500., 630., 800., 1000., 1250, 1600., 2000.]

true_fcs = []
fvec = np.fft.rfftfreq(16384, 1/16000)
for ffc in fc:
    find = np.argmin(fvec < ffc)
    true_fcs.append(fvec[find])

pinference_mag, ptrue_mag, nmse_, mac, cossim, Nmics = frequency_sweep_inference(model_direc,
                                                                                 true_fcs,
                                                                                 datadict,
                                                                                 subsample_ratio= 0.4
                                                                                 )
np.savez_compressed('Results/VAE_magnitude_inference_vs_freqs.npz',
                    pinference_mag = pinference_mag,
                    ptrue_mag = ptrue_mag,
                    nmse_ = nmse_,
                    mac = mac,
                    cossim = cossim,
                    Nmics = Nmics,
                    freqs = np.array(true_fcs)
                    )

# ptrue = csgan_data['ptrue']
# pcsgan = csgan_data['ppred']
#
# mac_vae = []
# nmse_vae = []
# cossim_vae = []
# corr_vae = []
# mac_csgm = []
# nmse_csgm = []
# cossim_csgm = []
# corr_csgm = []
#
# for ii in range(len(ptrue)):
#     mac_csgm.append(mac_similarity(abs(ptrue[ii]), abs(pcsgan[ii])))
#     nmse_csgm.append(nmse(abs(ptrue[ii]), abs(pcsgan[ii])))
#     cossim_csgm.append(cosine(abs(ptrue[ii]), abs(pcsgan[ii])))
#     corr_csgm.append(np.corrcoef(abs(ptrue[ii]), abs(pcsgan[ii]))[0,1])
#     mac_vae.append(mac_similarity(ptrue_mag[ii], pinference_mag[ii]))
#     nmse_vae.append(nmse(ptrue_mag[ii], pinference_mag[ii]))
#     cossim_vae.append(cosine(ptrue_mag[ii], pinference_mag[ii]))
#     corr_vae.append(np.corrcoef(ptrue_mag[ii], pinference_mag[ii])[0,1])
#
# plt.plot(Nmics, nmse_csgm, label = 'csgm')
# plt.plot(Nmics, nmse_vae, label = 'vae')
# plt.ylabel('NMSE [dB]')
# plt.xlabel('Number of microphones')
# plt.legend()
# plt.show()
#
# plt.plot(Nmics, corr_csgm, label = 'csgm')
# plt.plot(Nmics, corr_vae, label = 'vae')
# plt.ylabel('Correlation')
# plt.xlabel('Number of microphones')
# plt.legend()
# plt.show()

import numpy as np
import os
from tensorflow.keras.layers import Input, Activation, BatchNormalization, UpSampling2D  # noqa
from tensorflow.keras.layers import LeakyReLU, Conv2D, Concatenate  # noqa
from tensorflow.keras import Model  # noqa
import tensorflow as tf
from tensorflow.keras.models import load_model  # noqa

# os.chdir('./AE')
from AE.aux_func import bcolors, mask_pressure_field, get_mask, plot_sf
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from scipy.spatial.distance import cosine

from tueplots import axes, figsizes, bundles, cycler, markers
from tueplots.constants.color import palettes

# plt.rcParams.update({'figure.dpi': '100'})
# plt.style.use(['science','ieee'])

plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.tick_direction(x='in', y='in'))
plt.rcParams.update(axes.grid())
plt.rcParams.update(figsizes.neurips2021(nrows=2, ncols=2))
plt.rcParams.update(markers.inverted())
# plt.rcParams.update()
plt.rcParams.update(cycler.cycler(color=palettes.high_contrast))


def nmse(y_true, y_predicted, db=True):
    nmse_ = np.mean(abs(y_true - y_predicted) ** 2) / np.mean(abs(y_true) ** 2)
    if db:
        nmse_ = 10 * np.log10(nmse_)
    return nmse_


def mac_similarity(a, b):
    return abs(a.T.conj() @ b) ** 2 / ((a.T.conj() @ a) * (b.T.conj() @ b))


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
        return pnorm, (mu, var)


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


def measurements_inference(model_direc, ratios, datadict, frequency=250.):
    nmses = []
    macs = []
    cossim = []
    pinference = []
    ptrues = []
    t = trange(len(ratios), desc='Reconstructing...', leave=True)
    nmics_list = []
    G = load_model(model_direc)
    G.trainable = False
    for i in t:
        r = ratios[i]
        f_indx = np.argmin(datadict['f'] < frequency)
        ptrue = datadict['responses'][0, :, f_indx]
        # f_rec = datadict['f'][f_indx]
        ptrue_norm, _ = normalize_pressure(ptrue, normalization='maxabs')
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar = ptrue_norm.reshape(21, 21, 1)
        input_shape = tf.convert_to_tensor(ptrue_planar).shape
        mask = get_mask(input_shape, subsample_ratio=r, seed=1234)

        pm = ptrue_planar * mask.numpy()
        # unmasked_indices = np.argwhere(pm != 0.)
        nmics = np.count_nonzero(mask[..., 0].numpy())

        pm = tf.convert_to_tensor(pm)
        pm_mag = tf.abs(pm)
        # ptrue_planar_formatted = tf.convert_to_tensor(
        #     np.concatenate((ptrue_planar.real, ptrue_planar.imag), axis=-1)[None, ...])

        jitter = tf.random.normal(pm_mag.shape, mean=0, stddev=1e-6)
        # pressure is already scaled at this point
        pm_mag = tf.where(pm_mag == 0., jitter, pm_mag)

        ppred = G([pm_mag, mask])

        nmse_ = nmse(abs(ptrue_norm), ppred.numpy().flatten())
        mac_ = mac_similarity(abs(ptrue_norm), ppred.numpy().flatten())
        cosine_sim = cosine(abs(ptrue_norm), ppred.numpy().flatten())
        nmses.append(nmse_)
        macs.append(mac_)
        cossim.append(cosine_sim)
        pinference.append(ppred.numpy().flatten())
        ptrues.append(abs(ptrue_norm))
        nmics_list.append(nmics)
        t.set_description("Frequency: {} Hz Nmics : {} MAC: {:.4e} NMSE: {:.4e}".format(frequency, nmics,
                                                                                        mac_, nmse_))
    return np.array(pinference), np.array(ptrues), \
           np.array(nmses), np.array(macs), np.array(cossim), np.array(nmics_list)


def frequency_sweep(model_direc, datadict, subsample_ratio=0.3):
    # fc = [16., 20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250.,
    #       315., 400., 500., 630., 800., 1000., 1250, 1600., 2000.]
    fc = [100., 125., 160., 200., 250.,
          315., 400., 500., 630., 800., 1000., 1250, 1600., 2000.]

    true_fcs = []
    fvec = np.fft.rfftfreq(16384, 1 / 16000)
    for ffc in fc:
        find = np.argmin(fvec < ffc)
        true_fcs.append(fvec[find])
    nmses = []
    macs = []
    cossim = []
    pinference = []
    ptrues = []
    t = trange(len(true_fcs), desc='Reconstructing...', leave=True)
    nmics_list = []
    G = load_model(model_direc)
    G.trainable = False
    for i in t:
        f_indx = np.argmin(fvec < true_fcs[i])
        ptrue = datadict['responses'][0, :, f_indx]
        # f_rec = datadict['f'][f_indx]
        ptrue_norm, _ = normalize_pressure(ptrue, normalization='maxabs')
        # for sake of using exactly the same mask as autoencoder:
        ptrue_planar = ptrue_norm.reshape(21, 21, 1)
        input_shape = tf.convert_to_tensor(ptrue_planar).shape
        mask = get_mask(input_shape, subsample_ratio=subsample_ratio, seed=1234)

        pm = ptrue_planar * mask.numpy()
        # unmasked_indices = np.argwhere(pm != 0.)
        nmics = np.count_nonzero(mask[..., 0].numpy())

        pm = tf.convert_to_tensor(pm)
        pm_mag = tf.abs(pm)
        # ptrue_planar_formatted = tf.convert_to_tensor(
        #     np.concatenate((ptrue_planar.real, ptrue_planar.imag), axis=-1)[None, ...])

        jitter = tf.random.normal(pm_mag.shape, mean=0, stddev=1e-6)
        # pressure is already scaled at this point
        pm_mag = tf.where(pm_mag == 0., jitter, pm_mag)

        ppred = G([pm_mag, mask])

        nmse_ = nmse(abs(ptrue_norm), ppred.numpy().flatten())
        mac_ = mac_similarity(abs(ptrue_norm), ppred.numpy().flatten())
        cosine_sim = cosine(abs(ptrue_norm), ppred.numpy().flatten())
        nmses.append(nmse_)
        macs.append(mac_)
        cossim.append(cosine_sim)
        pinference.append(ppred.numpy().flatten())
        ptrues.append(abs(ptrue_norm))
        t.set_description("Frequency: {} Hz Nmics : {} MAC: {:.4e} NMSE: {:.4e}".format(true_fcs[i], nmics,
                                                                                        mac_, nmse_))
    return np.array(pinference), np.array(ptrues), \
           np.array(nmses), np.array(macs), np.array(cossim), nmics, np.array(fc)


# %%
# CONSTANTS
results_dir = './Results'
model = 'Conv64_unnormalised'
# model = 'Conv64x64'
datapath = './Transfer_learning_data/MeshRIR_set.npz'
cslr = 0.001
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_dir = os.path.join(results_dir, model)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_direc = os.path.join('./AE', model)
# G = load_model(model_direc)
datadict = get_measurement_vector(datapath)
# %%

ratios = np.linspace(0.05, .5, 25)
pinference_mag, ptrue_mag, nmse_, mac, cossim, Nmics = measurements_inference(model_direc,
                                                                              ratios,
                                                                              datadict,
                                                                              frequency=552)

# %%
# nmse, mac, Nmics, freq, ppred, ptrue
csgan_data = np.load('./CSGAN/Results/inference_meas_no_weigtht_opt_lambda_5e-1.npz')
vae_data = np.load('./VAE/Results/VAE_magnitude_inference_vs_mics.npz')

ptrue = csgan_data['ptrue']
pcsgan = csgan_data['ppred']

ptrue_vae = vae_data['ptrue_mag']
ppred_vae = vae_data['pinference_mag']


# %%
mac_ae = []
nmse_ae = []
cossim_ae = []
corr_ae = []
mac_vae = []
nmse_vae = []
cossim_vae = []
corr_vae = []
mac_csgm = []
nmse_csgm = []
cossim_csgm = []
corr_csgm = []

for ii in range(len(ptrue)):
    mac_csgm.append(mac_similarity(abs(ptrue[ii]), abs(pcsgan[ii])))
    nmse_csgm.append(nmse(abs(ptrue[ii]), abs(pcsgan[ii])))
    cossim_csgm.append(cosine(abs(ptrue[ii]), abs(pcsgan[ii])))
    corr_csgm.append(np.corrcoef(abs(ptrue[ii]), abs(pcsgan[ii]))[0, 1])
    mac_ae.append(mac_similarity(ptrue_mag[ii], pinference_mag[ii]))
    nmse_ae.append(nmse(ptrue_mag[ii], pinference_mag[ii]))
    cossim_ae.append(cosine(ptrue_mag[ii], pinference_mag[ii]))
    corr_ae.append(np.corrcoef(ptrue_mag[ii], pinference_mag[ii])[0, 1])
    mac_vae.append(mac_similarity(abs(ptrue_vae[ii]), abs(ppred_vae[ii])))
    nmse_vae.append(nmse(ptrue_vae[ii], ppred_vae[ii]))
    cossim_vae.append(cosine(ptrue_vae[ii], ppred_vae[ii]))
    corr_vae.append(np.corrcoef(ptrue_vae[ii], ppred_vae[ii])[0, 1])

with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=2)):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(Nmics, nmse_csgm, 'o-', linewidth=1, markersize=3, label='PWGAN')
    ax1.plot(Nmics, nmse_ae, 'o-', linewidth=1, markersize=3, label='AE')
    ax1.plot(Nmics, nmse_vae, 'o-', linewidth=1, markersize=3, label='VAE')
    ax1.set_ylabel('NMSE [dB]')
    ax1.set_xlabel('Number of microphones')
    ax1.grid(ls=':', color = 'k', which = 'both')

    ax2.plot(Nmics, mac_csgm, 'o-', linewidth=1, markersize=3, label='PWGAN')
    ax2.plot(Nmics, mac_ae, 'o-', linewidth=1, markersize=3, label='AE')
    ax2.plot(Nmics, mac_vae, 'o-', linewidth=1, markersize=3, label='VAE')
    ax2.set_ylabel('Spatial Similarity')
    ax2.set_xlabel('Number of microphones')
    ax2.grid(ls=':', color = 'k', which = 'both')
    ax2.legend()
    fig.show()

    fig.savefig('./PaperFigs/NMSE_mac_vs_nmics.pdf', dpi = 500)
# %% Frequency Sweep

pinference_mag, ptrue_mag, nmse_, mac, cossim, Nmics, fcs = frequency_sweep(model_direc,
                                                                       datadict,
                                                                       subsample_ratio=0.15)
# %%
csgan_data = dict(np.load('./CSGAN/Results/inference_vs_freq_4.npz'))
vae_data = np.load('./VAE/Results/VAE_magnitude_inference_vs_freqs.npz')

ptrue = csgan_data['ptrue']
# ptrue /= np.max(abs(ptrue))
pcsgan = csgan_data['ppred_2']
pptych = csgan_data['ppred_1']
# pcsgan /= np.max(abs(pcsgan))

ptrue_vae = vae_data['ptrue_mag']
ppred_vae = vae_data['pinference_mag']

#%%

mac_ae = []
nmse_ae = []
cossim_ae = []
corr_ae = []
mac_vae = []
nmse_vae = []
cossim_vae = []
corr_vae = []
mac_csgm = []
nmse_csgm = []
cossim_csgm = []
corr_csgm = []

for ii in range(len(ptrue)):
    ptruecsgm = ptrue[ii]/np.max(abs(ptrue[ii]))
    pcsgm = pcsgan[ii]/np.max(abs(pcsgan[ii]))
    ptrueae = ptrue_mag[ii]/np.max(abs(ptrue_mag[ii]))
    pae = pinference_mag[ii]/np.max(abs(pinference_mag[ii]))
    # ptruecsgm = ptrue[ii]
    # pcsgm = pcsgan[ii]
    # ptrueae = ptrue_mag[ii]
    # pae = pinference_mag[ii]
    mac_csgm.append(mac_similarity(abs(ptruecsgm), abs(pcsgm)))
    nmse_csgm.append(nmse(abs(ptruecsgm), abs(pcsgm)))
    cossim_csgm.append(cosine(abs(ptrue[ii]), abs(pcsgan[ii])))
    corr_csgm.append(np.corrcoef(abs(ptrue[ii]), abs(pcsgan[ii]))[0, 1])
    mac_ae.append(mac_similarity(ptrueae, pae))
    nmse_ae.append(nmse(ptrueae[ii], pae[ii]))
    cossim_ae.append(cosine(ptrue_mag[ii], pinference_mag[ii]))
    corr_ae.append(np.corrcoef(ptrue_mag[ii], pinference_mag[ii])[0, 1])
    mac_vae.append(mac_similarity(abs(ptrue_vae[7+ii]), abs(ppred_vae[7+ii])))
    nmse_vae.append(nmse(ptrue_vae[7+ii], ppred_vae[7+ii]))
    cossim_vae.append(cosine(ptrue_vae[7+ii], ppred_vae[7+ii]))
    corr_vae.append(np.corrcoef(ptrue_vae[7+ii], ppred_vae[7+ii])[0, 1])

with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=2)):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.semilogx(fcs, nmse_csgm, 'o-', linewidth=1, markersize=3, label='PWGAN')
    ax1.semilogx(fcs, nmse_ae, 'o-', linewidth=1, markersize=3, label='AE')
    ax1.semilogx(fcs, nmse_vae, 'o-', linewidth=1, markersize=3, label='VAE')
    ax1.set_ylabel('NMSE [dB]')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.grid(ls=':', color = 'k', which = 'both')

    ax2.semilogx(fcs, mac_csgm, 'o-', linewidth=1, markersize=3, label='PWGAN')
    ax2.semilogx(fcs, mac_ae, 'o-', linewidth=1, markersize=3, label='AE')
    ax2.semilogx(fcs, mac_vae, 'o-', linewidth=1, markersize=3, label='VAE')
    ax2.set_ylabel('Spatial Similarity')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.grid(ls=':', color = 'k', which = 'both')
    ax2.legend()
    fig.show()
    fig.savefig('./PaperFigs/NMSE_mac_vs_freq.pdf', dpi = 500)
#%%
from scipy.interpolate import griddata

fc = [100., 125., 160., 200., 250.,
      315., 400., 500., 630., 800., 1000., 1250, 1600., 2000.]
cmap1 = mpl.colors.ListedColormap(['none', 'black'])

true_fcs = []
Pressure = []
fvec = np.fft.rfftfreq(16384, 1 / 16000)
for ffc in fc:
    find = np.argmin(fvec < ffc)
    true_fcs.append(fvec[find])
input_shape = tf.zeros((21, 21, 1)).shape
mask = get_mask(input_shape, subsample_ratio=0.15, seed=1234, flatten= True)

for i in range(len(true_fcs)):
    f_indx = np.argmin(fvec < true_fcs[i])

    ptrue = datadict['responses'][0, :, f_indx]
    Pressure.append(ptrue)
mask = mask.numpy()
grid = datadict['grid_mic']
maskindx = np.argwhere(mask > 0)

mic_grid = grid[:, maskindx.squeeze(-1)]
X = grid[0]
Y = grid[1]
# x, y = X, Y
# clim = (abs(P).min(), abs(P).max())
dx = 0.5 * X.ptp() / mask.size
dy = 0.5 * Y.ptp() / mask.size
plt.rcParams.update(axes.tick_direction(x='out', y='out'))

with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=3)):
    fig, ax = plt.subplots(1,1)
    ax.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper',
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    fig.show()
    fig.savefig("./PaperFigs/mic_positions.pdf", dpi = 400, bbox_inches = 'tight')
with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=4)):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,
                                                  6,
                                                  gridspec_kw = {'wspace':0, 'hspace':0})
    ax1, _ = plot_sf(Pressure[1], grid[0], grid[1], ax = ax1, f = int(fc[1]), name = '')
    ax2, _ = plot_sf(Pressure[3], grid[0], grid[1], ax = ax2, f = int(fc[3]), name = '')
    ax3, _ = plot_sf(Pressure[6], grid[0], grid[1], ax = ax3, f = int(fc[6]), name = '')
    ax4, _ = plot_sf(Pressure[8], grid[0], grid[1], ax = ax4, f = int(fc[8]), name = '')
    ax5, _ = plot_sf(Pressure[10], grid[0], grid[1], ax = ax5, f = int(fc[10]), name = '')
    ax6, _ = plot_sf(Pressure[12], grid[0], grid[1], ax = ax6, f = int(fc[12]), name = '')
    ax1.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax2.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax3.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax4.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax5.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax6.imshow(mask.reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    # ax1.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    # ax2.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    # ax3.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    # ax4.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    # ax5.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    # ax6.scatter(mic_grid[0], mic_grid[1], color = 'k', marker = 'x', s = 5, linewidth = .8, alpha = 0.5)
    for axx in (ax1, ax2, ax3, ax4, ax5, ax6):
        axx.axis('off')
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.8, bottom = 0.01,
                        wspace=0, hspace=0)
    fig.show()
    fig.savefig('./PaperFigs/Frequency_soundfield_mic_plot.pdf', dpi = 400, bbox_inches = 'tight')
#%%
ratios = np.linspace(0.05, .5, 25)

Pressure = []
fvec = np.fft.rfftfreq(16384, 1 / 16000)

find = np.argmin(fvec < 552)

input_shape = tf.zeros((21, 21, 1)).shape
ptrue = datadict['responses'][0, :, find]
grid = datadict['grid_mic']
mic_grids = []
masks = []
nmics = []
for i in range(len(ratios)):
    mask = get_mask(input_shape, subsample_ratio=ratios[i], seed=1234, flatten = True)
    mask = mask.numpy()
    maskindx = np.argwhere(mask > 0)
    mic_grid = grid[:, maskindx.squeeze(-1)]

    mic_grids.append(mic_grid)
    masks.append(mask)
    nmics.append(np.count_nonzero(mask))
    print(np.count_nonzero(mask))
# mic_grids = np.hstack(mic_grids)
with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=3)):
    fig, ax = plt.subplots(1,1)
    ax, _ = plot_sf(ptrue, grid[0], grid[1], ax=ax)

    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    fig.show()
    fig.savefig("./PaperFigs/reference_sf_520Hz.pdf", dpi = 400, bbox_inches = 'tight')

with plt.rc_context(bundles.neurips2021(usetex=True, family='serif', ncols=4)):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,
                                                  6,
                                                  gridspec_kw = {'wspace':0, 'hspace':0})
    ax1, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax1, name = f'Mics: {nmics[0]}')
    ax2, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax2, name = f'Mics: {nmics[4]}')
    ax3, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax3, name = f'Mics: {nmics[9]}')
    ax4, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax4, name = f'Mics: {nmics[14]}')
    ax5, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax5, name = f'Mics: {nmics[17]}')
    ax6, _ = plot_sf(ptrue, grid[0], grid[1], ax = ax6, name = f'Mics: {nmics[20]}')
    ax1.imshow(masks[0].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax2.imshow(masks[4].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax3.imshow(masks[9].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax4.imshow(masks[14].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax5.imshow(masks[17].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax6.imshow(masks[20].reshape(21,21), cmap = cmap1,
               origin='upper', alpha = 0.5,
               extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    for axx in (ax1, ax2, ax3, ax4, ax5, ax6):
        axx.axis('off')
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(top=0.8, bottom = 0.01,
                        wspace=0, hspace=0)
    fig.show()
    fig.savefig('./PaperFigs/Soundfield_Nmics_plot.pdf', dpi = 400, bbox_inches = 'tight')


import tensorflow as tf
from tensorflow.keras.models import load_model # noqa
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from GANTrainingScripts.aux_functions import get_sensing_mat
from tqdm import trange
from src.CSGAN_auxfun import get_measurement_vector, get_optimizer, mask_measurements, plot_array_pressure, \
    nmse, cos_sim, stack_real_imag_H, MutualCoherence, columnwise_coherence
from src.algorithms import CSGAN, infer_sf, adaptive_CSGAN, Ridge_regression, CSGAN_Sparse, planewaveGAN, Lasso_regression
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import click
from librosa import resample

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# matplotlib params
# matplotlib.use("png")
tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

mpl.rcParams.update(tex_fonts)
# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# plt.rcParams["figure.figsize"] = (6.694, 5)
plt.rcParams['figure.constrained_layout.use'] = True

def load_list(key, dict):
    if key in dict.item().keys():
        val = dict.item()[key]
        return val
    else:
        return []

def set_size(width_pt=483.697, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


# To reset the global figure size back to default for subsequent plots:
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
def normalize_with_moments(x, axes=0, epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed, (mean, variance)

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

def losses(y_batch, y_hat_batch, z_batch):
    m_loss1_batch = tf.keras.losses.MAE(y_batch, y_hat_batch)
    m_loss2_batch = tf.keras.losses.MSE(y_batch, y_hat_batch)
    zp_loss_batch = tf.norm(z_batch,
                            ord=2,
                            name='z_batch')
    return m_loss1_batch, m_loss2_batch, zp_loss_batch


@click.command()
# options_metavar='<options>'
@click.option('--model', default='Real_FC_data_augment', type=str,
              help='Which trained generator to use')
@click.option('--use_complex_net', default=False,
              help='If flagged, use complex valued neural net')
@click.option('--conditional_net', default=False,
              help='If flagged, use frequency as conditional input')
@click.option('--data_dir', default='../data_folder', type=str,
              help='Path of (measurement) data')
@click.option('--dataset', default='real_data', type=str,
              help='Type of data')
@click.option('--recon_freq', default=850., type=float,
              help='Reconstruct this frequency or reconstruct up to this frequency '
                   'if --recon_full is True')
@click.option('--recon_full', default=False, is_flag=True,
              help='If flagged, reconstruct up to the variable set by '
                   '--recon_freq')
@click.option('--store_predictions', default=True,
              help='If flagged, will store predictions in a npz file for later use')
@click.option('--least_squares', default=True, is_flag=True,
              help='If flagged, will also run regularised least squares '
                   'regression with sklean (ridge or lasso)')
@click.option('--sparsegen', default=False, is_flag=True,
              help='If flagged, use sparse deviations from span of '
                   'Generator by adding a trainable vector (Dhar et. al)')
@click.option('--resample_wav', default=True,
              help='Resample RIRs from fs = 16 kHz to fs = 8 kHz')
@click.option('--plot_sfs', default=False, is_flag=True,
              help='If flagged, plot some results for unspecified frequencies')
@click.option('--lambda_z', default=0.001, type=float,
              help='Lagrangian weight of || z ||_2 when fitting CSGAN')
@click.option('--csgan_iters', default=1500, type=int,
              help='number of iterations for when fitting CSGAN')
@click.option('--n_csgan_resets', default=1, type=int,
              help='Number of resets for CSGAN to find a good latent representation')
@click.option('--csgan_lr', default=0.001, type=float,
              help='Learning rate of (AdaM) optimizer for CSGAN')
@click.option('--adaptcs_lr', default=0.0001, type=float,
              help='Learning rate of (AdaM) optimizer for CSGAN')
@click.option('--adgan_iters', default=900, type=int,
              help='Learning rate of (AdaM) optimizer for CSGAN')
@click.option('--subsample_ratio', default = 1., type=float,
              help='Subsample array set by a factor set here')
@click.option('--lambda_adcsnorm', default=5e-2, type=float,
              help='Lagrangian weight of R(G(z)) term (regularization) when'
                   ' fitting Adaptive CSGAN')
@click.option('--ptychlr', default=0.02, type=float,
              help='Learning rate of (AdaM) optimizer for Fourier Ptychography GAN')
@click.option('--ptych_iters', default=3500, type=int,
              help='Learning rate of (AdaM) optimizer for Fourier Ptychography GAN')
@click.option('--lambdaptych', default=5e-8, type=float,
              help='Lagrangian weight of ||x - x_hat||_2^2 term (regularization) when'
                   ' fitting Fourier Ptychography GAN')
def run_CSGAN(model,
              data_dir,
              dataset,
              recon_freq,
              recon_full,
              sparsegen,
              conditional_net,
              subsample_ratio,
              lambda_z,
              csgan_iters,
              n_csgan_resets,
              csgan_lr,
              adaptcs_lr,
              lambda_adcsnorm,
              ptychlr,
              least_squares,
              lambdaptych,
              adgan_iters,
              plot_sfs,
              store_predictions,
              use_complex_net,
              ptych_iters,
              resample_wav
              ):
    results_dir = './Results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    from glob import glob
    model_path = 'Generator Models'
    model_direc = os.path.join(model_path, model)
    model_direc = os.path.join(os.getcwd(), model_direc)

    print(model_direc)
    print(glob(model_direc + '/*'))

    results_dir = os.path.join(results_dir, model)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # sparse deviations from span of generator - sometimes works...
    Sparse_Gen = sparsegen
    if dataset == 'ISM_sphere':
        zlab = True

    # global variables
    fs = 16000  # kHz
    sample_size = 16384
    calculate_metrics = False

    p, ptrue, grid, grid_true = get_measurement_vector(data_dir, dataset=dataset)

    if resample_wav:
        p = resample(p, orig_sr=fs, target_sr=8000)
        ptrue = resample(ptrue, orig_sr=fs, target_sr=8000)
        fs = 8000
        sample_size = p.shape[-1]

    freqs = np.fft.rfftfreq(sample_size, d=1 / fs)
    frq_indx = np.argmax(freqs > recon_freq)
    inference_file = results_dir + '/' + model + '_inference_{}.npy'.format(dataset)
    if os.path.isfile(inference_file):
        temp_dict = np.load(inference_file, allow_pickle= True)
        iter = np.max(temp_dict.item()['iters'])
        p_csgan_glob = load_list('p_csgan_glob', temp_dict)
        p_adcsgan_glob = load_list('p_adcsgan_glob', temp_dict)
        p_ptychcsgan_glob = load_list('p_ptychcsgan_glob', temp_dict)
        p_ridge_glob = load_list('p_ridge_glob', temp_dict)
        p_lass_glob = load_list('p_lass_glob', temp_dict)
        p_true = load_list('p_true', temp_dict)
        p_meas = load_list('p_meas', temp_dict)
        p_norm_const = load_list('p_norm_const', temp_dict)
        mse_csgan_glob = load_list('mse_csgan_glob', temp_dict)
        mac_csgan_glob = load_list('mac_csgan_glob', temp_dict)
        mse_ad_csgan_glob = load_list('mse_ad_csgan_glob', temp_dict)
        mac_ad_csgan_glob = load_list('mac_ad_csgan_glob', temp_dict)
        mse_ptych_csgan_glob = load_list('mse_ptych_csgan_glob', temp_dict)
        mac_ptych_csgan_glob = load_list('mac_ptych_csgan_glob', temp_dict)
        mse_ridge_glob = load_list('mse_ridge_glob', temp_dict)
        mac_ridge_glob = load_list('mac_ridge_glob', temp_dict)
        mse_lass_glob = load_list('mse_lass_glob', temp_dict)
        mac_lass_glob = load_list('mac_lass_glob', temp_dict)
        iters = load_list('iters', temp_dict)
        freq_vec = load_list('freq_vec', temp_dict)
    else:
        iter = 0
        p_csgan_glob = []
        p_adcsgan_glob = []
        p_ptychcsgan_glob = []
        p_ridge_glob = []
        p_lass_glob = []
        p_true = []
        p_meas = []
        p_norm_const = []
        mse_csgan_glob = []
        mac_csgan_glob = []
        mse_ad_csgan_glob = []
        mac_ad_csgan_glob = []
        mse_ptych_csgan_glob = []
        mac_ptych_csgan_glob = []
        mse_ridge_glob = []
        mac_ridge_glob = []
        mse_lass_glob = []
        mac_lass_glob = []
        iters = []
        freq_vec = []
    if recon_full:
        freq_iter = freqs[iter:]
    else:
        freq_iter = [freqs[frq_indx]]

    t = trange(len(freq_iter), desc='Frequency', leave=True, disable = recon_full)

    global_var_out = {}

    # get whole dataset
    for kk in t:
        f_rec = freq_iter[kk]
        # subsample = 0.1, seed 101 good results
        # masked measurements
        Pm, gridm = mask_measurements(p, grid, decimate=True, subsample_ratio=subsample_ratio, frequency=f_rec,
                                      seed=101, fs = fs)
        # true array and reference fields
        Parr, gridar = mask_measurements(p, grid, decimate=False, subsample_ratio=subsample_ratio, frequency=f_rec,
                                         seed=101,
                                         fs = fs)
        Ptrue, gridtrue = mask_measurements(ptrue, grid_true, decimate=False, frequency=f_rec, seed=101,
                                            fs = fs)

        # make sure everything is float32
        gridm = gridm.astype('float32')
        gridtrue = gridtrue.astype('float32')
        gridar = gridar.astype('float32')

        # which set to use? [e.g. Pm (array) or Pplane (masked on reference plane)
        Ptouse = Pm
        gridtouse = gridm
        # Ptouse = Parr
        # gridtouse = gridar

        # number of measured points
        nmeas = len(Ptouse)
        # total set is of length N
        N = len(Parr)
        print(80 * '~')
        print("Frequency: {} - Total number of measurements: {}/{}".format(round(f_rec, 2), nmeas, N))
        print(80 * '~')

        # normalise sound fields so they have a unit norm (this is so they can be processed by the network ~ [-1, 1])
        Pm, _ = normalize_pressure(Pm)
        Ptrue, _ = normalize_pressure(Ptrue)
        Parr, _ = normalize_pressure(Parr)
        Ptouse, p_norm = normalize_pressure(Ptouse)

        print("P shape: {} - grid shape: {}".format(Ptouse.shape, gridtouse.shape))
        # load generator network
        G = load_model(model_direc)
        G.trainable = False
        Pvec = Ptouse[np.newaxis, :]


        # ---------------------  CS GAN PREDICTION ------------------------------
        # CS GAN params
        # cslr = tf.keras.optimizers.schedules.ExponentialDecay(
        #     csgan_lr,
        #     decay_steps=200,
        #     decay_rate=0.96)
        cslr = csgan_lr
        # get optimizers for CSGAN and SparseGen
        optz = get_optimizer(cslr, 'adam')
        optv = get_optimizer(cslr, 'adam')

        # sparse GEN
        if Sparse_Gen:
            Gvars = CSGAN_Sparse(G, optz, optv, Pvec, gridtouse, f_rec, n_resets=n_csgan_resets, lambda_v=lambda_z,
                                 max_itr=1000, verbose = not recon_full)
        else:
            Gvars = CSGAN(G, optz, Pvec, gridtouse, f_rec, n_resets=n_csgan_resets, lambda_z=lambda_z, max_itr=csgan_iters,
                          complex_net= use_complex_net, conditional_net = conditional_net,
                          verbose = not recon_full)

        # predict CSGAN sound field
        Ppred, _ = infer_sf(G, Gvars, f_rec, gridtrue, sparsegen=Sparse_Gen)

        # ------------------- LBGFS CS GAN PREDICTION -----------------------
        # outputs = CSGAN_LBGFS(G,Pvec,gridtouse, f_rec, max_itr=200, k_vec=None, complex_net = use_complex_net)
        #
        # Ppred_lbgfs, _ = infer_sf(G, outputs, f_rec, gridtrue, sparsegen=False)

        # ------------------- Adaptive CS GAN PREDICTION -----------------------
        # Adaptive CS GAN params
        newlr = tf.keras.optimizers.schedules.ExponentialDecay(
            adaptcs_lr,
            decay_steps=50,
            decay_rate=0.9)

        # Adaptive CS GAN optimizers
        # opt_w = get_optimizer(csgan_lr, 'adam')
        # opt_z_new = get_optimizer(adaptcs_lr, 'adam')
        opt_w = get_optimizer(adaptcs_lr, 'adam')
        opt_z_new = get_optimizer(0.001, 'adam')

        # for now setting new latent var
        # znew = tf.random.normal((1, 128))
        znew = Gvars['Ginputs'][0]
        # run Adaptive CSGAN algorithm
        adaptCSGANvars = adaptive_CSGAN(G, opt_z_new, opt_w, Pvec, gridtouse, f_rec, znew,
                                        lambda_=lambda_adcsnorm, k_vec=Gvars['wavenumber_vec'],
                                        max_itr=adgan_iters, complex_net = use_complex_net,
                                        conditional_net = conditional_net,
                                        verbose = not recon_full)

        # set weights from CSGAN step for ptych
        G = load_model(model_direc)
        G.trainable = False

        # Infer Adaptive CSGAN pressure
        Ppred2, H = infer_sf(G, adaptCSGANvars, f_rec, gridtrue)

        # ---------------- Fourier Ptychography GAN PREDICTION -------------------
        # Fourier Ptychography GAN params
        # ptychlr = tf.keras.optimizers.schedules.InverseTimeDecay(
        #     tf.constant(ptychlr),
        #     decay_steps=100,
        #     decay_rate=0.5)

        # Fourier Ptychography GAN optimizers
        opt_x = get_optimizer(ptychlr, 'adam')
        opt_w = get_optimizer(0.0001, 'adam')
        # run Fourier Ptychography GAN algorithm
        ptychvars = planewaveGAN(G, opt_x, Pvec, gridtouse, f_rec, Gvars['Ginputs'][0],
                                lambda_=lambdaptych, k_vec=Gvars['wavenumber_vec'], max_itr=ptych_iters,
                                complex_net= use_complex_net, conditional_net = conditional_net,
                                optimizer_theta= opt_w,
                                verbose = not recon_full)

        del opt_x
        # Infer Fourier Ptychography GAN pressure
        Ppred3, H = infer_sf(G, ptychvars, f_rec, gridtrue, ptych=True)

        # ------------------ Ridge regression PREDICTION ----------------------
        # Get sensing matrix for Ridge and Lasso regression
        Hr, k = get_sensing_mat(f_rec, G.output.shape[1], gridtouse[0], gridtouse[1], gridtouse[2])

        if least_squares:
            # squeeze from tensor to numpy
            Hridge = np.squeeze(Hr.numpy())
            # split into real + imaginary
            Hridge = stack_real_imag_H(Hridge)
            # Squeeze pressure from tensor + seperate pressure into [real, imaginary]
            Ppvec = np.squeeze(Pvec)
            Pm_ls = np.concatenate((Ppvec.real, Ppvec.imag))
            # Run regression algorithm
            qridge, alphas = Ridge_regression(Hridge, Pm_ls, G.output.shape[1])
            qlass, alphas_lass = Lasso_regression(Hridge, Pm_ls, G.output.shape[1])
            # qridge, alphas = Lasso_regression(Hridge, Pm_ls, G.output.shape[1])
            print("Lasso reg param: ", alphas_lass)
            print("Ridge reg param: ", alphas)
            # Get projection sensing matrix (project onto reference plane)
            Hr_extr, _ = get_sensing_mat(f_rec, G.output.shape[1], gridtrue[0], gridtrue[1], gridtrue[2], k_samp=k)
            Hridge_extr = np.squeeze(Hr_extr.numpy())

            # predict pressure p = Hx
            P_ridge = Hridge_extr @ qridge
            P_lass = Hridge_extr @ qlass

        # normalise CSGAN
        Ppred = Ppred.numpy()
        p_csgan_glob.append(Ppred)

        # Ppred, _ = normalize_pressure(Ppred)

        # normalise adaptive CSGAN
        Ppred2 = Ppred2.numpy()
        # Ppred2, _ = normalize_pressure(Ppred2)
        p_adcsgan_glob.append(Ppred2)

        # normalise adaptive Fourier Ptychography GAN
        Ppred3 = Ppred3.numpy()
        p_ptychcsgan_glob.append(Ppred3)
        # normalise Ridge regression
        if least_squares:
            # P_ridge, _ = normalize_pressure(P_ridge)
            # P_ridge = P_ridge * p_norm
            p_ridge_glob.append(P_ridge)
            # P_lass, _ = normalize_pressure(P_lass)
            # P_lass = P_lass * p_norm
            p_lass_glob.append(P_lass)

        p_norm_const.append(p_norm)
        p_true.append(Ptrue)
        p_meas.append(Pm)
        # Ptrue = Ptrue * p_norm
        # metric calculation

        if calculate_metrics:
            mse_csgan = round(nmse(Ptrue, Ppred), 2)
            cos_dist_csgan = round(cos_sim(Ptrue, Ppred), 2)

            mse_ad_csgan = round(nmse(Ptrue, Ppred2), 2)
            cos_dist_ad_csgan = round(cos_sim(Ptrue, Ppred2), 2)

            mse_ptych = round(nmse(Ptrue, Ppred3), 2)
            cos_dist_ptych = round(cos_sim(Ptrue, Ppred3), 2)

            if least_squares:
                mse_ridge = round(nmse(Ptrue, P_ridge), 2)
                cos_dist_ridge = round(cos_sim(Ptrue, P_ridge), 2)
                mse_lass = round(nmse(Ptrue, P_lass), 2)
                cos_dist_lass = round(cos_sim(Ptrue, P_lass), 2)

            mse_csgan_glob.append(mse_csgan)
            mac_csgan_glob.append(cos_dist_csgan)
            mse_ad_csgan_glob.append(mse_ad_csgan)
            mac_ad_csgan_glob.append(cos_dist_ad_csgan)
            mse_ptych_csgan_glob.append(mse_ptych)
            mac_ptych_csgan_glob.append(cos_dist_ptych)

            if least_squares:
                mse_ridge_glob.append(mse_ridge)
                mac_ridge_glob.append(cos_dist_ridge)
                mse_lass_glob.append(mse_lass)
                mac_lass_glob.append(cos_dist_lass)
            # print metrics NMSE + MAC
            print("CS GAN NMSE: {}".format(mse_csgan))
            print("CS Adaptive GAN NMSE: {}".format(mse_ad_csgan))
            print("CS Ptych NMSE: {}".format(mse_ptych))
            # print("CS LBGFS NMSE: {}".format(mse_lbgfs))

            if least_squares:
                print("Ridge Regression NMSE: {}".format(mse_ridge))
                print("Lasso Regression NMSE: {}".format(mse_lass))
                print("Ridge Regression MAC: {}".format(cos_dist_ridge))
                print("Lasso Regression MAC: {}".format(cos_dist_lass))

            print("CS GAN MAC: {}".format(cos_dist_csgan))
            print("CS Adaptive GAN MAC: {}".format(cos_dist_ad_csgan))
            print("CS Ptych MAC: {}".format(cos_dist_ptych))

        # end single loop
        t.set_description("FreqNo: {} f: {}".format(kk + iter, f_rec), refresh=True)

        if plot_sfs:
            width = 6.694
            maxnorm = max(abs(Ptrue.real.min()), abs(Ptrue.real.max()))
            norm = (-maxnorm, maxnorm)
            # Evaluate sensing matrix
            hmut = tf.squeeze(adaptCSGANvars['H']).numpy()
            mu, gram = MutualCoherence(hmut)
            print("mutual coherence of sensing mat H: ", mu)

            fig, ax = plt.subplots(1, 1, figsize=(width / 1.3, width / 3))
            ax.set_aspect('equal')
            coh, ax, sc = columnwise_coherence(hmut, k, True, ax=ax)

            divider = make_axes_locatable(ax)

            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar(sc, cax=cax, label='Correlation', extend='both')
            # fig.suptitle("Sensing Matrix columnwise coherence")
            fig.show()

            fig.savefig(results_dir + '/SenseMat_{}Hz.png'.format(f_rec), dpi=150)

            fig = plt.figure(figsize=(1.7 * width / 2, 6 * width / 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax, sc = plot_array_pressure(Parr, gridar, ax=ax, norm=norm,
                                         z_label=True, plane=True)
            cax = fig.add_axes([ax.get_position().x1 + 0.07, ax.get_position().y0, 0.02, ax.get_position().height])

            fig.colorbar(sc, fraction=0.046, pad=0.04, cax=cax, label='Normalised Sound Pressure [Pa]', extend='both')
            fig.show()

            fig.savefig(results_dir + '/FullArraypressure_{}Hz.png'.format(f_rec))

            fig = plt.figure(figsize=(1.7 * width / 2, 6 * width / 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax, sc = plot_array_pressure(Ptouse, gridtouse, ax=ax, norm=norm,
                                         z_label=True, plane=True)
            cax = fig.add_axes([ax.get_position().x1 + 0.07, ax.get_position().y0, 0.02, ax.get_position().height])

            fig.colorbar(sc, fraction=0.046, pad=0.04, cax=cax, label='Normalised Sound Pressure [Pa]', extend='both')

            fig.show()

            fig.savefig(results_dir + '/MaskedArraypressure_{}Hz.png'.format(f_rec))

            fig = plt.figure(figsize=(width / 1.3, 1 * width / 2))
            axd = fig.subplot_mosaic(
                """
                ABC
                """
            )

            axd['A'], sc = plot_array_pressure(Ptrue, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['A'])
            axd['A'].set_title('Ground Truth\n')
            axd['A'].set_aspect('equal')

            if least_squares:
                Pplot = P_ridge
                title = 'Regularised Least Squares'
                mse = mse_ridge
                mac = cos_dist_ridge
            else:
                Pplot = Ppred
                title = 'CS GAN'
                mse = mse_csgan
                mac = cos_dist_csgan
            axd['B'], sc = plot_array_pressure(Pplot, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['B'])
            axd['B'].set_title('{}\n'.format(title))
            axd['B'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse, mac), transform=axd['B'].transAxes,
                          fontsize=8, ha='center', va='center')

            # axd['B'].xaxis.set_visible(False)
            axd['B'].yaxis.set_visible(False)
            axd['B'].xaxis.set_visible(False)

            axd['B'].set_aspect('equal')

            axd['C'], sc = plot_array_pressure(P_lass, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['C'])
            axd['C'].set_aspect('equal')

            axd['C'].set_title('Lasso\n')
            axd['C'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse_lass, cos_dist_lass),
                          transform=axd['C'].transAxes,
                          fontsize=8, ha='center', va='center')

            axd['C'].yaxis.set_visible(False)

            cbaxes = fig.add_axes([0.425, 0.15, 0.26, 0.05])

            fig.colorbar(sc, cbaxes, orientation='horizontal', label='Normalised Sound Pressure [Pa]', extend='both')

            fig.show()
            fig.savefig(results_dir + '/soundfields_{}Hz.png'.format(f_rec))

            fig = plt.figure(figsize=(1.7 * width / 2, 2.2 * width / 3))
            axd = fig.subplot_mosaic(
                """
                ABC
                D.F
                """
            )

            axd['A'], sc = plot_array_pressure(Ptrue, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['A'])
            axd['A'].set_aspect('equal')

            axd['A'].set_title('Ground Truth\n')
            # axd['A'].xaxis.set_visible(False)

            axd['B'], sc = plot_array_pressure(Pplot, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['B'])
            axd['B'].set_aspect('equal')

            axd['B'].set_title('{}\n'.format(title))
            axd['B'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse, mac), transform=axd['B'].transAxes,
                          fontsize=8, ha='center', va='center')

            # axd['B'].xaxis.set_visible(False)
            axd['B'].yaxis.set_visible(False)

            axd['C'], sc = plot_array_pressure(P_lass, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['C'])
            axd['C'].set_aspect('equal')

            axd['C'].set_title('Lasso\n')
            axd['C'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse_lass, cos_dist_lass),
                          transform=axd['C'].transAxes,
                          fontsize=8, ha='center', va='center')

            # axd['C'].xaxis.set_visible(False)
            axd['C'].yaxis.set_visible(False)

            axd['D'], sc = plot_array_pressure(Ppred2, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['D'])
            axd['D'].set_aspect('equal')

            axd['D'].set_title('CSGAN\n')
            axd['D'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse_ad_csgan, cos_dist_ad_csgan),
                          transform=axd['D'].transAxes,
                          fontsize=8, ha='center', va='center')

            axd['F'], sc = plot_array_pressure(Ppred3, gridtrue,
                                               norm=norm,
                                               z_label=False,
                                               ax=axd['F'])
            axd['F'].set_aspect('equal')

            axd['F'].set_title('projection CSGAN\n')
            axd['F'].text(0.5, 1.04, 'NMSE: {} dB - MAC: {}'.format(mse_ptych, cos_dist_ptych),
                          transform=axd['F'].transAxes,
                          fontsize=8, ha='center', va='center')

            axd['F'].yaxis.set_visible(False)

            cbaxes = fig.add_axes([0.425, 0.3, 0.25, 0.05])

            fig.colorbar(sc, cbaxes, orientation='horizontal', label='Normalised Sound\n Pressure [Pa]', extend='both')

            # fig.colorbar(sc)
            fig.show()
            fig.savefig(results_dir + '/Allsoundfields_{}Hz.png'.format(f_rec))


        if store_predictions:
            if calculate_metrics:
                global_var_out['csgan_nmse'] = mse_csgan_glob
                global_var_out['csgan_mac'] = mac_csgan_glob
                global_var_out['ad_csgan_nmse'] = mse_ad_csgan_glob
                global_var_out['ad_csgan_mac'] = mac_ad_csgan_glob
                global_var_out['ptych_nmse'] = mse_ptych_csgan_glob
                global_var_out['ptych_mac'] = mac_ptych_csgan_glob
                if least_squares:
                    global_var_out['ridge_nmse'] = mse_ridge_glob
                    global_var_out['ridge_mac'] = mac_ridge_glob
                    global_var_out['lass_nmse'] = mse_lass_glob
                    global_var_out['lass_mac'] = mac_lass_glob

            iters.append(kk + iter)
            freq_vec.append(f_rec)
            global_var_out['iters'] = iters
            global_var_out['freq_vec'] = freq_vec
            global_var_out['p_csgan_glob'] = p_csgan_glob
            global_var_out['p_adcsgan_glob'] = p_adcsgan_glob
            global_var_out['p_ptychcsgan_glob'] = p_ptychcsgan_glob
            global_var_out['p_ridge_glob'] = p_ridge_glob
            global_var_out['p_lass_glob'] = p_lass_glob
            global_var_out['p_norm_const'] = p_norm_const
            global_var_out['p_true'] = p_true
            global_var_out['p_m'] = p_meas
            np.save(results_dir + '/' + model + '_inference_{}'.format(dataset), global_var_out)

if __name__ == "__main__":
    run_CSGAN()

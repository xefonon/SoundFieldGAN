import click
import numpy as np
import tensorflow as tf
import wandb
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from tensorflow.keras.models import save_model
from glob import glob
import yaml
import sys
sys.path.append('../')
from GANTrainingScripts.aux_functions import config_from_yaml, print_training_stats, bcolors
from GANTrainingScripts.sf_reconstruction_utils import reference_grid, sensing_mat_transfer_learning, \
    cmplx_to_array, array_to_complex
import datetime
from pathlib import Path
from VAE_model import create_sound_fields_dataset, VAE_SF, \
    sample_from_real_data, sample_from_generator, create_transfer_learning_dataset, train_step

def create_tf_dataset(data_dir):
    from glob import glob
    filenames = glob(data_dir + '/*.npz')
    assert filenames != [], 'Directory is empty or does not exist, ' \
                            'please provide the right directory'
    test_train_ratio = 0.8
    N = len(filenames)
    np.random.shuffle(filenames)
    files_train = filenames[:int(test_train_ratio * N)]
    files_test = filenames[int(test_train_ratio * N):]
    freqs_train, sf_train = [], []
    freqs_test, sf_test = [], []

    for f in files_train:
        with np.load(f) as data:
            # p_array = data['p_array']
            # grid_ref = data['grid_ref']
            # grid_array = data['grid_array']
            freqs_train.append(data['f'])
            sf_train.append(data['p_ref'])
    for f in files_test:
        with np.load(f) as data:
            # p_array = data['p_array']
            # grid_ref = data['grid_ref']
            # grid_array = data['grid_array']
            # p_ref = data['p_ref']
            # freq = data['f']
            freqs_test.append(data['p_ref'])
            sf_test.append(data['f'])

    train_dataset = tf.data.Dataset.from_tensor_slices((sf_train, freqs_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((sf_test, freqs_test))

    return train_dataset, test_dataset


def tf_data_generator(file_list, batch_size=20):
    i = 0
    while True:  # This loop makes the generator an infinite loop
        if i * batch_size >= len(file_list):
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i * batch_size:(i + 1) * batch_size]
            sf_train = []
            freqs_train = []
            # label_classes = tf.constant(["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"])
            for file in file_chunk:
                with np.load(file) as data:
                    # p_array = data['p_array']
                    # grid_ref = data['grid_ref']
                    # grid_array = data['grid_array']
                    freqs_train.append(data['f'])
                    sf_train.append(data['p_ref'])
            data = np.asarray(sf_train)
            labels = np.asarray(freqs_train)
            first_dim = data.shape[0]
            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            dataset = dataset.batch(batch_size=first_dim)

            yield dataset
            i = i + 1


def read_npy_file(filename):
    with np.load(filename.numpy().decode()) as data:
        # p_array = data['p_array']
        # grid_ref = data['grid_ref']
        # grid_array = data['grid_array']
        freqs_train = data['f']
        sf_train = data['p_ref']
    return [sf_train.astype(np.float32), freqs_train.astype(np.float32)]


def read_soundfield(filename):
    [sf, ff] = tf.py_function(read_npy_file, inp=[filename], Tout=[tf.float32, tf.float32])
    # data,label = tf.py_function(data_preprocessing,[image],[tf.float32,tf.float32])
    return sf, ff


def generate_data(file_path):
    list_ds = tf.data.Dataset.list_files(file_path + '/*.npz')
    sf_ds = list_ds.map(read_soundfield, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return sf_ds


# list_ds = tf.data.Dataset.list_files(data_dir+ '/*.npz')
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot_sf_hist(pressure, frequencies, title_sup):
    cmp = cm.get_cmap('tab20c', 5)
    color = cmp(range(5))
    fig = plt.figure(constrained_layout=True)
    axd = fig.subplot_mosaic(
        [["p1", "p2", "p3"],
         ["p4", "BLANK", "p5"]
         ],
        empty_sentinel="BLANK",
    )
    spl = 20 * np.log10((abs(pressure) + 1e-12) / 2e-5)
    fig.suptitle('SPL distribution - {}'.format(title_sup))
    # fig.show()
    legend_list = list(map('f = {:.2f} Hz'.format, frequencies))
    spl = np.nan_to_num(spl)
    axd['p1'].hist(spl[0].flatten(), bins=100, density=True, color=color[0])
    axd['p1'].set_xlabel('SPL [dB]')
    axd['p1'].set_ylabel('Density')
    axd['p1'].legend([legend_list[0]])

    axd['p2'].hist(spl[1].flatten(), bins=100, density=True, color=color[1])
    axd['p2'].set_xlabel('SPL [dB]')
    axd['p2'].set_ylabel('Density')
    axd['p2'].legend([legend_list[1]])

    axd['p3'].hist(spl[2].flatten(), bins=100, density=True, color=color[2])
    axd['p3'].set_xlabel('SPL [dB]')
    axd['p3'].set_ylabel('Density')
    axd['p3'].legend([legend_list[2]])

    axd['p4'].hist(spl[3].flatten(), bins=100, density=True, color=color[3])
    axd['p4'].set_xlabel('SPL [dB]')
    axd['p4'].set_ylabel('Density')
    axd['p4'].legend([legend_list[3]])

    axd['p5'].hist(spl[4].flatten(), bins=100, density=True, color=color[4])
    axd['p5'].set_xlabel('SPL [dB]')
    axd['p5'].set_ylabel('Density')
    axd['p5'].legend([legend_list[4]])
    return fig


def plot_sfs(pressure, frequencies, title_sup):
    fig = plt.figure(constrained_layout=True)
    # widths = [1, 1, 1]
    # heights = [1, 3, 2]

    axd = fig.subplot_mosaic(
        [["p1", "p2", "p3"],
         ["p4", "BLANK", "p5"]
         ],
        empty_sentinel="BLANK"
    )
    spl = 20 * np.log10((abs(pressure) + 1e-12) / 2e-5)
    fig.suptitle('SPL distribution - {}'.format(title_sup))
    # fig.show()
    # ic(frequencies)
    title_list = list(map('f = {:.2f} Hz'.format, frequencies))

    min, max = spl.min(), spl.max()
    axd['p1'].imshow(spl[0], extent=[-.7, .7, -.7, .7], cmap='hot', origin='lower', vmin=min, vmax=max)
    axd['p1'].set_xlabel('x [m]')
    axd['p1'].set_ylabel('y [m]')
    axd['p1'].set_title(title_list[0])

    axd['p2'].imshow(spl[1], extent=[-.7, .7, -.7, .7], cmap='hot', origin='lower', vmin=min, vmax=max)
    axd['p2'].set_xlabel('x [m]')
    axd['p2'].set_ylabel('y [m]')
    axd['p2'].set_title(title_list[1])

    axd['p3'].imshow(spl[2], extent=[-.7, .7, -.7, .7], cmap='hot', origin='lower', vmin=min, vmax=max)
    axd['p3'].set_xlabel('x [m]')
    axd['p3'].set_ylabel('y [m]')
    axd['p3'].set_title(title_list[2])

    axd['p4'].imshow(spl[3], extent=[-.7, .7, -.7, .7], cmap='hot', origin='lower', vmin=min, vmax=max)
    axd['p4'].set_xlabel('x [m]')
    axd['p4'].set_ylabel('y [m]')
    axd['p4'].set_title(title_list[3])

    im = axd['p5'].imshow(spl[4], extent=[-.7, .7, -.7, .7], cmap='hot', origin='lower', vmin=min, vmax=max)
    axd['p5'].set_xlabel('x [m]')
    axd['p5'].set_ylabel('y [m]')
    axd['p5'].set_title(title_list[4])
    cbaxes = fig.add_axes([0.4, 0.3, 0.25, 0.05])

    fig.colorbar(im, cbaxes, orientation='horizontal', label='Sound Pressure Level [dB]', extend='both')
    return fig


@click.command()
# options_metavar='<options>'
@click.option('--epochs', default=100, type=int,
              help='Number of epochs to train the GAN for')
@click.option('--log_interval', default=50, type=int,
              help='Iteration steps at which to checkpoint model and log loss')
@click.option('--sample_interval', default=50, type=int,
              help='Iteration steps at which to plot FRFs')
@click.option('--resume_training', is_flag=True,
              help='Resume training from latest checkpoint in "Checkpoint_folder"')
@click.option('--use_wandb', is_flag=True,
              help='Use weights and biases to monitor training')
@click.option('--config_file', default='./config_vae.yaml', type=str,
              help='Configuration (.yaml) file including hyperparameters for training')
def run_VAE(epochs,
              log_interval,
              resume_training,
              sample_interval,
              config_file,
              use_wandb):
    if use_wandb:
        date = datetime.date.today().strftime('%m-%d')
        time = datetime.datetime.now().strftime("%H:%M")
        print("Using Weights and Biases to track training!")
        wandb.login()
        config_dict = yaml.load(open(config_file), Loader=yaml.FullLoader)
        run = wandb.init(project='Plane_wave_VAE_training',
                         name=config_dict['model_name']['value'] + date + '_' + time,
                         config=config_file,
                         group = config_dict['model_name']['value'],
                         notes = config_dict['notes']['value'],
                         save_code= True)
        config = wandb.config
        print(config)
    else:
        config = config_from_yaml(config_file)
    # init params
    config.epochs = epochs
    sound_field_shape = [config.grid_dimension, config.grid_dimension, 1] # complex valued shape

    latent_dim = config.latent_dim
    fs = config.sample_rate
    freq_vector = np.fft.rfftfreq(n=config.NFFT, d=1 / fs)

    # checkpoint directory
    check_dir = './' + config.model_name + '_chkpnts'
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    if not config.transfer_learning:
        total_n_fields = 60000
        batched_dataset = create_sound_fields_dataset(sampling_rate=config.sample_rate,
                                                      N_fft_size=config.NFFT,
                                                      avg_snr=config.avg_snr,
                                                      n_plane_waves=config.n_plane_waves,
                                                      grid_dimension=config.grid_dimension,
                                                      batch_size=config.batch_size,
                                                      epochs=config.epochs,
                                                      n_fields=total_n_fields,
                                                      augment = config.augmented_dset,
                                                      config = config)
        total_batches = total_n_fields // config.batch_size
        temp_grid = reference_grid(config.grid_dimension, xmin=-.5,
                                   xmax=.5)
    else:
        batched_dataset, temp_grid, total_n_fields = create_transfer_learning_dataset(data_dir=config.real_data_dir,
                                                                                      batch_size=config.batch_size,
                                                                                      epochs=config.epochs)
        total_batches = total_n_fields // config.batch_size
    # TODO: config.activation, config.decoder_bias, config.decoder_bias, config.lr,
    #  config.decoder_final_activation,
    vae = VAE_SF(latent_dim=latent_dim,
                 kl_weight=config.kl_weight,
                 activation = config.activation,
                 decoder_bias = config.decoder_bias,
                 decoder_final_activation = config.decoder_final_activation
                 )
    # optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)

    # Checkpoint manager (resume training / load model / save model / etc.)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               vae=vae,
                               optimizer=optim)

    manager = tf.train.CheckpointManager(ckpt, check_dir, max_to_keep=3)

    if resume_training:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(80 * '*')
            print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}Restored from {manager.latest_checkpoint}")
            print(80 * f'{bcolors.ENDC}*')

    else:
        print(80 * '*')
        print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}Initializing from scratch.")
        print(80 * f'{bcolors.ENDC}*')

    epoch = ckpt.step // total_batches

    z_fixed = vae.get_latent(size = 5)
    networktitle = 'real valued VAE'

    my_file = Path('./' + check_dir + '/VAEloss_info.npz')
    if my_file.is_file():
        LossDict = np.load(my_file)
        recorded_vaeloss = list(LossDict['recorded_vaeloss'])
        step_list = list(LossDict['steps'])
        epochslist = list(LossDict['epochs'])
    else:
        recorded_vaeloss = []
        step_list = []
        epochslist = []
    print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}initialising training with {networktitle} network...")
    print(f"{bcolors.ENDC}")

    # metrics:
    total_loss = tf.keras.metrics.Mean("total_loss", dtype=tf.float32)
    kl_loss = tf.keras.metrics.Mean("kl_loss", dtype=tf.float32)
    reconstruction_loss = tf.keras.metrics.Mean("reconstruction_loss", dtype=tf.float32)

    # Training:
    for ii, data in enumerate(batched_dataset):
        if not config.transfer_learning:
            sfields, grid, H, frq = data
        else:
            sfields, frq = data
            grid = temp_grid
            H = sensing_mat_transfer_learning(grid, frq, temperature=17.1, n_plane_waves=config.n_plane_waves)
            sfields = tf.reshape(sfields, [config.batch_size, config.grid_dimension, config.grid_dimension])
            if not config.complex_network:
                sfields = cmplx_to_array(sfields)
            else:
                sfields = tf.expand_dims(sfields, -1)
        losses = train_step(sfields, vae, H, optim, reconstruction_loss, kl_loss, total_loss)
        print_training_stats(epoch, config.epochs, total_batches, losses, ii)
        if ii % log_interval == 0:
            save_path = manager.save()
            if use_wandb:
                wandb.log({"avg_recon_loss": float(reconstruction_loss.result().numpy()),
                           "avg_kl_loss": float(kl_loss.result().numpy()),
                           "avg_total_loss": float(total_loss.result().numpy())}, step=ii)
            recorded_vaeloss.append(total_loss.result().numpy())
            epochslist.append(epoch)
            step_list.append(ckpt.step)
            np.savez(my_file, recorded_vaeloss = recorded_vaeloss,
                     epochs = epochslist, steps = step_list)
        if ii % sample_interval == 0:
            if not config.transfer_learning:
                p_sampled_real, H_real, fc_real = sample_from_real_data(fs=config.sample_rate,
                                                                        Nfft=config.NFFT,
                                                                        n_waves=config.n_plane_waves,
                                                                        snr=config.avg_snr,
                                                                        grid_dim=config.grid_dimension,
                                                                        n_bands=len(z_fixed),
                                                                        config = config
                                                                        )
            else:
                indices = np.random.choice(np.arange(config.batch_size, dtype = int), 5, replace = False)
                p_sampled_real, H_real, fc_real = tf.gather(sfields, indices), tf.gather(H, indices), tf.gather(frq, indices)
                p_sampled_real = tf.complex(p_sampled_real[...,0], p_sampled_real[...,1])

            p_sampled_fake, fc_fake = sample_from_generator(model=vae,
                                                            grid=grid,
                                                            fs=config.sample_rate,
                                                            Nfft=config.NFFT,
                                                            H=H_real,
                                                            z_latent_vec=z_fixed,
                                                            latent_dim=config.latent_dim,
                                                            n_bands=len(z_fixed),
                                                            config = config
                                                            )
            if use_wandb:
                p_sampled_real = array_to_complex(p_sampled_real)
                if tf.is_tensor(fc_real):
                    fc_real = np.squeeze(fc_real.numpy(), -1)
                fig_g_hist = plot_sf_hist(p_sampled_fake.numpy().real,
                                          fc_fake, 'Generated')
                wandb.log({"vae_PDFs": wandb.Image(fig_g_hist)}, step=ii)
                if config.augmented_dset:
                    fig_real_hist = plot_sf_hist(p_sampled_real[-10:].real,
                                                 fc_real[-10:], 'Real')
                    wandb.log({"real_pressure_PDFs": wandb.Image(fig_real_hist)}, step=ii)
                    fig_real_sf = plot_sfs(p_sampled_real[-10:].real,
                                           fc_real[-10:], 'Real')
                    wandb.log({"real_sfs": wandb.Image(fig_real_sf)}, step=ii)

                fig_real_sf = plot_sfs(p_sampled_real.real,
                                       fc_real, 'Simulated')
                wandb.log({"simulated_sfs": wandb.Image(fig_real_sf)}, step=ii)

                fig_g_sf = plot_sfs(p_sampled_fake.numpy().real,
                                    fc_fake, 'Generated')
                wandb.log({"generated_sfs": wandb.Image(fig_g_sf)}, step=ii)

                plt.close('all')

        if ii % total_batches == 0:
            reconstruction_loss.reset_states()
            kl_loss.reset_states()
            total_loss.reset_states()
            save_path = os.path.join(check_dir, 'VAE model')
            save_model(vae, save_path, overwrite=True)
            print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}Saved model at epoch {epoch} as: {save_path}...")
            if ii != 0:
                epoch += 1


if __name__ == "__main__":
    print(80 * '-')
    print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}Initialising plane wave GAN model, please wait...")
    print(80 * f'{bcolors.ENDC}-')

    run_VAE()

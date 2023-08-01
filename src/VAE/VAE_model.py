""" Implementation based on tutorial at https://www.tensorflow.org/tutorials/generative/cvae"""
import numpy as np
import tensorflow as tf
tfkl = tf.keras.layers
from GANTrainingScripts.sf_reconstruction_utils import simulate_measurements, read_real_dataset
from aux_functions import find_nearest, get_centre_freq_octave_bands, get_sensing_mat


class Sampler_Z(tf.keras.layers.Layer):

    def call(self, inputs):
        """reparametrization trick"""
        mu, rho = inputs
        sd = tf.math.log(1 + tf.math.exp(rho))
        batch_size = tf.shape(mu)[0]
        dim_z = tf.shape(mu)[1]
        z_sample = mu + sd * tf.random.normal(shape=(batch_size, dim_z))
        return z_sample, sd


class Encoder_Z(tf.keras.layers.Layer):

    def __init__(self, dim_z, name="encoder", **kwargs):
        super(Encoder_Z, self).__init__(name=name, **kwargs)
        self.dim_x = (21, 21, 1)
        self.dim_z = dim_z
        self.conv_layer_1 = tfkl.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                        padding='valid', activation='relu')
        self.conv_layer_2 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                        padding='valid', activation='relu')
        self.flatten_layer = tfkl.Flatten()
        self.dense_mean = tfkl.Dense(self.dim_z, activation=None, name='z_mean')
        self.dense_raw_stddev = tfkl.Dense(self.dim_z, activation=None, name='z_raw_stddev')
        self.sampler_z = Sampler_Z()

    # Functional
    def call(self, x_input):
        z = self.conv_layer_1(x_input)
        z = self.conv_layer_2(z)
        z = self.flatten_layer(z)
        mu = self.dense_mean(z)
        rho = self.dense_raw_stddev(z)
        z_sample, sd = self.sampler_z((mu, rho))
        return z_sample, mu, sd


class Decoder_X(tf.keras.layers.Layer):

    def __init__(self, dim_z, name="decoder", activation='relu',
                 last_activation='tanh', use_bias=True, **kwargs):
        super(Decoder_X, self).__init__(name=name, **kwargs)
        self.dim_z = dim_z
        self.use_bias = use_bias
        self.activation = activation
        self.last_activation = last_activation
        self.dense_z_input = tfkl.Dense(16 * 16 * 32, activation=None)
        self.reshape_layer = tfkl.Reshape((16, 16, 32))
        self.conv_transpose_layer_1 = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                                                           padding='same', activation=self.activation,
                                                           use_bias=self.use_bias)
        self.conv_transpose_layer_2 = tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                                                           padding='same', activation=self.activation,
                                                           use_bias=self.use_bias)
        self.conv_transpose_layer_3 = tfkl.Conv2DTranspose(filters=2, kernel_size=3, strides=1,
                                                           padding='same',
                                                           use_bias=self.use_bias,
                                                           activation=self.last_activation)
        self.flatten = tfkl.Reshape((64 * 64, 2))

    # Functional
    def call(self, z):
        x_output = self.dense_z_input(z)
        x_output = self.reshape_layer(x_output)
        x_output = self.conv_transpose_layer_1(x_output)
        x_output = self.conv_transpose_layer_2(x_output)
        x_output = self.conv_transpose_layer_3(x_output)
        x_output = self.flatten(x_output)
        return x_output


class VAE_SF(tf.keras.Model):

    def __init__(self, latent_dim, kl_weight=1, name="autoencoder",
                 activation='relu', decoder_bias=True,
                 decoder_final_activation='tanh',
                 **kwargs):
        super(VAE_SF, self).__init__(name=name, **kwargs)
        self.dim_x = (21, 21, 1)
        self.latent_dim = latent_dim
        self.activation = activation
        self.decoder_final_activation = decoder_final_activation
        self.decoder_bias = decoder_bias
        self.encoder = Encoder_Z(dim_z=self.latent_dim)
        self.decoder = Decoder_X(dim_z=self.latent_dim,
                                 use_bias=self.decoder_bias,
                                 last_activation=self.decoder_final_activation)
        self.kl_weight = kl_weight

    def call(self, x_input):
        z_sample, mu, sd = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)

        # Σ_β [β x 16]
        kl_divergence = - 0.5 * tf.math.reduce_sum(1 + tf.math.log(
            tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd), axis=1)
        # kl_divergence = tf.math.reduce_mean(kl_divergence)
        # self.add_loss(lambda: self.kl_weight * kl_divergence)
        self.add_loss(self.kl_weight * kl_divergence)
        return x_recons_logits

    def get_latent(self, size=5):
        return tf.random.normal(shape=(size, self.latent_dim))

    def array_to_complex(self, arr):
        cmplx = tf.complex(arr[..., 0], arr[..., 1])
        return cmplx

    def cmplx_to_array(self, cmplx):
        real = tf.math.real(cmplx)
        imag = tf.math.imag(cmplx)
        arr = tf.concat([real, imag], axis=-1)
        return arr

# @tf.function
def train_step(x_true, model, H, optimizer, loss_metric_mse, loss_metric_kl, loss_metric_tot):
    with tf.GradientTape() as tape:
        H = H*H.shape[-1]
        x_recons_logits = model(x_true)
        batch_size, grid_dim1, grid_dim2, channels = x_true.shape
        fake_coefficients_cmplx = model.array_to_complex(x_recons_logits)
        fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
        fake_sound_fields = tf.reshape(fake_sound_fields, [batch_size] + [grid_dim1,
                                                                          grid_dim2,
                                                                          1])
        fake_sound_fields = model.cmplx_to_array(fake_sound_fields)

        cost = tf.math.squared_difference(x_true, fake_sound_fields)
        likelihood = .5*tf.reduce_mean(cost, axis=[1, 2, 3])
        kl_loss = model.losses[0]
        # kl_loss = tf.math.reduce_sum(model.losses)  # vae.losses is a list
        total_vae_loss = likelihood + kl_loss
    gradients = tape.gradient(total_vae_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric_mse(tf.reduce_mean(likelihood))
    loss_metric_kl(tf.reduce_mean(kl_loss))
    loss_metric_tot(total_vae_loss)
    return {
        "recon_loss": tf.reduce_mean(likelihood),
        "kl_loss": tf.reduce_mean(kl_loss),
        "total_vae_loss": tf.reduce_mean(total_vae_loss)}


def sample_from_generator(model, grid, grid_nd=21, fs=16000, Nfft=16384, H=None, z_latent_vec=None,
                          latent_dim=16,
                          n_bands=10, config=None):
    def array_to_complex(arr):
        return tf.complex(arr[..., 0], arr[..., 1])

    freq = np.fft.rfftfreq(Nfft, 1 / fs)
    frq = get_centre_freq_octave_bands(bands=n_bands)
    new_freqs = [find_nearest(freq, f) for f in frq]

    new_freqs = tf.convert_to_tensor(new_freqs)
    if z_latent_vec is None:
        z = tf.random.normal((n_bands, latent_dim), 0., 1.)
    else:
        z = z_latent_vec
    fake_coefficients = model.decoder(z)

    if H is None:
        H, _ = get_sensing_mat(new_freqs,
                               model.output.shape[1],
                               grid[0],
                               grid[1],
                               grid[2])

    fake_coefficients_cmplx = array_to_complex(fake_coefficients)
    fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
    fake_sound_fields = tf.reshape(fake_sound_fields, (n_bands, grid_nd, grid_nd))
    return fake_sound_fields, new_freqs


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
                                     deterministic=True)
    ps = []
    Hs = []
    for data in data_gen:
        pm, _, Hm, frq_tensor = data
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
        args=[f, n_plane_waves, avg_snr, 1, grid_dimension, n_fields,
              config.use_gaussian_prior, augment, config.real_data_dir
              ],
        output_types=(tf.float32, tf.float32, tf.complex64, tf.float32),
        output_shapes=((grid_dimension, grid_dimension, 2),
                       (3, np.prod((grid_dimension, grid_dimension))),
                       (np.prod((grid_dimension, grid_dimension)), n_plane_waves),
                       (1,))

    )
    # output_shapes=([21, 21], [3, 21*21], [21*21, 2000]))
    ds_series_batch = ds_series.repeat(epochs).batch(batch_size).prefetch(2 * batch_size)
    return ds_series_batch


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

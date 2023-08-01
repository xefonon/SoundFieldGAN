import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv2D, Dense, Conv2DTranspose, Conv1DTranspose,
                                     BatchNormalization, LayerNormalization)
from aux_functions import tf_sensing_mat, SpectralNormalization, instance_norm
from sf_reconstruction_utils import simulate_measurements, read_real_dataset, fib_sphere, sph_harm_all


# from icecream import ic


# datasets
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


def create_sound_fields_dataset(sampling_rate=16000,
                                N_fft_size=16384,
                                snr_lims=[10, 40],
                                n_plane_waves=3000,
                                grid_dimension=21,
                                batch_size=16,
                                epochs=100,
                                n_fields=80000,
                                freq_vector=None,
                                augment=False,
                                config=None,
                                sph_harm_order=14):
    if freq_vector is None:
        f = np.fft.rfftfreq(N_fft_size, 1 / sampling_rate)
    else:
        f = freq_vector
    # initialise spherical grid for spherical harmonics
    grid_sphere = tf.squeeze(fib_sphere(1000), 0)
    # initialise spherical harmonics
    Ybasis = sph_harm_all(nMax=sph_harm_order, grid=grid_sphere)
    ds_series = tf.data.Dataset.from_generator(
        simulate_measurements,
        args=[f, n_plane_waves, snr_lims, 1, grid_dimension, n_fields,
              config.use_gaussian_prior, augment, config.real_data_dir,
              config.complex_network, config.normalize_data, grid_sphere, Ybasis],
        output_types=(tf.float32, tf.float32, tf.complex64, tf.complex64, tf.float32, tf.complex64, tf.complex64),
        output_shapes=((grid_dimension, grid_dimension, 2),
                       (3, np.prod((grid_dimension, grid_dimension))),
                       (np.prod((grid_dimension, grid_dimension)), n_plane_waves),
                       (1000, n_plane_waves),
                       (1,),
                       (n_plane_waves,),
                       # ((sph_harm_order + 1) ** 2))
                       ())
    )
    # output_shapes=([21, 21], [3, 21*21], [21*21, 2000]))
    ds_series_batch = ds_series.repeat(epochs).batch(batch_size).prefetch(2 * batch_size)
    return ds_series_batch


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
    if config is not None:
        if config.use_freq_label:
            fake_coefficients = generator([z, new_freqs], training=False)
        else:
            fake_coefficients = generator(z, training=False)
    else:
        fake_coefficients = generator(z, training=False)

    if H is None:
        H, _ = tf_sensing_mat(new_freqs,
                              generator.output.shape[1], grid)
        # H, _ = get_sensing_mat(new_freqs,
        #                        generator.output.shape[1],
        #                        grid[0],
        #                        grid[1],
        #                        grid[2])

    fake_coefficients_cmplx = array_to_complex(fake_coefficients)
    fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
    fake_sound_fields = tf.reshape(fake_sound_fields, (n_bands, grid_nd, grid_nd))
    return fake_sound_fields, new_freqs


def sample_from_real_data(fs=16000, Nfft=16384, n_waves=4096, snr_lims=[15, 40], grid_dim=21, n_bands=10, config=None):
    complex_to_array = lambda x: np.concatenate((x[..., None].real, x[..., None].imag), axis=-1)

    freq = np.fft.rfftfreq(Nfft, 1 / fs)
    frq = get_centre_freq_octave_bands(bands=n_bands)
    new_freqs = [find_nearest(freq, f) for f in frq]

    data_gen = simulate_measurements(new_freqs,
                                     n_plane_waves=n_waves,
                                     snr_lims=snr_lims,
                                     batchsize=1,
                                     grid_dim=grid_dim,
                                     deterministic=True)
    ps = []
    Hs = []
    for data in data_gen:
        pm, _, Hm, _, frq_tensor, pwcoeffs, _ = data
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


class Injective_Constraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self):
        self.s_sq = ()

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


def D_conv_block(x,
                 filters,
                 activation,
                 kernel_size=(4,4),
                 strides=(1, 1),
                 padding="same",
                 use_bn=False,
                 name='',
                 use_sn=False,
                 bias=False
                 ):
    if activation == 'lrelu':
        activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    if use_sn:
        x = SpectralNormalization(tf.keras.layers.Conv2D(filters=filters,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         activation=activation,
                                                         use_bias=bias,
                                                         name=name))(x)
    else:
        x = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            activation=activation,
            padding=padding,
            name=name,
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    if use_bn:
        # x = BatchNormalization(momentum=0.9, epsilon=0.00002)(x)
        x = instance_norm(x)
    return x


def get_coefficient_discriminator(input_shape, class_embedding=False, use_bn=False):
    # MLP discriminator network
    coeff_input = layers.Input(shape=input_shape, name='coefficients', dtype=tf.float64)
    if class_embedding:
        class_id = layers.Input(shape=[1], name='Class_Label')  # frequency vector index

        embedded_id = layers.Embedding(input_dim=8000, output_dim=256, name='Class_Embedding')(class_id)
        embedded_id = Dense(units=input_shape[0], name='Class_embedding_Dense',
                            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(embedded_id)
        embedded_id = layers.Reshape(target_shape=[input_shape[0]], name='Embedded_reshaped')(
            embedded_id)

        x = layers.Concatenate(axis=1)([coeff_input, embedded_id])
    else:
        x = coeff_input
    # build fully connected network with batch normalization
    x = Dense(units=512, name='Dense_1', kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dense(units=512, name='Dense_2', kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    # reshape for conv1d
    x = layers.Reshape(target_shape=[1, 512, 1], name='Reshape_1')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=16, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=16, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=16, strides=2, padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = instance_norm(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    # flatten
    x = layers.Flatten()(x)
    x = Dense(units=1, name='coeff_discrim_out', kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    if class_embedding:
        d_model = keras.models.Model([coeff_input, class_id], x, name="coeff_discriminator")
    else:
        d_model = keras.models.Model(coeff_input, x, name="coeff_discriminator")
    return d_model


def get_discriminator_model(input_shape, class_embedding=False, use_bn=False, use_sn=False):
    sf_input = layers.Input(shape=input_shape, name='Soundfield_Input', dtype=tf.float64)

    # embedded layer for class index
    if class_embedding:
        class_id = layers.Input(shape=[1], name='Class_Label')  # frequency vector index

        embedded_id = layers.Embedding(input_dim=8000, output_dim=256, name='Class_Embedding')(class_id)
        embedded_id = Dense(units=input_shape[0] * input_shape[1], name='Class_embedding_Dense',
                            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(embedded_id)
        embedded_id = layers.Reshape(target_shape=[input_shape[0], input_shape[1], 1], name='Embedded_reshaped')(
            embedded_id)

        x = layers.Concatenate(axis=3)([sf_input, embedded_id])
    else:
        x = sf_input
    x = D_conv_block(
        x,
        32,
        kernel_size=(4,4),
        strides=(1, 1),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_0'
    )
    x = D_conv_block(
        x,
        32,
        kernel_size=(4,4),
        strides=(2, 2),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_1'
    )

    x = D_conv_block(
        x,
        64,
        kernel_size=(4,4),
        strides=(2, 2),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_2'
    )
    x = D_conv_block(
        x,
        128,
        kernel_size=(4,4),
        strides=(2, 2),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_3'
    )
    x = D_conv_block(
        x,
        256,
        kernel_size=(4,4),
        strides=(2, 2),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_4'
    )
    x = D_conv_block(
        x,
        512,
        kernel_size=(4,4),
        strides=(2, 2),
        use_bn=use_bn,
        use_sn=use_sn,
        activation='lrelu',
        name='D_Conv1d_5'
    )
    x = layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dense(1, name='critic_out', kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(x)

    if class_embedding:
        d_model = keras.models.Model([sf_input, class_id], x, name="critic")
    else:
        d_model = keras.models.Model(sf_input, x, name="critic")

    return d_model


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


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start + x_len]
    x.set_shape([b, x_len, nch])

    return x


def upsample_block(x,
                   filters,
                   kernel_size=(2,2),
                   strides=2,
                   padding="same",
                   use_bn=False,
                   use_bias=False,
                   name='',
                   apply_injenction=True,
                   activity_constraint=None,
                   dropout_rate=0.0
                   ):
    if apply_injenction:
        kernel_constraint = Injective_Constraint()
    else:
        kernel_constraint = None
    x = Conv2DTranspose(filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        name=name,
                        use_bias=use_bias,
                        kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                        kernel_constraint=kernel_constraint,
                        activity_regularizer=activity_constraint)(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    if use_bn:
        # x = BatchNormalization(momentum=0.9, epsilon=0.00002)(x)
        x = instance_norm(x)
    return x


def upsample_block1d(x,
                     filters,
                     kernel_size=25,
                     strides=2,
                     padding="same",
                     use_bn=False,
                     use_bias=False,
                     name='',
                     activity_constraint=None,
                     apply_injenction=False
                     ):
    x = Conv1DTranspose(filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        name=name,
                        use_bias=use_bias,
                        kernel_initializer='glorot_uniform',
                        activity_regularizer=activity_constraint)(x)
    if use_bn:
        # x = BatchNormalization()(x)
        x = instance_norm(x)
    return x


def build_fc_generator(latent_dim,
                       use_bn=False,
                       class_embedding=False):
    relu = lambda z: tf.nn.relu(z)
    tanh = lambda z: tf.nn.tanh(z)
    if not class_embedding:
        first_layer_size = 512
    else:
        first_layer_size = 480
    latent = layers.Input(shape=(latent_dim,), name='latent_z', dtype=tf.float64)
    noise = layers.Reshape(target_shape=(1, 1, latent_dim), input_shape=(latent_dim,), name='reshape_latent')(latent)
    x = layers.Dense(first_layer_size, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                     use_bias=False,
                     name='InputTConv2D')(noise)
    if class_embedding:
        x = relu(x)
        class_id = layers.Input(shape=(1,), name='Class_Label')  # frequency vector index
        # embedded layer for class index
        embedded_id = layers.Embedding(input_dim=8000, output_dim=256, name='Class_Embedding')(class_id)
        embedded_id = Dense(units=32, name='Class_embedding_Dense',
                            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                            use_bias=False)(embedded_id)
        inputs = layers.Concatenate(axis=3, name='Concat_input_embed')([x, tf.expand_dims(embedded_id, axis=1)])
    else:
        inputs = x
        class_id = None
    if use_bn:
        # inputs = BatchNormalization()(x)
        inputs = instance_norm(x)
    inputs = relu(inputs)
    for i in range(10, 12):
        if i == 10:
            x_ = inputs
        x_ = Dense(2 ** i, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                   use_bias=True,
                   name=f'Dense_{i - 9}')(x_)
        if use_bn:
            x_ = instance_norm(x_)
            # x_ = LayerNormalization()(x_)
        x_ = relu(x_)
    # two final layers (double size of coefficients so you have equal sizes of real and imag)
    x_ = Dense(4096, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
               use_bias=True,
               name=f'Dense_pre_upsample')(x_)  # layer final 1
    if use_bn:
        x_ = instance_norm(x_)
        # x_ = LayerNormalization()(x_)
    x_ = relu(x_)

    x_ = Dense(8192, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
               use_bias=True,
               name=f'Dense_final')(x_)  # layer final 2
    if use_bn:
        x_ = instance_norm(x_)
        # x_ = LayerNormalization()(x_)
    # x_ = tanh(x_)

    x_re, x_im = tf.split(x_, num_or_size_splits=2, axis=-1, name='Split_Real_Imag')
    coefficients = layers.Concatenate(axis=-1, name='Concat_out')([x_re[..., None], x_im[..., None]])
    coefficients_out = layers.Reshape((64 * 64, 2))(coefficients)
    return latent, class_id, coefficients_out


def build_conv_generator(latent_dim,
                         apply_injective_constraint=True,
                         use_bn=False,
                         activations='leaky_relu',
                         class_embedding=False,
                         last_layer_activation='tanh',
                         use_sn=False
                         ):
    if activations == 'relu':
        activation = lambda z: tf.nn.relu(z)
    else:
        activation = lambda z: tf.nn.leaky_relu(z)
    tanh = lambda z: tf.nn.tanh(z)

    latent = layers.Input(shape=(latent_dim,), name='Latent_Z', dtype=tf.float64)
    # noise = layers.Reshape(target_shape=(1, 1, latent_dim), input_shape=(latent_dim,), name='reshape_latent')(latent)
    noise = latent
    x = layers.Dense(4 * 4 * 1024, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))(noise)
    x = layers.Reshape(target_shape=(4, 4, 1024), input_shape=(4 * 4 * 1024,), name='reshape_latent')(x)
    # x = layers.Conv2DTranspose(1024, kernel_size=(1,1), strides=1,
    #                            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
    #                            use_bias=False,
    #                            name='InputTConv2D')(noise)
    if class_embedding:
        x = activation(x)
        class_id = layers.Input(shape=(1,), name='Class_Label')  # frequency vector index
        # embedded layer for class index
        embedded_id = layers.Embedding(input_dim=8000, output_dim=256, name='Class_Embedding')(class_id)
        embedded_id = Dense(units=4 * 4, name='Class_embedding_Dense',
                            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                            use_bias=False)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(4, 4, 1), name='reshape_embedded')(embedded_id)
        inputs = layers.Concatenate(axis=3, name='Concat_input_embed')([x, embedded_id])
    else:
        inputs = x
        class_id = None
    if use_bn:
        # x = BatchNormalization()(inputs)
        x = instance_norm(inputs)
    else:
        x = inputs
    x = activation(x)

    x = upsample_block(
        x,
        512,
        strides=2,
        use_bn=use_bn,
        padding="same",
        name='G_ConvT1',
        apply_injenction=apply_injective_constraint,
        dropout_rate=0.05
    )
    x = activation(x)

    x = upsample_block(
        x,
        256,
        strides=2,
        use_bn=use_bn,
        padding="same",
        name='G_ConvT2',
        apply_injenction=apply_injective_constraint,
        dropout_rate=0.
    )
    x = activation(x)

    x = upsample_block(
        x,
        128,
        strides=2,
        use_bn=use_bn,
        padding="same",
        name='G_ConvT3',
        apply_injenction=apply_injective_constraint,
        dropout_rate=0.
    )
    x = activation(x)

    coefficients_out = upsample_block(
        x, 2, strides=2, use_bn=use_bn, name='G_ConvT4', apply_injenction=False
    )

    # At this point, we have an output which has a shape of (64, 64, 2) - corresponds to complex coefficients
    # flatten them for output
    # coefficients_out = layers.Flatten()(coefficients_out)
    # coefficients_out = activation(coefficients_out)
    #
    # # dense layer
    # coefficients_out = layers.Dense(64 * 64 * 2, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
    #                                 use_bias=False,
    #                                 name='Dense_out')(coefficients_out)
    coefficients_out = layers.Reshape((64 * 64, 2))(coefficients_out)
    if last_layer_activation == 'tanh':
        coefficients_out = tanh(coefficients_out)

    return latent, class_id, coefficients_out


def get_generator_model(latent_dim,
                        apply_injective_constraint=True,
                        dense_generator=False,
                        use_label=False,
                        last_layer_activation='tanh',
                        use_spec_norm=False,
                        use_bn=False):
    if dense_generator:
        z, label, output = build_fc_generator(latent_dim, class_embedding=use_label)
    else:
        z, label, output = build_conv_generator(latent_dim, apply_injective_constraint,
                                                class_embedding=use_label,
                                                last_layer_activation=last_layer_activation,
                                                use_bn=use_bn,
                                                use_sn=use_spec_norm)

    if use_label:
        g_model = keras.models.Model([z, label], output, name="generator")
    else:
        g_model = keras.models.Model(z, output, name="generator")

    return g_model


class WGAN(keras.Model):
    def __init__(self,
                 discriminator,
                 generator,
                 latent_dim,
                 grid,
                 freq_vector,
                 discriminator_extra_steps=3,
                 gp_weight=10.0,
                 sound_field_shape=[32, 32, 2],
                 n_plane_waves=4096,
                 use_freq_embedding=False
                 ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.freq_vector = freq_vector
        self.sound_field_shape = sound_field_shape
        self.n_plane_waves = n_plane_waves
        self.use_freq_embedding = use_freq_embedding
        self.get_latent = lambda shape: tf.random.normal(shape, mean=0., stddev=1.)

        X, Y, Z = grid
        self.X = X
        self.Y = Y
        self.Z = Z
        self.grid = tf.reshape(grid, (3, -1))

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, d_adv_metric, g_adv_metric):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.g_adv_metric = g_adv_metric
        self.d_adv_metric = d_adv_metric

    def gradient_penalty(self, batch_size, real_sound_fields, fake_sound_fields, frequencies):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated sound fields
        and added to the discriminator loss.
        """
        # Get the interpolated sound fields
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        # alpha = tf.cast(alpha, tf.complex64)
        diff = fake_sound_fields - real_sound_fields
        interpolated = real_sound_fields + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated sound fields.
            if self.use_freq_embedding:
                pred = self.discriminator([interpolated, frequencies], training=True)
            else:
                pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated sound fields.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

        # norm = tf.norm(grads, axis=[1, 2, 3])  # keeps complex valued format (as opposed to above)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        # gp = tf.cast(gp, tf.complex64)
        return gp

    def array_to_complex(self, arr):
        cmplx = tf.complex(arr[..., 0], arr[..., 1])
        return cmplx

    def cmplx_to_array(self, cmplx):
        real = tf.math.real(cmplx)
        imag = tf.math.imag(cmplx)
        arr = tf.concat([real, imag], axis=-1)
        return arr

    def train_step(self, real_sound_fields, H, H_sph, frequencies, coefficients, sph_coefficients):
        # if isinstance(real_sound_fields, tuple):
        #     frequencies = real_sound_fields[1]
        #     real_sound_fields = tf.cast(real_sound_fields[0], tf.float32)
        # else:
        #     raise ValueError('You must provide discrete frequencies to use with batch')
        # Get the batch size

        batch_size = tf.shape(real_sound_fields)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.

        # random_frequencies = np.random.choice(self.freq_vector, size=(batch_size,))

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            # Get a batch of transfer matrices (plane wave kernels)
            # H, _ = get_sensing_mat(frequencies, 4096, self.X, self.Y, self.Z)

            with tf.GradientTape() as tape:
                # Generate 'fake' coefficients from the latent vector
                if self.use_freq_embedding:
                    fake_coefficients = self.generator([random_latent_vectors, frequencies], training=True)
                else:
                    fake_coefficients = self.generator(random_latent_vectors, training=True)
                # split into complex number
                fake_coefficients_cmplx = self.array_to_complex(fake_coefficients)
                # extrapolate sound fields
                fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
                # reshape to original sound field shape
                fake_sound_fields = tf.reshape(fake_sound_fields, [batch_size] + [self.sound_field_shape[0],
                                                                                  self.sound_field_shape[1],
                                                                                  1]
                                               )
                fake_sound_fields = self.cmplx_to_array(fake_sound_fields)
                # Get the logits for the fake sound fields
                if self.use_freq_embedding:
                    fake_logits = self.discriminator([fake_sound_fields, frequencies], training=True)
                    # Get the logits for the real sound fields
                    real_logits = self.discriminator([real_sound_fields, frequencies], training=True)
                else:
                    fake_logits = self.discriminator(fake_sound_fields, training=True)
                    # Get the logits for the real sound fields
                    real_logits = self.discriminator(real_sound_fields, training=True)
                # Calculate the discriminator loss using the fake and real sound field logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_sound_fields, fake_sound_fields, frequencies)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
                self.d_adv_metric.update_state(d_loss)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # ic(len(d_gradient))
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors_G = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Get frequency labels
        random_frequencies = np.random.choice(self.freq_vector, size=(batch_size,))
        # H_G, _ = get_sensing_mat(random_frequencies, self.n_plane_waves, self.X, self.Y, self.Z)
        H_G, _ = tf_sensing_mat(random_frequencies, self.n_plane_waves, self.grid)
        # ic(self.X.shape)
        with tf.GradientTape() as tape:
            # Generate fake soun fields using the generator
            if self.use_freq_embedding:
                generated_coeffs = self.generator([random_latent_vectors_G, random_frequencies], training=True)
            else:
                generated_coeffs = self.generator(random_latent_vectors_G, training=True)
            # split into complex number
            generated_coefficients_cmplx = self.array_to_complex(generated_coeffs)

            # extrapolate sound fields
            generated_sound_fields = tf.einsum('ijk, ik -> ij', H_G, generated_coefficients_cmplx)
            # reshape to original sound field shape
            G_cmplx_sf = tf.reshape(generated_sound_fields, [batch_size] + [self.sound_field_shape[0],
                                                                            self.sound_field_shape[1],
                                                                            1])
            generated_sound_fields = self.cmplx_to_array(G_cmplx_sf)
            # Get the discriminator logits for fake images
            if self.use_freq_embedding:
                gen_sf_logits = self.discriminator([generated_sound_fields, random_frequencies], training=True)
            else:
                gen_sf_logits = self.discriminator(generated_sound_fields, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_sf_logits)
            self.g_adv_metric(g_loss)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "avg_dloss": self.d_adv_metric,
            "avg_gloss": self.g_adv_metric,
            "g_examples": G_cmplx_sf[:5],
            "g_freq": random_frequencies[:5],
            "gradient_penalty": gp * self.gp_weight,
            "wasserstein_dist": d_cost}


def tf_cart2sph(x, y, z):
    """
    Convert cartesian to spherical coordinates
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :return: r, theta, phi
    """
    r = tf.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = tf.acos(z / r)
    phi = tf.atan2(y, x)
    return r, theta, phi


def sph_harm_tf(lmax, theta, phi):
    from scipy.special import sph_harm
    # Compute the spherical harmonics for each degree and order up to lmax
    Ylm = np.zeros(((lmax + 1) ** 2, len(theta)), dtype=np.complex128)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Ylm[idx, :] = sph_harm(m, l, phi, theta)
            idx += 1

    # Compute the coefficients by performing a least-squares fit
    return tf.constant(Ylm, dtype=tf.complex64)


def batch_sph_fourier_tf(Ylm_batch, p):
    return tf.squeeze(tf.linalg.lstsq(Ylm_batch, tf.transpose(p, (0, 2, 1)), fast=True), -1)


class LSRGAN(keras.Model):
    def __init__(self,
                 discriminator,
                 generator,
                 latent_dim,
                 coeff_discriminator=None,
                 discriminator_extra_steps=3,
                 sound_field_shape=[32, 32, 2],
                 n_plane_waves=4096,
                 use_freq_embedding=False,
                 freq_vector=None,
                 grid_xy=None,
                 grid_yz=None,
                 grid_xz=None,
                 decay=0.999,
                 Nmax=14
                 ):
        super(LSRGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.coeff_discriminator = coeff_discriminator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.sound_field_shape = sound_field_shape
        self.n_plane_waves = n_plane_waves
        self.use_freq_embedding = use_freq_embedding
        self.freq_vector = freq_vector
        self.grid_xy = tf.reshape(grid_xy, (3, -1))
        self.grid_yz = tf.reshape(grid_yz, (3, -1))
        self.grid_xz = tf.reshape(grid_xz, (3, -1))
        self.grids = [self.grid_xy, self.grid_yz, self.grid_xz]
        self.get_latent = lambda shape: tf.random.normal(shape, mean=0., stddev=1.)
        self.decay = decay
        self.sph_grid = tf.squeeze(fib_sphere(1000))
        self.Ylm = sph_harm_all(nMax=Nmax, grid=self.sph_grid)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, d_adv_metric, g_adv_metric):
        super(LSRGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.g_adv_metric = g_adv_metric
        self.d_adv_metric = d_adv_metric

    def array_to_complex(self, arr):
        cmplx = tf.complex(arr[..., 0], arr[..., 1])
        return cmplx

    def cmplx_to_array(self, cmplx):
        real = tf.math.real(cmplx)
        imag = tf.math.imag(cmplx)
        arr = tf.concat([real, imag], axis=-1)
        return arr

    # def relativistic_discriminator_loss(self, real_outputs, fake_outputs):
    #     real_output_avg = tf.reduce_mean(real_outputs, axis=0)
    #     fake_output_avg = tf.reduce_mean(fake_outputs, axis=0)
    #     Real_Fake_relativistic_average_out = real_outputs - fake_output_avg
    #     Fake_Real_relativistic_average_out = fake_outputs - real_output_avg
    #     # LS GAN
    #     d_loss = tf.reduce_mean(tf.math.square(Real_Fake_relativistic_average_out - 1)) + \
    #              tf.reduce_mean(tf.math.square(Fake_Real_relativistic_average_out + 1))
    #     # Hinge
    #     # d_loss = tf.reduce_mean(tf.maximum(1 - Real_Fake_relativistic_average_out, 0)) + \
    #     #          tf.reduce_mean(tf.maximum(1 + Fake_Real_relativistic_average_out, 0))
    #
    #     return d_loss
    #
    # def relativistic_generator_loss(self, real_outputs, fake_outputs):
    #     real_output_avg = tf.reduce_mean(real_outputs, axis=0)
    #     fake_output_avg = tf.reduce_mean(fake_outputs, axis=0)
    #     Real_Fake_relativistic_average_out = real_outputs - fake_output_avg
    #     Fake_Real_relativistic_average_out = fake_outputs - real_output_avg
    #     g_loss = tf.reduce_mean(tf.math.square(Fake_Real_relativistic_average_out - 1)) + \
    #              tf.reduce_mean(tf.math.square(Real_Fake_relativistic_average_out + 1))
    #     # g_loss = tf.reduce_mean(tf.maximum(1 - Fake_Real_relativistic_average_out, 0)) + \
    #     #          tf.reduce_mean(tf.maximum(1 + Real_Fake_relativistic_average_out, 0))
    #
    #     return g_loss
    def relativistic_discriminator_loss(self, discriminator_real_outputs,
                                        discriminator_gen_outputs,
                                        scope=None):
        """Relativistic Average GAN discriminator loss.
        This loss introduced in `The relativistic discriminator: a key element missing
        from standard GAN` (https://arxiv.org/abs/1807.00734).
        D_ra(x, y) = D(x) - E[D(y)]
        L = E[log(D_ra(real, fake))] - E[log(1 - D_ra(fake, real)]
        where D(x) and D(y) are discriminator logits, E[] represents the operation
        of taking average for all data in a mini-batch.
        Args:
          discriminator_real_outputs: Discriminator output on real data.
          discriminator_gen_outputs: Discriminator output on generated data. Expected
                                      to be in the range of (-inf, inf).
          scope: The scope for the operations performed in computing the loss.
        Returns:
          A loss Tensor.
        """
        with tf.compat.v1.name_scope(
                scope,
                'relativistic_discriminator_loss',
                values=[discriminator_real_outputs, discriminator_gen_outputs]):
            def get_logits(x, y):
                return x - tf.reduce_mean(y)

            real_logits = get_logits(discriminator_real_outputs,
                                     discriminator_gen_outputs)
            gen_logits = get_logits(discriminator_gen_outputs,
                                    discriminator_real_outputs)

            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits), logits=real_logits))
            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(gen_logits), logits=gen_logits))

        return real_loss + gen_loss

    def relativistic_generator_loss(self, discriminator_real_outputs,
                                    discriminator_gen_outputs,
                                    scope=None):
        """Relativistic Average GAN generator loss.
        This loss introduced in `The relativistic discriminator: a key element missing
        from standard GAN` (https://arxiv.org/abs/1807.00734).
        D_ra(x, y) = D(x) - E[D(y)]
        L = E[log(1 - D_ra(real, fake))] - E[log(D_ra(fake, real)]
        where D(x) and D(y) are discriminator logits, E[] represents the operation
        of taking average for all data in a mini-batch.
        Args:
          discriminator_real_outputs: Discriminator output on real data.
          discriminator_gen_outputs: Discriminator output on generated data. Expected
            to be in the range of (-inf, inf).
          scope: The scope for the operations performed in computing the loss.
        Returns:
          A loss Tensor.
        """
        with tf.compat.v1.name_scope(
                scope,
                'relativistic_generator_loss',
                values=[discriminator_real_outputs, discriminator_gen_outputs]):
            def get_logits(x, y):
                return x - tf.reduce_mean(y)

            real_logits = get_logits(discriminator_real_outputs,
                                     discriminator_gen_outputs)
            gen_logits = get_logits(discriminator_gen_outputs,
                                    discriminator_real_outputs)

            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(real_logits), logits=real_logits))
            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(gen_logits), logits=gen_logits))

        return real_loss + gen_loss

    def train_step(self, real_sound_fields, H, H_sph, frequencies, coefficients, sph_coefficients):
        # Get the batch size
        batch_size = tf.shape(real_sound_fields)[0]
        Ylm_batch = tf.tile(tf.expand_dims(self.Ylm, 0), [batch_size, 1, 1])
        coefficients = self.cmplx_to_array(coefficients)
        sph_coefficients = self.cmplx_to_array(sph_coefficients)
        # Generate the latent vector
        latent = self.get_latent((batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Generate 'fake' coefficients from the latent vector
            if self.use_freq_embedding:
                fake_coefficients = self.generator([latent, frequencies], training=True)
            else:
                fake_coefficients = self.generator(latent, training=True)
            # split into complex number
            fake_coefficients_cmplx = self.array_to_complex(fake_coefficients)
            # extrapolate sound fields
            fake_sound_fields = tf.einsum('ijk, ik -> ij', H, fake_coefficients_cmplx)
            fake_sph_sound_fields = tf.einsum('ijk, ik -> ij', H_sph, fake_coefficients_cmplx)
            if self.coeff_discriminator is not None:
                fake_sph_coeffs = batch_sph_fourier_tf(Ylm_batch, tf.expand_dims(fake_sph_sound_fields, 1))
                fake_sph_coeffs = self.cmplx_to_array(fake_sph_coeffs)
            # reshape to original sound field shape
            fake_sound_fields = tf.reshape(fake_sound_fields, [batch_size] + [self.sound_field_shape[0],
                                                                              self.sound_field_shape[1],
                                                                              1]
                                           )
            fake_sound_fields = self.cmplx_to_array(fake_sound_fields)

            d_cost = tf.constant(0.0)
            disc_trainable_variables = []
            # Get the logits for the fake sound fields
            if self.use_freq_embedding:
                if self.discriminator is not None:
                    fake_logits = self.discriminator([fake_sound_fields, frequencies], training=True)
                    # Get the logits for the real sound fields
                    real_logits = self.discriminator([real_sound_fields, frequencies], training=True)
                    d_cost += self.relativistic_discriminator_loss(real_logits, fake_logits)
                    disc_trainable_variables += self.discriminator.trainable_variables

                if self.coeff_discriminator is not None:
                    # D_coeff_input_fake = tf.reshape(fake_coefficients, [batch_size] + [2 * self.n_plane_waves])
                    # D_coeff_input_real = tf.reshape(coefficients, [batch_size] + [2 * self.n_plane_waves])
                    fake_coeff_logits = self.coeff_discriminator([fake_sph_coeffs, frequencies], training=True)
                    # Get the logits for the real sound fields
                    real_coeff_logits = self.coeff_discriminator([sph_coefficients, frequencies], training=True)
                    d_cost += self.relativistic_discriminator_loss(real_coeff_logits, fake_coeff_logits)
                    disc_trainable_variables += self.coeff_discriminator.trainable_variables
            else:
                if self.discriminator is not None:
                    fake_logits = self.discriminator(fake_sound_fields, training=True)
                    # Get the logits for the real sound fields
                    real_logits = self.discriminator(real_sound_fields, training=True)
                    d_cost += self.relativistic_discriminator_loss(real_logits, fake_logits)
                    disc_trainable_variables += self.discriminator.trainable_variables
                if self.coeff_discriminator is not None:
                    # D_coeff_input_fake = tf.reshape(fake_coefficients, [batch_size] + [2 * self.n_plane_waves])
                    # D_coeff_input_real = tf.reshape(coefficients, [batch_size] + [2 * self.n_plane_waves])
                    fake_coeff_logits = self.coeff_discriminator(fake_sph_coeffs, training=True)
                    # Get the logits for the real sound fields
                    real_coeff_logits = self.coeff_discriminator(sph_coefficients, training=True)
                    d_cost += self.relativistic_discriminator_loss(real_coeff_logits, fake_coeff_logits)
                    disc_trainable_variables += self.coeff_discriminator.trainable_variables
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_cost, disc_trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, disc_trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors_G = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Get frequency labels
        random_frequencies = np.random.choice(self.freq_vector, size=(batch_size,))
        grid_indx = np.random.choice([0, 1, 2])
        H_G, _ = tf_sensing_mat(random_frequencies, self.n_plane_waves, self.grids[grid_indx])
        # ic(self.X.shape)
        with tf.GradientTape() as tape:
            # Generate fake sound fields using the generator
            if self.use_freq_embedding:
                generated_coeffs = self.generator([random_latent_vectors_G, random_frequencies], training=True)
            else:
                generated_coeffs = self.generator(random_latent_vectors_G, training=True)
            # split into complex number
            generated_coefficients_cmplx = self.array_to_complex(generated_coeffs)

            # extrapolate sound fields
            generated_sound_fields = tf.einsum('ijk, ik -> ij', H_G, generated_coefficients_cmplx)
            generated_sph_sound_fields = tf.einsum('ijk, ik -> ij', H_sph, generated_coefficients_cmplx)
            g_sph_coeffs = batch_sph_fourier_tf(Ylm_batch, tf.expand_dims(generated_sph_sound_fields, 1))

            # reshape to original sound field shape
            G_cmplx_sf = tf.reshape(generated_sound_fields, [batch_size] + [self.sound_field_shape[0],
                                                                            self.sound_field_shape[1],
                                                                            1])
            generated_sound_fields = self.cmplx_to_array(G_cmplx_sf)
            g_sph_coeffs = self.cmplx_to_array(g_sph_coeffs)
            # Get the discriminator logits for fake sound fields
            g_loss = tf.constant(0.0)
            if self.use_freq_embedding:
                if self.discriminator is not None:
                    gen_sf_logits = self.discriminator([generated_sound_fields, random_frequencies], training=True)
                    real_logits = self.discriminator([real_sound_fields, frequencies], training=True)
                    g_loss += self.relativistic_generator_loss(real_logits, gen_sf_logits)
                if self.coeff_discriminator is not None:
                    # D_coeff_input_fake = tf.reshape(generated_coeffs, [batch_size] + [2 * self.n_plane_waves])
                    gen_coeff_logits = self.coeff_discriminator([g_sph_coeffs, random_frequencies], training=True)
                    real_coeff_logits = self.coeff_discriminator([sph_coefficients, frequencies], training=True)
                    g_loss += self.relativistic_generator_loss(real_coeff_logits, gen_coeff_logits)
            else:
                if self.discriminator is not None:
                    gen_sf_logits = self.discriminator(generated_sound_fields, training=True)
                    real_logits = self.discriminator(real_sound_fields, training=True)
                    g_loss += self.relativistic_generator_loss(real_logits, gen_sf_logits)
                if self.coeff_discriminator is not None:
                    # D_coeff_input_fake = tf.reshape(generated_coeffs, [batch_size] + [2 * self.n_plane_waves])
                    gen_coeff_logits = self.coeff_discriminator(g_sph_coeffs, training=True)
                    real_coeff_logits = self.coeff_discriminator(sph_coefficients, training=True)
                    g_loss += self.relativistic_generator_loss(real_coeff_logits, gen_coeff_logits)
            # Update the metrics tracking the loss
            self.g_adv_metric(g_loss)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {
            "d_loss": d_cost,
            "g_loss": g_loss,
            "avg_dloss": self.d_adv_metric,
            "avg_gloss": self.g_adv_metric,
            "g_examples": G_cmplx_sf[:5],
            "g_freq": random_frequencies[:5],
            "gradient_penalty": None,
            "wasserstein_dist": None}

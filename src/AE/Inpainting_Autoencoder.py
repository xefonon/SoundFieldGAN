from tensorflow.keras.layers import Input, Activation, \
    BatchNormalization, UpSampling2D, LeakyReLU, Conv2D,\
    Concatenate, Cropping2D
from tensorflow.keras import Model
from utils import PartialConv2D as PConv2D
import tensorflow as tf


def build_model(train_bn=True, input_size=(21, 21, 1)):
    """.
        Args:
        train_bn: boolean (optional)
        Returns: keras model, K.tensor
    """

    inputs_sf = Input(input_size,
                      name='inputs_sf')
    inputs_mask = Input(input_size,
                        name='inputs_mask')

    def encoder_layer(sf_in, mask_in, filters, kernel_size, bn=True, strides = 2):
        conv, mask = PConv2D(filters, kernel_size, strides=strides, padding='same',
                             name='encoder_partialconv_' + str(encoder_layer.counter))([sf_in, mask_in])
        if bn:
            conv = BatchNormalization(name='encoder_bn_' + str(encoder_layer.counter))(conv, training=train_bn)
        conv = Activation('relu')(conv)
        encoder_layer.counter += 1
        return conv, mask

    encoder_layer.counter = 0
    e_conv1, e_mask1 = encoder_layer(inputs_sf, inputs_mask, 64, 5, bn=False)
    e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 3)
    e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 3)
    e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3, strides = 1)

    def decoder_layer(sf_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True, strides = 2):
        up_sf = UpSampling2D(size=(strides, strides), name='upsampling_sf_' + str(decoder_layer.counter))(sf_in)
        up_mask = UpSampling2D(size=(strides, strides), name='upsampling_mk_' + str(decoder_layer.counter))(mask_in)
        if (e_conv.shape[1] % 2 != 0) & (e_conv.shape[1] > 10):
            up_sf = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None,
                               name='cropping_sf_' + str(decoder_layer.counter))(up_sf)
            up_mask = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None,
                               name='cropping_mk_' + str(decoder_layer.counter))(up_mask)
        concat_sf = Concatenate(axis=-1)([e_conv, up_sf])
        concat_mask = Concatenate(axis=-1)([e_mask, up_mask])
        conv, mask = PConv2D(filters, kernel_size, padding='same',
                             name='decoder_partialconv_' + str(decoder_layer.counter))([concat_sf, concat_mask])
        if bn:
            conv = BatchNormalization(name='encoder_bn_' + str(decoder_layer.counter))(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        decoder_layer.counter += 1
        return conv, mask

    decoder_layer.counter = encoder_layer.counter
    d_conv5, d_mask5 = decoder_layer(e_conv4, e_mask4, e_conv3, e_mask3, 256, 3, strides= 1)
    d_conv6, d_mask6 = decoder_layer(d_conv5, d_mask5, e_conv2, e_mask2, 128, 3)
    d_conv7, d_mask7 = decoder_layer(d_conv6, d_mask6, e_conv1, e_mask1, 64, 3)
    d_conv8, d_mask8 = decoder_layer(d_conv7, d_mask7, inputs_sf, inputs_mask, 1, 3, bn=False)
    outputs = Conv2D(1, 1, activation='sigmoid', name='outputs_sf')(d_conv8)

    # Setup the model inputs / outputs
    model = Model(inputs=[inputs_sf, inputs_mask], outputs=outputs)

    return model, inputs_mask


l1 = lambda y, x: tf.reduce_mean(tf.abs(y - x), axis=[1, 2, 3])


def loss_hole(mask, y_true, y_pred):
    """ Computes L1 loss within the mask.
    Args:
        mask: K.tensor
        y_true: K.tensor
        y_pred: K.tensor
    Returns: K.tensor
    """
    return l1((1 - mask) * y_true, (1 - mask) * y_pred)


def loss_valid(mask, y_true, y_pred):
    """ Computes L1 loss outside the mask
    Args:
        mask: K.tensor
        y_true: K.tensor
        y_pred: K.tensor
    Returns: K.tensor
    """
    return l1(mask * y_true, mask * y_pred)


def loss_total(mask, y_true, y_pred):
    return loss_valid(mask, y_true, y_pred) + 12*loss_hole(mask, y_true, y_pred)

@tf.function
def train_step(y_true, y_masked, mask, model, optimizer, loss_metric):
    with tf.GradientTape() as tape:
        y_hat = model([y_masked, mask])
        # batch_size, grid_dim1, grid_dim2, channels = y_true.shape

        cost = loss_total(mask, y_true, y_hat)

    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(tf.reduce_mean(cost))
    return {
        "total_loss": tf.reduce_mean(cost)}


G, masks = build_model(input_size=(21, 21, 2))

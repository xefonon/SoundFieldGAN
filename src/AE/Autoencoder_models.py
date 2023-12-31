""" Models based on https://github.com/polimi-ispl/nah-khcnn/blob/main/src/network_module/cnn_models.py"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, \
    Conv2DTranspose, Concatenate

def cnn_model_1024(input_layer, start_neurons, out_channel=1, name=""):
    # *** ENCODER ***
    x1, x2, x3, x4, encoder = encoder_1024(input_layer, start_neurons)

    # *** MIDDLE ***
    center = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same', name='CENTER_' + name)(encoder)
    center = BatchNormalization()(center)

    # *** DECODER ***

    # decoder_1 = decoder_1024(center, start_neurons, x4, x3, x2, x1)
    # v_s_output_imgs = Conv2D(out_channel, (1, 1), padding="same", activation="linear", name='v_s_output_imgs_' + name)(decoder_1)
    #
    decoder_2 = decoder_1024(center, start_neurons, x4, x3, x2, x1)
    p_s_output_imgs = Conv2D(out_channel, (1, 1), padding="same", activation="linear", name='p_s_output_imgs_' + name)(decoder_2)

    return  p_s_output_imgs


def cnn_model_64(input_layer, start_neurons, out_channel=1, name=""):
    # *** ENCODER ***

    x1, x2, x3, encoder = encoder_64(input_layer, start_neurons)

    # *** MIDDLE ***
    center = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same', name='CENTER_' + name)(encoder)
    center = BatchNormalization()(center)

    # *** DECODER ***

    decoder_1 = decoder_64(center, start_neurons, x3, x2, x1)
    v_s_output_imgs = Conv2D(out_channel, (1, 1), padding="same", activation="linear", name='v_s_output_imgs_' + name)(decoder_1)

    decoder_2 = decoder_64(center, start_neurons, x3, x2, x1)
    p_s_output_imgs = Conv2D(out_channel, (1, 1), padding="same", activation="linear", name='p_s_output_imgs_' + name)(decoder_2)

    return v_s_output_imgs, p_s_output_imgs


# encoder / decoder blocks
def encoder_1024(input_layer, start_neurons):
    x1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    pool1 = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    x2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    pool2 = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    x3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    pool3 = MaxPooling2D((2, 2))(x3)

    x4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    x4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    pool4 = MaxPooling2D((2, 2))(x4)

    return x1, x2, x3, x4, pool4


def decoder_1024(center, start_neurons, x4, x3, x2, x1):
    y4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(center)
    y4 = BatchNormalization()(y4)
    y4 = Concatenate()([y4, x4])  # keras.layers.add -> summation
    y4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(y4)
    y4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(y4)
    y4 = BatchNormalization()(y4)

    y3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(y4)
    y3 = BatchNormalization()(y3)
    y3 = Concatenate()([y3, x3])  # keras.layers.add -> summation
    y3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(y3)
    y3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(y3)
    y3 = BatchNormalization()(y3)

    y2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(y3)
    y2 = BatchNormalization()(y2)
    y2 = Concatenate()([y2, x2])  # keras.layers.add -> summation
    y2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(y2)
    y2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(y2)
    y2 = BatchNormalization()(y2)

    y1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(y2)
    y1 = BatchNormalization()(y1)
    y1 = Concatenate()([y1, x1])  # keras.layers.add -> summation
    y1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(y1)
    y1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(y1)
    y1 = BatchNormalization()(y1)

    return y1


def encoder_64(input_layer, start_neurons):
    # *** ENCODER ***

    x1 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    pool1 = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool1)
    x2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    pool2 = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool2)
    x3 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    pool3 = MaxPooling2D((2, 2))(x3)

    return x1, x2, x3, pool3


def decoder_64(center, start_neurons, x3, x2, x1):
    # *** DECODER ***

    y4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(center)
    y4 = BatchNormalization()(y4)
    y4 = Concatenate()([y4, x3])
    y4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(y4)
    y4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(y4)
    y4 = BatchNormalization()(y4)

    y3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(y4)
    y3 = BatchNormalization()(y3)
    y3 = Concatenate()([y3, x2])
    y3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(y3)
    y3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(y3)
    y3 = BatchNormalization()(y3)

    y2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(y3)
    y2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Concatenate()([y2, x1])

    y1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(1, 2), padding="same")(y2)
    y1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(y1)
    y1 = BatchNormalization()(y1)

    y0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(1, 2), padding="same")(y1)
    y0 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(y0)
    y0 = BatchNormalization()(y0)

    y = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(y0)
    y = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(y)
    y = BatchNormalization()(y)

    return y

# @tf.function
def train_step(y_true, y_masked, model, optimizer, loss_metric):
    with tf.GradientTape() as tape:
        y_hat = model(y_masked)
        batch_size, grid_dim1, grid_dim2, channels = y_true.shape

        cost = tf.reduce_mean(tf.math.squared_difference(y_true, y_hat), axis = [1,2,3])

    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(tf.reduce_mean(cost))
    return {
        "total_loss": tf.reduce_mean(cost)}

#
# input = tf.keras.layers.Input([21,21, 2])
# filters = 32
# cnn = tf.keras.Model(inputs = input , outputs = cnn_model_1024(input, filters, out_channel= 2))


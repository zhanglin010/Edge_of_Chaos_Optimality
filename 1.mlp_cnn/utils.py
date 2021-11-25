import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_fashion_data():
    """
    Get fashion mnist data from keras.datasets and flatten it
    """
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    x_train = np.reshape(
        x_train, [x_train.shape[0], -1]).astype('float32')/255.
    x_val = np.reshape(x_val, [x_val.shape[0], -1]).astype('float32')/255.

    return (x_train, y_train), (x_val, y_val)


def get_fashion_data2():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def get_cifar10_data(args):
    """
    Get Cifar10 data from keras.datasets
    """
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')/255.
    x_val = x_val.astype('float32')/255.
    if args.subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_val -= x_train_mean
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

    return (x_train, y_train), (x_val, y_val)


def get_datagen(x_train):
    # This will do preprocessing and realtime data augmentation:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    return datagen

def get_datagen2(x_train):
    # This will do preprocessing and realtime data augmentation:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train, seed=0)

    return datagen

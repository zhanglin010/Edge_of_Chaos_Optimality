import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


def get_cifar10_data1(args):
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


def get_cifar10_data2(args):
    """
    Get Cifar10 data from keras.datasets
    """
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

    return (x_train, y_train), (x_val, y_val)


def get_datagen1(x_train):
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
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
    datagen = ImageDataGenerator(rotation_range=15,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train, seed=0)

    return datagen


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    x *= 0.017  # scale values

    return x

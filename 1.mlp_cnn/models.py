import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, Input, concatenate, Lambda
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def mlp(args):
    '''Get a mlp model
    '''
    activation_func = tf.nn.relu
    model = Sequential([
        Dense(784, activation=activation_func, input_shape=(784,)),
        Dense(100, activation=activation_func),
        Dense(784, activation=activation_func),
        Dense(10, activation=tf.nn.softmax)
    ])

    return model


def cnn_3(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 4)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_7(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 8)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_9(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 16)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_without_pooling(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 4)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_with_aug(args):
    '''Get a cnn model without dropout layers
    '''
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 4)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def cnn_dropout(args):
    '''Get a cnn model with dropout layers
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 4)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

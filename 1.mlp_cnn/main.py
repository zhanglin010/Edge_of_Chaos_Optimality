# TensorFlow and other packages
import argparse
import os
import pickle
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

import loggingreporter
import models
import plot_figure2a
import plot_figure2b
import plot_figure_s1_2
import plot_figure_s7_8
import utils


# Training settings
parser = argparse.ArgumentParser(description='Jacobian study')
parser.add_argument('--architecture', type=str, default='mlp_0',
                    help='architecture of neural networks')
parser.add_argument('--activation-func', type=str, default='relu',
                    help='activation function for hidden layers')
parser.add_argument('--epochs', default=51, type=int, metavar='N',
                    help='number of total epochs to run, should > 3')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='batch size for training')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer used for training')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD')
parser.add_argument('--beta-1', default=0.9, type=float, metavar='M',
                    help='beta_1 in Adam')
parser.add_argument('--beta-2', default=0.999, type=float, metavar='M',
                    help='beta_2 in Adam')
parser.add_argument('--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')

parser.add_argument('--num-iterations', type=int, default=100,
                    help='asymptotic iterations that activations will be saved')
parser.add_argument('--num-repeats', type=int, default=10,
                    help='number of simulation repeats')


def main():
    start_time = time.time()
    args = parser.parse_args()
    args.log_epochs = np.arange(args.epochs)
    # Get the dataset for different architectures
    if args.architecture == 'mlp':
        args.layer = 'dense_2'
        (x_train, y_train), (x_val, y_val) = utils.get_fashion_data()
        loss = 'sparse_categorical_crossentropy'
    elif args.architecture == 'cnn':
        args.layer = 'activation_4'
        args.subtract_pixel_mean = False
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data(args)
        print(x_train.shape)
        args.input_shape = x_train.shape[1:]
        loss = 'categorical_crossentropy'
    elif args.architecture == 'cnn_with_aug':
        args.layer = 'activation_4'
        args.subtract_pixel_mean = False
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data(args)
        print(x_train.shape)
        args.input_shape = x_train.shape[1:]
        loss = 'categorical_crossentropy'
        datagen = utils.get_datagen2(x_train)

    args.dir = '{}/{}_{}/{}_{}_{}'.format(args.architecture, args.optimizer,
                                          args.momentum, args.activation_func,
                                          args.epochs, args.num_iterations)
    for num_repeat in range(args.num_repeats):
        # break
        # num_repeat += 12
        print('num_repeat={}'.format(num_repeat))
        args.save_lyapunov0s_dir = 'rawdata/lyapunov0s/{}/{}'.format(
            args.dir, num_repeat)
        args.save_lyapunov1s_dir = 'rawdata/lyapunov1s/{}/{}'.format(
            args.dir, num_repeat)
        args.save_lyapunov2s_dir = 'rawdata/lyapunov2s/{}/{}'.format(
            args.dir, num_repeat)
        args.save_losses_dir = 'rawdata/losses/{}/{}'.format(
            args.dir, num_repeat)

        model = getattr(models, args.architecture)(args)
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        model.summary()
        intermediate_model = Model(inputs=model.input,
                                   outputs=model.get_layer(args.layer).output)
        reporter = loggingreporter.LoggingReporter(args, intermediate_model,
                                                   x_train, y_train,
                                                   x_val, y_val)
        if args.architecture == 'cnn_with_aug':
            model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                      epochs=args.epochs, verbose=0,
                      steps_per_epoch=math.ceil(
                          x_train.shape[0] / args.batch_size),
                      callbacks=[reporter, ])
        else:
            model.fit(x_train, y_train, epochs=args.epochs,
                      verbose=0, callbacks=[reporter, ])
        tf.keras.backend.clear_session()

    # Plot Fig.2(a). Replace plot_figure2a with plot_figure2b for Fig.2(b).
    plot_figure2a.plot_lyapunov1s(args)
    plot_figure2a.plot_acc(args)
    plot_figure2a.plot_poincare(args)
    # plot_figure2a.plot_poincare_separate(args)

    # Plot Fig.S1 or S2.
    plot_figure_s1_2.plot_lyapunov0s(args)
    plot_figure_s1_2.plot_lyapunov11s(args)
    plot_figure_s1_2.plot_lyapunov2s(args)

    # Plot Fig.S7 or S8.
    plot_figure_s7_8.plot_lyapunov1s(args)
    plot_figure_s7_8.plot_acc(args)

    end_time = time.time()
    print('elapsed time is {} mins'.format((end_time-start_time)/60))


if __name__ == "__main__":
    main()

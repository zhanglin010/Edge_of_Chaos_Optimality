# TensorFlow and other packages
import argparse
import math
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.models import Model

import loggingreporter as loggingreporter
import models_res as models
# import models_dense as models
import plot_figure3a
import plot_figure3b
import plot_figure_s9
import plot_figure_s10
import utils

# Training settings
parser = argparse.ArgumentParser(description='Jacobian study')
parser.add_argument('--architecture', type=str, default='mlp_0',
                    help='architecture of neural networks')
parser.add_argument('--activation-func', type=str, default='relu',
                    help='activation function for hidden layers')
parser.add_argument('--epochs', default=4, type=int, metavar='N',
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
    # Get the dataset for different architectures
    loss = 'categorical_crossentropy'
    if args.architecture == 'resnet':
        # (version1 20, conv first)
        args.layer = 'activation_19'
        # (version2 20, conv first)
        # args.layer = 'activation_18'
        # (version2 20, conv last)
        # args.layer = 'conv2d_22'
        print('layer is {}'.format(args.layer))
        args.log_epochs = np.concatenate(
            (np.arange(0, 8, 4), np.arange(4, 1004, 4)))
        args.subtract_pixel_mean = True
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data1(args)
        args.input_shape = x_train.shape[1:]
        datagen = utils.get_datagen1(x_train)
    elif args.architecture == 'resnet_without_aug':
        args.layer = 'activation_19'
        print('layer is {}'.format(args.layer))
        args.log_epochs = np.concatenate(
            (np.arange(0, 8, 4), np.arange(4, 1004, 4)))
        args.subtract_pixel_mean = True
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data1(args)
        args.input_shape = x_train.shape[1:]
    elif args.architecture == 'densenet':
        # conv_first 16
        # args.layer = 'activation_14'
        # conv_first special 16
        args.layer = 'activation_15'
        # conv_last
        # args.layer = 'conv2d_15'
        # args.log_epochs = np.arange(args.epochs)
        args.log_epochs = np.arange(0, 201, 1)
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data2(args)
        args.input_shape = x_train.shape[1:]
        datagen = utils.get_datagen2(x_train)
    elif args.architecture == 'densenet_without_aug':
        args.layer = 'conv2d_15'
        (x_train, y_train), (x_val, y_val) = utils.get_cifar10_data2(args)
        args.input_shape = x_train.shape[1:]

    args.dir = '{}/{}_{}/{}_{}_{}'.format(args.architecture, args.optimizer,
                                          args.momentum, args.activation_func,
                                          args.epochs, args.num_iterations)
    for num_repeat in range(args.num_repeats):
        # break
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
        # Three callbacks for saving jacobian norms and learning rate adjustment.
        reporter = loggingreporter.LoggingReporter(args, intermediate_model,
                                                   x_train, y_train,
                                                   x_val, y_val)
        lr_scheduler = LearningRateScheduler(utils.lr_schedule)
        lr_reducer1 = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                        cooldown=0,
                                        patience=5,
                                        min_lr=0.5e-6)
        lr_reducer2 = ReduceLROnPlateau(monitor='val_accuracy', factor=np.sqrt(0.1),
                                        cooldown=0, patience=5, min_lr=1e-5)
        if args.architecture == 'resnet':
            model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                      validation_data=(x_val, y_val),
                      epochs=args.epochs, verbose=0, workers=4,
                      steps_per_epoch=math.ceil(
                          x_train.shape[0] / args.batch_size),
                      callbacks=[reporter, ]
                      # callbacks=[reporter, lr_scheduler, lr_reducer1])
                      )
        elif args.architecture == 'resnet_without_aug':
            model.fit(x_train, y_train, epochs=args.epochs,
                      validation_data=(x_val, y_val), verbose=0,
                      callbacks=[reporter, ]
                      # callbacks=[reporter, lr_scheduler, lr_reducer1])
                      )
        elif args.architecture == 'densenet':
            model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                      validation_data=(x_val, y_val),
                      epochs=args.epochs, verbose=0, workers=4,
                      steps_per_epoch=math.ceil(
                          x_train.shape[0] / args.batch_size),
                      # callbacks=[reporter, lr_reducer2]
                      callbacks=[reporter, ]
                      )
        elif args.architecture == 'densenet_without_aug':
            model.fit(x_train, y_train, epochs=args.epochs,
                      validation_data=(x_val, y_val), verbose=0,
                      # callbacks=[reporter, lr_reducer2]
                      callbacks=[reporter, ]
                      )
        tf.keras.backend.clear_session()

    # Plot Fig.3(a). Replace plot_figure3a with plot_figure3b for Fig.3(b).
    plot_figure3a.plot_attractors(args)
    plot_figure3a.plot_lyapunov1s(args)
    plot_figure3a.plot_acc(args)
    plot_figure3a.plot_poincare(args)
    # plot_figure3a.plot_poincare_separate(args)

    # Plot Fig.S9. Replace plot_figure_s9 with plot_figure_s10 for Fig.S10.
    plot_figure_s9.plot_attractors(args)
    plot_figure_s9.plot_acc(args)

    end_time = time.time()
    print('elapsed time is {} mins'.format((end_time-start_time)/60))


if __name__ == "__main__":
    main()

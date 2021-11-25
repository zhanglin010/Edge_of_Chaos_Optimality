import os
import pickle
from collections import OrderedDict

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import tensorflow as tf
from tensorflow import keras


class LoggingReporter(keras.callbacks.Callback):
    """Save the activations to files at after some epochs.

    Args:
        args: configuration options dictionary
        intermediate_model: the intermediate model at certain stage
        x_train: train data
        y_train: train label
        x_val: validate data
        y_val: validate label
    """

    def __init__(self, args, intermediate_model, x_train,
                 y_train, x_val, y_val, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.args = args
        self.intermediate_model = intermediate_model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def on_train_begin(self, logs=None):
        if not os.path.exists(self.args.save_lyapunov0s_dir):
            os.makedirs(self.args.save_lyapunov0s_dir)
        if not os.path.exists(self.args.save_lyapunov1s_dir):
            os.makedirs(self.args.save_lyapunov1s_dir)
        if not os.path.exists(self.args.save_lyapunov2s_dir):
            os.makedirs(self.args.save_lyapunov2s_dir)
        if not os.path.exists(self.args.save_losses_dir):
            os.makedirs(self.args.save_losses_dir)
        self.losses = {}
        self.losses['train'] = []
        self.losses['val'] = []

    def on_epoch_begin(self, epoch, logs=None):
        print('epoch is {}'.format(epoch))
        if epoch in self.args.log_epochs:
            # Compute lyapunov exponent by different methods
            # 0: numerical, 1: theoretical 1, 2: theoretical 2
            lyapunov0 = save_lyapunov0(self.args, self.intermediate_model,
                                       self.x_val[:100])
            lyapunov1 = save_lyapunov1(self.args, self.intermediate_model,
                                       self.x_val[:100])
            lyapunov2 = save_lyapunov2(self.args, self.intermediate_model,
                                       self.x_val[:1])

            # Save the lyapunovs
            fname0 = '{}/epoch{:08d}'.format(self.args.save_lyapunov0s_dir,
                                             epoch)
            fname1 = '{}/epoch{:08d}'.format(self.args.save_lyapunov1s_dir,
                                             epoch)
            fname2 = '{}/epoch{:08d}'.format(self.args.save_lyapunov2s_dir,
                                             epoch)
            with open(fname0, 'wb') as f:
                pickle.dump(lyapunov0, f, pickle.HIGHEST_PROTOCOL)
            with open(fname1, 'wb') as f:
                pickle.dump(lyapunov1, f, pickle.HIGHEST_PROTOCOL)
            with open(fname2, 'wb') as f:
                pickle.dump(lyapunov2, f, pickle.HIGHEST_PROTOCOL)
        # Compute loss and accuracy
        self.losses['train'].append(self.model.evaluate(self.x_train,
                                                        self.y_train,
                                                        verbose=0))
        self.losses['val'].append(self.model.evaluate(self.x_val,
                                                      self.y_val,
                                                      verbose=0))

    def on_train_end(self, logs=None):
        fname = '{}/losses'.format(self.args.save_losses_dir)
        with open(fname, 'wb') as f:
            pickle.dump(self.losses, f, pickle.HIGHEST_PROTOCOL)


def save_lyapunov0(args, model, x):
    """Save lyapounov exponent calculated by numerical method.

    Args:
        args: configuration options dictionary
        model: the intermediate layer model at certain stage
        x: the input to calculate lyapounov exponent

    Returns:
        a dictionary contains lyapounov exponent.
    """
    x1 = x
    x2 = np.add(x1, np.random.normal(0, 0.0001, (x1.shape)))
    length0 = np.linalg.norm((x1-x2).reshape(x1.shape[0], -1), axis=1)
    lyapunovs = []
    length1s = []
    length1s_proj = []
    length2s = []
    length2s_proj = []
    length_diffs = []
    mean_vector = np.mean(x1.reshape(x1.shape[0], -1), axis=0)
    for i in range(args.num_iterations):
        length_diff = np.linalg.norm((x1-x2).reshape(x1.shape[0], -1), axis=1)
        if i > 0:
            lya = np.log(np.divide(length_diff, length0))/i
            lyapunovs.append(lya)
        # Calculate lengths of x1 and x2
        length1 = np.linalg.norm(x1.reshape(x1.shape[0], -1), axis=1)
        length1_proj = np.matmul(x1.reshape(
            x1.shape[0], -1), mean_vector) / np.linalg.norm(mean_vector)

        length2 = np.linalg.norm(x2.reshape(x2.shape[0], -1), axis=1)
        length2_proj = np.matmul(x2.reshape(
            x2.shape[0], -1), mean_vector) / np.linalg.norm(mean_vector)

        length1s.append(length1)
        length1s_proj.append(length1_proj)
        length2s.append(length2)
        length2s_proj.append(length2_proj)
        length_diffs.append(length_diff)
        x1 = model.predict(x1)
        x2 = model.predict(x2)

    # lyapunov = np.mean(np.asarray(lyapunovs), axis=1)
    lyapunov = np.asarray(lyapunovs)

    return [lyapunov, np.asarray(length1s), np.asarray(length1s_proj), np.asarray(length2s), np.asarray(length2s_proj), np.asarray(length_diffs)]


def save_lyapunov1(args, model, x):
    """Save the lyapunov exponent calculated by theoretical method 1.

    Args:
        args: configuration options dictionary
        model: the intermediate layer model at certain stage
        x: the input to calculate jacobians

    Returns:
        a dictionary contains lyapounov exponent.
    """
    x = tf.convert_to_tensor(x)
    for num_iteration in range(1, args.num_iterations+1):
        x = model(x)
        length = np.linalg.norm(x.numpy().reshape(x.shape[0], -1), axis=1)
        if np.mean(length) > 1e10 or np.mean(length) < 1e-10 or num_iteration == args.num_iterations:
            print('length of x is {}'.format(np.mean(length)))
            print('num_iteration for theoretical method 1 is {}'.format(num_iteration))
            x_previous = x
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_previous)
                x = model(x_previous)
                # print('Asymptotic value of x is {}'.format(x[0, 0, 0, :]))
            jacobian = tape.batch_jacobian(
                x, x_previous, experimental_use_pfor=False).numpy()
            jacobian = np.squeeze(jacobian)
            break

    if args.architecture == 'mlp':
        lyapunov = np.array([LA.norm(jacobian[i])/28 for i in range(100)])
    else:
        lyapunov = np.array([LA.norm(jacobian[i])/np.sqrt(3072)
                             for i in range(100)])

    return lyapunov


def save_lyapunov2(args, model, x):
    """Save the lyapunov exponent calculated by theoretical method 2.

    Args:
        args: configuration options dictionary
        model: the intermediate layer model at certain stage
        x: the input to calculate jacobians

    Returns:
        a dictionary contains lyapounov exponent.
    """
    x = tf.convert_to_tensor(x)
    num_images = x.shape[0]
    jacobians = []
    for num_iteration in range(1, args.num_iterations+1):
        x = model(x)
        x_previous = x
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_previous)
            x = model(x_previous)
            # print('Asymptotic value of x is {}'.format(x[0, 0, 0, :]))
        jacobian = tape.batch_jacobian(
            x, x_previous, experimental_use_pfor=False).numpy()
        if args.architecture == 'mlp':
            jacobian = jacobian.reshape(num_images, 784, 784)
        else:
            # jacobian = jacobian.reshape(num_images, 784, 784)
            jacobian = jacobian.reshape(num_images, 3072, 3072)
        jacobians.append(jacobian)

        length = np.linalg.norm(x.numpy().reshape(x.shape[0], -1), axis=1)
        if np.mean(length) > 1e10 or np.mean(length) < 1e-10 or num_iteration == args.num_iterations:
            print('length of x is {}'.format(np.mean(length)))
            print('num_iteration for theoretical method 2 is {}'.format(num_iteration))
            jacobians = np.asarray(jacobians)
            lyapounovs = []
            for i in range(num_images):
                multiplied_jacobians = jacobians[num_iteration//2-1, i, :, :]
                num_jacobians = 1
                for j in range(num_iteration//2, num_iteration):
                    multiplied_jacobians = np.matmul(
                        multiplied_jacobians, jacobians[j, i, :, :])
                    num_jacobians += 1
                max_eigenvalue = eigs(multiplied_jacobians, k=1)[0]
                print('max eigenvalue is {}'.format(max_eigenvalue))
                lyapounov = (1/num_jacobians) * np.log(LA.norm(max_eigenvalue))
                lyapounovs.append(lyapounov)
            break

    return np.asarray(lyapounovs)

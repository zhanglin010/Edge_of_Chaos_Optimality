import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.linalg import eig
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
import matplotlib.lines as mlines


def plot_lyapunov0s(args):
    """Plot lyapounov exponent calculated by numerical method.

    Args:
        args: configuration options dictionary
    """
    lyapunovs, _, _, _, _, _ = read_lyapunov0s(args)
    losses = read_losses(args)

    # Plot loss and lyapunovs
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[lyapunovs[num_repeat][epoch]
                            for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])

    ax1.errorbar(x[:26], lyapunovss.mean(axis=(0, 2))[:26], yerr=lyapunovss.mean(axis=2).std(axis=0)[:26],
                 fmt='o-', c='C3', ms=2.5, lw=1.5, capsize=1, elinewidth=0.5,
                 label=r'$\frac{1}{T} \ln \frac{|{\delta \bf x}_{T}|}{|\delta {\bf x}_{0}|}$')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.spines['left'].set_color('C3')
    ax1.spines['right'].set_color('royalblue')
    ax1.locator_params(axis='y', nbins=4)
    ax1.set_xlabel('Epoch', fontsize=25)
    # ax1.set_ylabel(
    #     r'$\lim_{t \to\infty} \frac{1}{t} \ln \frac{|{\delta \bf x}_{t}|}{|\delta {\bf x}_{0}|}$', fontsize=25)
    ax1.set_ylabel(
        r'$\frac{1}{T} \ln \frac{|{\delta \bf x}_{T}|}{|\delta {\bf x}_{0}|}$', fontsize=25)
    if args.architecture == 'mlp':
        # ax1.set_ylim(-1.3, 2.75)
        ax1.set_ylim(-1, 1.6)
    elif args.architecture == 'cnn':
        ax1.set_ylim(-1.5, 3.2)
    elif args.architecture == 'cnn_dropout':
        ax1.set_ylim(-2, 2.5)

    # ax2 = ax1.twinx()
    ax2.errorbar(x[:26], val_loss.mean(axis=0)[:26], yerr=val_loss.std(axis=0)[:26], fmt='o-',
                 c='royalblue', ms=2.5, lw=1.5, capsize=1, elinewidth=0.4, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.locator_params(axis='y', nbins=4)
    ax2.tick_params(axis='y', colors='royalblue')
    ax2.yaxis.label.set_color('royalblue')
    ax2.set_ylabel('Test loss', fontsize=25)
    if args.architecture == 'mlp':
        # ax2.set_ylim(0.2, 0.64)
        ax2.set_ylim(0.2, 0.57)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.24, 0.93))
    elif args.architecture == 'cnn':
        ax2.set_ylim(0, 2.5)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.2, 0.93))
    elif args.architecture == 'cnn_dropout':
        ax2.set_ylim(0., 1.9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov0s_26'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300)


def plot_lyapunov11s(args):
    """Plot mutual information and etc.

    Args:
        args: configuration options dictionary
    """
    lyapunovs = read_lyapunov1s(args)
    losses = read_losses(args)

    # Plot loss and lyapunovs
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[np.log(lyapunovs[num_repeat][epoch])
                            for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])

    ax1.errorbar(x[:26], lyapunovss.mean(axis=(0, 2))[:26], yerr=lyapunovss.mean(axis=2).std(axis=0)[:26],
                 fmt='o-', c='C3', ms=2.5, lw=1.5, capsize=1, elinewidth=0.4, zorder=10,
                 label=r'$\ln (\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert})$')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.spines['left'].set_color('C3')
    ax1.spines['right'].set_color('royalblue')
    ax1.locator_params(axis='y', nbins=4)
    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel(
        r'$\ln (\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert})$', fontsize=25)
    if args.architecture == 'mlp':
        ax1.set_ylim(-1.3, 2.75)
    elif args.architecture == 'cnn':
        ax1.set_ylim(-1.5, 3.2)
    elif args.architecture == 'cnn_dropout':
        ax1.set_ylim(0, 50)

    # ax2 = ax1.twinx()
    ax2.errorbar(x[:26], val_loss.mean(axis=0)[:26], yerr=val_loss.std(axis=0)[:26], fmt='o-',
                 c='royalblue', ms=2.5, lw=1.5, capsize=1, elinewidth=0.4, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='y', colors='royalblue')
    ax2.yaxis.label.set_color('royalblue')
    ax2.locator_params(axis='y', nbins=4)
    ax2.set_ylabel('Test loss', fontsize=25)
    if args.architecture == 'mlp':
        ax2.set_ylim(0.2, 0.64)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.24, 0.93))
    elif args.architecture == 'cnn':
        ax2.set_ylim(0, 2.5)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.20, 0.93))
    elif args.architecture == 'cnn_dropout':
        ax2.set_ylim(0., 1.8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov11s_26'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300)


def plot_lyapunov2s(args):
    """Plot mutual information and etc.

    Args:
        args: configuration options dictionary
    """
    lyapunovs = read_lyapunov2s(args)
    losses = read_losses(args)

    # Plot loss and lyapunovs
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[lyapunovs[num_repeat][epoch]
                            for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss_avg = np.ma.masked_invalid(lyapunovss).mean(axis=2)
    # lyapunovss_avg = lyapunovss.mean(axis=2)
    print('lyapunov2ss_avg of epoch 0-10 is {}'.format(
        lyapunovss_avg[:, :10]))

    ax1.errorbar(x[:26], lyapunovss_avg.mean(axis=0)[:26], yerr=lyapunovss_avg.std(axis=0)[:26],
                 fmt='o-', c='C3', ms=2.5, lw=1.5, capsize=1, elinewidth=0.4, zorder=10,
                 label=r'$\frac{1}{\tau}\ln |\lambda_1|$')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.spines['left'].set_color('C3')
    ax1.spines['right'].set_color('royalblue')
    ax1.locator_params(axis='y', nbins=4)
    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel(r'$\frac{1}{\tau}\ln |\lambda_1|$', fontsize=25)
    if args.architecture == 'mlp':
        ax1.set_ylim(-1.3, 2.75)
    elif args.architecture == 'cnn':
        ax1.set_ylim(-1.5, 3.2)
    elif args.architecture == 'cnn_dropout':
        ax1.set_ylim(-2.2, 2.5)

    # ax2 = ax1.twinx()
    ax2.errorbar(x[:26], val_loss.mean(axis=0)[:26], yerr=val_loss.std(axis=0)[:26], fmt='o-',
                 c='royalblue', ms=2.5, lw=1.5, capsize=1, elinewidth=0.4, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='y', colors='royalblue')
    ax2.yaxis.label.set_color('royalblue')
    ax2.locator_params(axis='y', nbins=4)
    ax2.set_ylabel('Test loss', fontsize=25)
    if args.architecture == 'mlp':
        ax2.set_ylim(0.2, 0.64)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.19, 0.93))
    elif args.architecture == 'cnn':
        ax2.set_ylim(0, 2.5)
        fig.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.18, 0.93))
    elif args.architecture == 'cnn_dropout':
        ax2.set_ylim(0., 1.8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov2s_26'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300)


def read_lyapunov0s(args):
    """Read the files contains lyapunov0s.

    Args:
        args: configuration options dictionary

    Returns:
        lyapunov0s
    """
    lyapunovs = {}
    length1s = {}
    length1s_proj = {}
    length2s = {}
    length2s_proj = {}
    length_diffs = {}
    for num_repeat in range(args.num_repeats):
        lyapunovs[num_repeat] = {}
        length1s[num_repeat] = {}
        length1s_proj[num_repeat] = {}
        length2s[num_repeat] = {}
        length2s_proj[num_repeat] = {}
        length_diffs[num_repeat] = {}
        for epoch in args.log_epochs:
            fname = 'rawdata/lyapunov0s/{}/{}/epoch{:08d}'.format(
                args.dir, num_repeat, epoch)
            with open(fname, 'rb') as f:
                d = pickle.load(f)
            final_lyapunovs = np.array(
                [get_final(d[0][:, i]) for i in range(100)])
            lyapunovs[num_repeat][epoch] = final_lyapunovs
            length1s[num_repeat][epoch] = d[1]
            length1s_proj[num_repeat][epoch] = d[2]
            length2s[num_repeat][epoch] = d[3]
            length2s_proj[num_repeat][epoch] = d[4]
            length_diffs[num_repeat][epoch] = d[5]

    return lyapunovs, length1s, length1s_proj, length2s, length2s_proj, length_diffs


def get_final(xs):
    temp = 0.0
    for x in xs:
        if np.isfinite(x):
            temp = x
    return temp


def read_lyapunov1s(args):
    """Read the files contains lyapunov1s.

    Args:
        args: configuration options dictionary

    Returns:
        lyapunov1s
    """
    lyapunovs = {}
    for num_repeat in range(args.num_repeats):
        lyapunovs[num_repeat] = {}
        for epoch in args.log_epochs:
            fname = 'rawdata/lyapunov1s/{}/{}/epoch{:08d}'.format(
                args.dir, num_repeat, epoch)
            with open(fname, 'rb') as f:
                d = pickle.load(f)
            lyapunovs[num_repeat][epoch] = d

    return lyapunovs


def read_lyapunov2s(args):
    """Read the files contains lyapunov1s.

    Args:
        args: configuration options dictionary

    Returns:
        lyapunov1s
    """
    lyapunovs = {}
    for num_repeat in range(args.num_repeats):
        lyapunovs[num_repeat] = {}
        for epoch in args.log_epochs:
            fname = 'rawdata/lyapunov2s/{}/{}/epoch{:08d}'.format(
                args.dir, num_repeat, epoch)
            with open(fname, 'rb') as f:
                d = pickle.load(f)
            lyapunovs[num_repeat][epoch] = d

    return lyapunovs


def read_losses(args):
    """Get the losses from files

    Args:
        args: configuration options dictionary

    Returns:
        loss and accuracy
    """
    losses = {}
    for num_repeat in range(args.num_repeats):
        fname = 'rawdata/losses/{}/{}/losses'.format(args.dir, num_repeat)
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            losses[num_repeat] = d
            # losses[num_repeat] = {'train': 0, 'val': 0}

    return losses

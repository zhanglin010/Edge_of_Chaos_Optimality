import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.linalg import eig
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
import matplotlib.lines as mlines


def plot_lyapunov1s(args):
    """Plot mutual information and etc.

    Args:
        args: configuration options dictionary
    """
    lyapunovs = read_lyapunov1s(args)
    losses = read_losses(args)

    # Plot loss and lyapunovs
    fig, ax1 = plt.subplots(figsize=(8.147, 5))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[lyapunovs[num_repeat][epoch]
                            for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    print(
        'lyapunov1ss of epoch 0-10 is {}'.format(lyapunovss.mean(axis=2)[:, :10]))
    ax1.axhline(y=1, color='k', linestyle='--')
    ax1.errorbar(x[:51], lyapunovss.mean(axis=(0, 2))[:51],
                 yerr=lyapunovss.mean(axis=2).std(axis=0)[:51],
                 # yerr=lyapunovss.std(axis=(0, 2))[:51],
                 fmt='o-', c='C3', ms=2, lw=1.2, capsize=1, elinewidth=0.4,
                 label=r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$')
    ax1.set_yscale('symlog')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.spines['left'].set_color('C3')
    ax1.spines['right'].set_color('royalblue')
    # ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel(
        r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$', fontsize=25)
    if args.architecture == 'mlp':
        ax1.set_ylim(0, 90)
    elif args.architecture == 'cnn':
        # CNN5 and CNN7
        ax1.set_ylim(0, 60)
        # CNN3
        # ax1.set_ylim(0, 90)
        # CNN5 without pooling
        ax1.set_ylim(0, 90)
    elif args.architecture == 'cnn_dropout':
        ax1.set_ylim(0, 60)
    elif args.architecture == 'cnn_with_aug':
        # CNN5 and CNN7
        ax1.set_ylim(0, 5)

    # ax2 = ax1.twinx()
    ax2.errorbar(x[:51], val_loss.mean(axis=0)[:51], yerr=val_loss.std(axis=0)[:51], fmt='o-',
                 c='royalblue', ms=2, lw=1.2, capsize=1, elinewidth=0.4, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.set_xticks([])
    ax2.tick_params(axis='y', colors='royalblue')
    # ax2.locator_params(axis='y', nbins=5)
    ax2.yaxis.label.set_color('royalblue')
    ax2.set_ylabel('Test loss', fontsize=25)
    if args.architecture == 'mlp':
        # MLP100-784
        # ax2.set_ylim(0.195, 0.91)
        # MLP784-784
        # ax2.set_ylim(0.195, 0.91)
        # MLP784-784-784
        ax2.set_ylim(0.2, 0.91)
        # MLP100-100-784
        # ax2.set_ylim(0.195, 0.92)
        ax2.set_yticks([0.3, 0.6, 0.9])
        fig.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.22, 0.91))
    elif args.architecture == 'cnn':
        # CNN7
        # ax2.set_ylim(0., 3.4)
        # ax2.set_yticks([0.5, 1.5, 2.5])
        # CNN5
        ax2.set_ylim(0., 3.7)
        ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
        # CNN3
        # ax2.set_ylim(0, 5.3)
        # ax2.set_yticks([0.5, 2.5, 4.5])
        # CNN5 without pooling
        # ax2.set_ylim(-0.1, 5.9)
        # ax2.set_yticks([0.5, 2.5, 4.5])
        # CNN9
        # ax2.set_ylim(0, 5.3)
        # ax2.set_yticks([0.5, 2.5, 4.5])
        # for no last avgpool
        # ax2.set_ylim(0, 4.6)
        fig.legend(fontsize=20, loc='upper left',
                   bbox_to_anchor=(0.175, 0.945))
    elif args.architecture == 'cnn_dropout':
        ax2.set_ylim(0., 1.9)
    elif args.architecture == 'cnn_with_aug':
        # CNN7
        # ax2.set_ylim(0., 3.4)
        # ax2.set_yticks([0.5, 1.5, 2.5])
        # CNN5
        # ax2.set_ylim(0., 3.7)
        # ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
        # CNN3
        ax2.set_ylim(0, 5.3)
        ax2.set_yticks([0.5, 2.5, 4.5])
        # CNN5 without pooling
        ax2.set_ylim(-0.1, 5.9)
        ax2.set_yticks([0.5, 2.5, 4.5])
        # CNN9
        # ax2.set_ylim(0, 5.3)
        # ax2.set_yticks([0.5, 2.5, 4.5])
        # for no last avgpool
        # ax2.set_ylim(0, 4.6)
        fig.legend(fontsize=20, loc='upper left',
                   bbox_to_anchor=(0.175, 0.945))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov1s_51'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300, transparent=True)


def plot_acc(args):
    """Plot accuracy.

    Args:
        args: configuration options dictionary
    """
    losses = read_losses(args)
    fig, ax1 = plt.subplots(figsize=(6.755, 2.6))
    x = np.arange(args.epochs//1)
    train_acc = np.array([[losses[num_repeat]['train'][epoch][1]
                           for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][1]
                         for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(x[:51], train_acc.mean(axis=0)[:51], yerr=train_acc.std(axis=0)[:51],
                 fmt='o-', c='C7', ms=2, lw=1.2, capsize=1, elinewidth=0.4, zorder=10,
                 label='Train')
    ax1.errorbar(x[:51], val_acc.mean(axis=0)[:51], yerr=val_acc.std(axis=0)[:51],
                 fmt='o-', c='C5', ms=2, lw=1.2, capsize=1, elinewidth=0.4,
                 label='Test')
    ax1.tick_params(axis='both', which='major', labelsize=25.5)
    # MLP
    ax1.set_ylim(0.56, 1)
    ax1.set_yticks([0.6, 0.9])
    # CNN
    ax1.set_ylim(0.03, 1.05)
    ax1.set_yticks([0.2, 0.8])
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    # ax1.yaxis.label.set_color('C5')
    # ax1.spines['right'].set_color('C5')
    # ax1.tick_params(axis='y', colors='C5')
    ax1.set_ylabel('Accuracy', fontsize=25.5)
    # ax1.spines['left'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([0, 10, 20, 30, 40, 50])
    # ax1.set_xlim(-0.4, 20.4)
    ax1.set_xlabel('Epoch', fontsize=25.5)
    ax1.legend(fontsize=19, loc=4, ncol=2)

    # fig.patch.set_facecolor('None')
    # fig.patch.set_alpha(0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/acc_loss_1_51'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300, transparent=True)


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

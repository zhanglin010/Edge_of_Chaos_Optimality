import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from scipy.linalg import eig
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
import matplotlib.lines as mlines


def plot_attractors(args):
    """Plot length related results.

    Args:
        args: configuration options dictionary
    """
    image_num = 0
    _, length1s, length1s_proj, length2s, length2s_proj, length_diffs = read_lyapunov0s(
        args)
    losses = read_losses(args)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])

    fig, ax1 = plt.subplots(figsize=(7.835, 5), dpi=50)
    length_diffs_all = np.array([[[np.mean(length_diffs[num_repeat][epoch][args.num_iterations//2:, image_num2])
                                   for image_num2 in range(100)] for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(args.log_epochs[1:201], length_diffs_all.mean(axis=(0, 2))[1:201], yerr=length_diffs_all.std(
        axis=(0, 2))[1:201], fmt='o-', c='C3', ms=2.5, lw=1.2, capsize=1, elinewidth=0.5, label='$\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='x', nbins=6)
    # ax1.set_xlabel('Epoch', fontsize25)
    ax1.set_ylabel(
        'Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=25)
    # ax1.legend(fontsize=25, loc=(0.53, 0.85))
    if args.architecture == 'densenet':
        # DenseNet28
        ax1.set_ylim(-3, 8.2)
        # DenseNet40
        ax1.set_ylim(-3, 7)
        ax1.set_yticks([0, 5])
        # DenseNet40_2
        ax1.set_ylim(-4, 12)
        ax1.set_yticks([0, 8])
        # DenseNet28_2
        # ax1.set_ylim(-4, 12)
        # ax1.set_yticks([0, 8])
    elif args.architecture == 'densenet_without_aug':
        ax1.set_ylim(0, 80)
    elif args.architecture == 'resnet_without_aug':
        ax1.set_ylim(0, 80)

    ax3 = ax1.twinx()
    ax3.plot(args.log_epochs[1:201], val_loss.mean(axis=0)[1:201], 'o-', color='royalblue',
             ms=2.5, lw=1.2, label='Test loss')
    ax3.tick_params(axis='both', which='major', labelsize=25)
    ax3.tick_params(axis='y', colors='royalblue')
    ax3.spines['left'].set_color('C3')
    ax3.spines['right'].set_color('royalblue')
    ax3.yaxis.label.set_color('royalblue')
    ax3.locator_params(axis='y', nbins=3)
    ax3.set_xlim(-1, 203)
    ax3.set_xticks([])
    ax3.set_ylabel('Test loss', fontsize=25)
    # ax3.legend(fontsize=25, loc=(0.53, 0.72))
    if args.architecture == 'densenet':
        # DenseNet28
        ax3.set_ylim(0.18, 2.3)
        ax3.set_yticks([0.5, 1.0, 1.5, 2.0])
        # DenseNet40
        ax3.set_ylim(0.18, 2.3)
        ax3.set_yticks([0.5, 1.0, 1.5, 2.0])
        # DenseNet40_2
        ax3.set_ylim(0.18, 3.2)
        ax3.set_yticks([0.5, 1.5, 2.5])
        # DenseNet28_2
        # ax3.set_ylim(0.18, 3.2)
        # ax3.set_yticks([0.5, 1.5, 2.5])
    elif args.architecture == 'densenet_without_aug':
        ax3.set_ylim(0.6, 1.8)
    elif args.architecture == 'resnet_without_aug':
        ax3.set_ylim(0.6, 1.8)

    fig.legend(fontsize=24, loc='upper right', bbox_to_anchor=(0.908, 0.892))
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/attractor_size_appendix'.format(args.dir, args.num_repeats),
                bbox_inches='tight', transparent=True)


def plot_acc(args):
    """Plot accuracy.

    Args:
        args: configuration options dictionary
    """
    losses = read_losses(args)
    fig, ax1 = plt.subplots(figsize=(7.513, 2.6))
    x = np.arange(args.epochs//1)
    train_acc = np.array([[losses[num_repeat]['train'][epoch][1]
                           for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][1]
                         for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(args.log_epochs[1:201], train_acc.mean(axis=0)[1:201], yerr=train_acc.std(axis=0)[1:201],
                 fmt='o-', c='C7', ms=2, lw=1.2, capsize=1, elinewidth=0.4, zorder=10,
                 label='Train')
    ax1.errorbar(args.log_epochs[1:201], val_acc.mean(axis=0)[1:201], yerr=val_acc.std(axis=0)[1:201],
                 fmt='o-', c='C5', ms=2, lw=1.2, capsize=1, elinewidth=0.4,
                 label='Test')
    ax1.tick_params(axis='both', which='major', labelsize=25.5)
    # DenseNet28
    ax1.set_ylim(0.61, 1.02)
    ax1.set_yticks([0.7, 0.9])
    # DenseNet40
    ax1.set_ylim(0.61, 1.02)
    ax1.set_yticks([0.7, 0.9])
    # DenseNet28_2
    # ax1.set_ylim(0.55, 1.02)
    # ax1.set_yticks([0.6, 0.9])
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Accuracy', fontsize=25.5)
    ax1.set_xlim(-1, 203)
    ax1.set_xticks([0, 40, 80, 120, 160, 200])
    # ax1.set_xlim(-0.4, 20.4)
    ax1.set_xlabel('Epoch', fontsize=25.5)
    ax1.legend(fontsize=22, loc=4, ncol=2)

    # fig.patch.set_facecolor('None')
    # fig.patch.set_alpha(0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/acc_loss_1_appendix'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=50, transparent=True)


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

    return losses

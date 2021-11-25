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
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[lyapunovs[num_repeat][epoch]
                            for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(x[:13], lyapunovss.mean(axis=(0, 2))[:13],
                 yerr=lyapunovss.mean(axis=2).std(axis=0)[:13],
                 fmt='o-', c='C3', ms=3.5, lw=1.5, capsize=1.5, elinewidth=0.8,
                 label=r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$')
    ax1.axhline(y=1, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.set_yticks([1, 3, 5])
    ax1.locator_params(axis='y', nbins=3)
    # ax1.set_xlabel('Epoch', fontsize=23)
    ax1.set_ylabel(
        r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$', fontsize=22)
    if args.architecture == 'mlp':
        ax1.set_ylim(0.25, 3.8)
    elif args.architecture == 'cnn':
        ax1.set_ylim(0, 6)
    elif args.architecture == 'cnn_dropout':
        ax1.set_ylim(0, 60)

    ax2 = ax1.twinx()
    ax2.errorbar(x[:13], val_loss.mean(axis=0)[:13], yerr=val_loss.std(axis=0)[:13], fmt='o-',
                 c='dimgrey', ms=3.5, lw=1.5, capsize=1.5, elinewidth=0.8, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='y', colors='dimgrey')
    ax2.locator_params(axis='y', nbins=3)
    ax2.spines['left'].set_color('C3')
    ax2.spines['right'].set_color('dimgrey')
    ax2.yaxis.label.set_color('dimgrey')
    ax2.set_ylabel('Test loss', fontsize=22)
    ax2.set_xticks([])
    ax2.set_xlim([-0.24, 12.24])
    ax2.set_yticks([0.5, 1.5, 2.5])
    if args.architecture == 'mlp':
        ax2.set_ylim(0.295, 0.5)
    elif args.architecture == 'cnn':
        # CNN5
        # ax2.set_ylim(0.5, 2.07)
        # ax2.set_ylim(0.4, 2.3)
        ax2.set_ylim(0.25, 3.55)
    elif args.architecture == 'cnn_dropout':
        ax2.set_ylim(0., 1.9)

    # fig.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.9195, 1.113))
    fig.legend(fontsize=19, loc='upper right', bbox_to_anchor=(0.918, 0.938))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov1s_13'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300, transparent=True)


def plot_acc(args):
    """Plot accuracy.

    Args:
        args: configuration options dictionary
    """
    losses = read_losses(args)
    fig, ax1 = plt.subplots(figsize=(11.085, 2.6))
    x = np.arange(args.epochs//1)
    train_acc = np.array([[losses[num_repeat]['train'][epoch][1]
                           for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][1]
                         for epoch in range(args.epochs//1)] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(x[:13], train_acc.mean(axis=0)[:13], yerr=train_acc.std(axis=0)[:13],
                 fmt='o-', c='darkolivegreen', ms=3.5, lw=1.5, capsize=1.3, elinewidth=0.8,
                 label='Train')
    ax1.errorbar(x[:13], val_acc.mean(axis=0)[:13], yerr=val_acc.std(axis=0)[:13],
                 fmt='o-', c='C5', ms=3.5, lw=1.5, capsize=1.5, elinewidth=0.8, label='Test')
    ax1.tick_params(axis='both', which='major', labelsize=22.2)
    ax1.set_ylim(0.25, 1.04)
    ax1.set_yticks([0.3, 0.9])
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    # ax1.yaxis.label.set_color('C5')
    # ax1.spines['right'].set_color('C5')
    # ax1.tick_params(axis='y', colors='C5')
    ax1.set_ylabel('Accuracy', fontsize=22.2)
    ax1.set_xticks([0, 4, 8, 12])
    ax1.set_xlim(-0.24, 12.24)
    ax1.set_xlabel('Epoch', fontsize=23.2)
    ax1.legend(fontsize=20, loc=4, ncol=2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/acc_loss_1_13'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300, transparent=True)


def plot_poincare(args):
    """Plot length related results.

    Args:
        args: configuration options dictionary
    """
    image_num = 0
    num_repeats = [0]
    _, _, length1s_proj, _, length2s_proj, length_diffs = read_lyapunov0s(args)

    # Plot length1s_projection's poincare
    x = np.arange(1, args.num_iterations+1, 1)
    length1s_proj = np.array([[length1s_proj[num_repeat][epoch] for epoch in range(
        args.epochs//1)] for num_repeat in range(args.num_repeats)])
    length2s_proj = np.array([[length2s_proj[num_repeat][epoch] for epoch in range(
        args.epochs//1)] for num_repeat in range(args.num_repeats)])
    for num_repeat in num_repeats:
        fig, axs = plt.subplots(4, 4, figsize=(30, 30))
        for indx, epoch in enumerate(np.arange(0, 15)):
            for num_iterations in range(args.num_iterations//2-1, args.num_iterations-1):
                axs[indx//4, indx % 4].plot(length1s_proj[num_repeat, epoch, num_iterations, image_num],
                                            length1s_proj[num_repeat, epoch, num_iterations+1, image_num], 'o', color='C0', ms=3.5)

            axs[indx//4, indx % 4].plot(length1s_proj[num_repeat, epoch, args.num_iterations-2, image_num],
                                        length1s_proj[num_repeat, epoch, args.num_iterations-1, image_num], 's', color='C1', ms=12)
            axs[indx//4, indx % 4].plot(length2s_proj[num_repeat, epoch, args.num_iterations-2, image_num],
                                        length2s_proj[num_repeat, epoch, args.num_iterations-1, image_num], 'D', color='C2', ms=10)
            axs[indx//4, indx %
                4].tick_params(axis='both', which='major', labelsize=24)
            axs[indx//4, indx % 4].locator_params(axis='y', nbins=4)
            axs[indx//4, indx % 4].locator_params(axis='x', nbins=4)
            axs[indx//4, indx % 4].set_xlabel(r'$\overline{X}_t$', fontsize=24)
            axs[indx//4, indx %
                4].set_ylabel(r'$\overline{X}_{t+1}$', fontsize=24)
            if indx == 0:
                if args.architecture == 'mlp':
                    axs[indx//4, indx % 4].set_xlim(0.814, 0.826)
                    axs[indx//4, indx % 4].set_ylim(0.814, 0.826)
            axs[indx//4, indx %
                4].set_title('Epoch {}'.format(epoch), fontsize=26)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
            os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
        plt.savefig('results/{}_{}/poincare_{}'.format(args.dir, args.num_repeats, num_repeat),
                    bbox_inches='tight', dpi=300)


def plot_poincare_separate(args):
    """Plot length related results.

    Args:
        args: configuration options dictionary
    """
    image_num = 0
    num_repeat = 0
    _, _, length1s_proj, _, length2s_proj, _ = read_lyapunov0s(args)

    # Plot length1s_projection's poincare
    x = np.arange(1, args.num_iterations+1, 1)
    length1s_proj = np.array([[length1s_proj[num_repeat][epoch] for epoch in range(
        args.epochs//1)] for num_repeat in range(args.num_repeats)])
    length2s_proj = np.array([[length2s_proj[num_repeat][epoch] for epoch in range(
        args.epochs//1)] for num_repeat in range(args.num_repeats)])
    for indx, epoch in enumerate([1, 3, 6]):
        if indx == 0:
            fig, ax1 = plt.subplots(figsize=(1.48, 2), dpi=300)
            ax1.set_xlim(0.65355, 0.6566)
            ax1.set_ylim(0.65355, 0.6566)
            ax1.set_xticks([0.654, 0.656])
            ax1.set_yticks([0.654, 0.656])
            ax1.set_yticklabels([])
            ax1.set_ylabel(r'$\overline{X}_{t+1}$', fontsize=12)
            ms1 = 6
            ms2 = 4
        elif indx == 1:
            fig, ax1 = plt.subplots(figsize=(1.31, 2), dpi=300)
            ax1.set_xlim(0.73, 0.828)
            ax1.set_ylim(0.73, 0.828)
            ax1.set_xticks([0.74, 0.82])
            ax1.set_yticks([0.74, 0.82])
            ax1.set_yticklabels([])
            ms1 = 6
            ms2 = 4
        elif indx == 2:
            fig, ax1 = plt.subplots(figsize=(1.3, 2), dpi=300)
            ax1.set_xlim([1.09, 1.31])
            ax1.set_ylim([1.09, 1.31])
            ax1.set_xticks([1.1, 1.3])
            ax1.set_yticks([1.1, 1.3])
            ax1.set_yticklabels([])
            ms1 = 4
            ms2 = 4
        fig.patch.set_facecolor('None')
        fig.patch.set_alpha(0)
        for num_iterations in range(args.num_iterations//2-1, args.num_iterations-1):
            ax1.plot(length1s_proj[num_repeat, epoch, num_iterations, image_num],
                     length1s_proj[num_repeat, epoch, num_iterations+1, image_num], 'o', color='C0', ms=0.5)
        ax1.plot(length1s_proj[num_repeat, epoch, args.num_iterations-2, image_num],
                 length1s_proj[num_repeat, epoch, args.num_iterations-1, image_num], 's', color='C1', ms=ms1)
        ax1.plot(length2s_proj[num_repeat, epoch, args.num_iterations-2, image_num],
                 length2s_proj[num_repeat, epoch, args.num_iterations-1, image_num], 'D', color='C2', ms=ms2)
        ax1.tick_params(axis='x', which='major', labelsize=12)
        ax1.set_xlabel(r'$\overline{X}_t$', fontsize=12)
        ax1.set_title('Epoch {}'.format(epoch), fontsize=13)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
            os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
        plt.savefig('results/{}_{}/poincare_{}_{}'.format(args.dir, args.num_repeats, num_repeat, epoch),
                    bbox_inches='tight', facecolor=fig.get_facecolor())


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

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

    fig, ax1 = plt.subplots(figsize=(12, 5.5), dpi=300)
    x = np.arange(args.epochs//1)
    length_diffs_all = np.array([[[np.mean(length_diffs[num_repeat][epoch][args.num_iterations//2:, image_num2])
                                   for image_num2 in range(100)] for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(args.log_epochs[1:201], length_diffs_all.mean(axis=(0, 2))[1:201], yerr=length_diffs_all.std(
        axis=(0, 2))[1:201], fmt='o-', c='C3', ms=2.8, lw=1.2, capsize=1, elinewidth=0.5, label='$\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.yaxis.label.set_color('C3')
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.tick_params(axis='y', colors='C3')
    ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='x', nbins=6)
    # ax1.set_xlabel('Epochs', fontsize=23)
    ax1.set_ylabel(
        'Asymptotic distance $\|\mathbf{x}\'_\infty-\mathbf{x}_\infty\|$', fontsize=22)
    # ax1.legend(fontsize=22, loc=(0.53, 0.85))
    if args.architecture == 'densenet':
        ax1.set_ylim(-2.5, 10.5)
    elif args.architecture == 'densenet_without_aug':
        ax1.set_ylim(0, 80)
    elif args.architecture == 'resnet':
        ax1.set_ylim(0, 20)
    elif args.architecture == 'resnet_without_aug':
        ax1.set_ylim(6, 60)

    ax3 = ax1.twinx()
    ax3.plot(args.log_epochs[1:201], val_loss.mean(axis=0)[1:201], 'o-', color='dimgrey',
             ms=2.8, lw=1.2, label='Test loss')
    ax3.set_xlim(-1, 202)
    ax3.set_xticks([])
    ax3.tick_params(axis='both', which='major', labelsize=22)
    ax3.locator_params(axis='y', nbins=4)
    ax3.tick_params(axis='y', colors='dimgrey')
    ax3.set_yticks([0.5, 1.5, 2.5])
    ax3.set_ylabel('Test loss', fontsize=22)
    ax3.yaxis.label.set_color('dimgrey')
    ax3.spines['left'].set_color('C3')
    ax3.spines['right'].set_color('dimgrey')
    # ax3.legend(fontsize=22, loc=(0.53, 0.72))
    if args.architecture == 'densenet':
        ax3.set_ylim(0.3, 2.6)
    elif args.architecture == 'densenet_without_aug':
        ax3.set_ylim(0.6, 1.8)
    elif args.architecture == 'resnet':
        ax3.set_ylim(0.5, 5.8)
    elif args.architecture == 'resnet_without_aug':
        ax3.set_ylim(1, 9)

    # fig.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.91, 1.053))
    fig.legend(fontsize=19, loc='upper right', bbox_to_anchor=(0.9107, 1.065))
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/attractor_size'.format(args.dir, args.num_repeats),
                bbox_inches='tight', transparent=True)


def plot_acc(args):
    """Plot accuracy.

    Args:
        args: configuration options dictionary
    """
    losses = read_losses(args)
    # fig, ax1 = plt.subplots(figsize=(10.56, 2.6))
    fig, ax1 = plt.subplots(figsize=(10.6, 2.6))
    train_acc = np.array([[losses[num_repeat]['train'][epoch][1]
                           for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    val_acc = np.array([[losses[num_repeat]['val'][epoch][1]
                         for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    ax1.errorbar(args.log_epochs[1:201], train_acc.mean(axis=0)[1:201], yerr=train_acc.std(axis=0)[1:201],
                 fmt='o-', c='darkolivegreen', ms=2.8, lw=1.2, capsize=1, elinewidth=0.4,
                 label='Train')
    ax1.errorbar(args.log_epochs[1:201], val_acc.mean(axis=0)[1:201], yerr=val_acc.std(axis=0)[1:201],
                 fmt='o-', c='C5', ms=2.8, lw=1.2, capsize=1, elinewidth=0.4,
                 label='Test')
    ax1.set_xticks([0, 40, 80, 120, 160, 200])
    ax1.set_xlim(-1, 202)
    ax1.set_xlabel('Epoch', fontsize=23.5)
    ax1.tick_params(axis='both', which='major', labelsize=22.5)
    ax1.set_ylim(0.58, 1)
    ax1.set_yticks([0.6, 0.9])
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    # ax1.yaxis.label.set_color('C5')
    # ax1.spines['right'].set_color('C5')
    # ax1.tick_params(axis='y', colors='C5')
    ax1.set_ylabel('Accuracy', fontsize=22.5)
    # ax1.spines['left'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    ax1.legend(fontsize=20, loc=4, ncol=2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/acc_loss_1'.format(args.dir, args.num_repeats),
                bbox_inches='tight', dpi=300, transparent=True)


def plot_poincare(args):
    """Plot length related results.

    Args:
        args: configuration options dictionary
    """
    image_num = 0
    _, length1s, length1s_proj, length2s, length2s_proj, length_diffs = read_lyapunov0s(
        args)

    # Plot length1s_projection's poincare
    fig, ax1 = plt.subplots(15, 15, figsize=(60, 45), dpi=300)
    x = np.arange(1, args.num_iterations+1, 1)
    length1s_proj = np.array([[length1s_proj[num_repeat][epoch]
                               for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    length2s_proj = np.array([[length2s_proj[num_repeat][epoch]
                               for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    for indx, epoch in enumerate(range(201)):
        # for indx, epoch in enumerate([0, 1, 5, 10, 17, 19, 20, 25, 26, 30, 40, 50, 55, 60, 70, 75, 78, 80, 84, 90, 95, 100]):
        # for indx, epoch in enumerate([0, 1, 5, 8, 10, 20, 30, 40, 50, 59, 60, 63, 70, 75, 77, 80, 90, 100, 110, 120, 130, 140, 145, 150, 160, 170, 180, 185, 190, 200]):
        # for indx, epoch in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
        for num_iterations in range(args.num_iterations//2-1, args.num_iterations-1):
            ax1[indx//15, indx % 15].plot(length1s_proj[0, epoch, num_iterations, image_num],
                                          length1s_proj[0, epoch, num_iterations+1, image_num], 'ro', ms=4)
        ax1[indx//15, indx % 15].plot(length1s_proj[0, epoch, args.num_iterations-2, image_num],
                                      length1s_proj[0, epoch, args.num_iterations-1, image_num], 'bo')
        ax1[indx//15, indx %
            15].plot(length2s_proj[0, epoch, args.num_iterations-2, image_num], length2s_proj[0, epoch, args.num_iterations-1, image_num], 'go')
        ax1[indx//15, indx %
            15].tick_params(axis='both', which='major', labelsize=20)
        ax1[indx//15, indx % 15].set_xlabel(r'$\overline{X}_t$', fontsize=20)
        ax1[indx//15, indx %
            15].set_ylabel(r'$\overline{X}_{t+1}$', fontsize=20)
        ax1[indx//15, indx %
            15].set_title('Epoch {}'.format(epoch), fontsize=20)
        if indx == 0:
            ax1[indx//15, indx % 15].legend(fontsize=14)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/poincare'.format(args.dir, args.num_repeats),
                bbox_inches='tight')

    # Plot attractor vs iterations for all epochs
    fig, ax1 = plt.subplots(15, 15, figsize=(60, 45), dpi=300)
    x = np.arange(1, args.num_iterations+1, 1)
    length_diffs = np.array([[length_diffs[num_repeat][epoch]
                              for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    for indx, epoch in enumerate(range(args.epochs-20)):
        ax1[indx//15, indx % 15].plot(x, length_diffs[0, epoch, :, image_num], 'b-', ms=3,
                                      label='Epoch {}'.format(epoch))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/attractor_size_iterations'.format(args.dir, args.num_repeats),
                bbox_inches='tight')


def plot_poincare_separate(args):
    """Plot length related results.

    Args:
        args: configuration options dictionary
    """
    image_num = 0
    _, length1s, length1s_proj, length2s, length2s_proj, length_diffs = read_lyapunov0s(
        args)

    # Plot length1s_projection's poincare
    x = np.arange(1, args.num_iterations+1, 1)
    length1s_proj = np.array([[length1s_proj[num_repeat][epoch]
                               for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    length2s_proj = np.array([[length2s_proj[num_repeat][epoch]
                               for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    # for indx, epoch in enumerate(range(100)):
    for indx, epoch in enumerate([8, 52, 112, 120, 156]):
        if indx == 0:
            # fig, ax1 = plt.subplots(figsize=(1.85, 2), dpi=300)
            fig, ax1 = plt.subplots(figsize=(1.48, 2), dpi=300)
            ax1.set_ylabel(r'$\overline{X}_{t+1}$', fontsize=12)
            ax1.set_yticks([0.8, 1.0])
            ax1.set_xticks([0.8, 1.0])
            ax1.set_yticklabels([])
            ms1 = 4
            ms2 = 4
        elif indx == 1:
            # fig, ax1 = plt.subplots(figsize=(1.62, 2), dpi=300)
            fig, ax1 = plt.subplots(figsize=(1.22, 2), dpi=300)
            ax1.set_ylim([0.276, 0.322])
            ax1.set_xlim([0.276, 0.322])
            ax1.set_yticks([0.28, 0.31])
            ax1.set_xticks([0.28, 0.31])
            ax1.set_yticklabels([])
            ms1 = 6
            ms2 = 4
        elif indx == 2:
            # fig, ax1 = plt.subplots(figsize=(1.9, 2), dpi=300)
            fig, ax1 = plt.subplots(figsize=(1.35, 2), dpi=300)
            # ax1.set_ylim([0.2012, 0.2052])
            # ax1.set_xlim([0.2012, 0.2052])
            ax1.set_yticks([0.200, 0.201])
            ax1.set_xticks([0.200, 0.201])
            ax1.set_yticklabels([])
            ms1 = 4
            ms2 = 4
        elif indx == 3:
            # fig, ax1 = plt.subplots(figsize=(1.63, 2), dpi=300)
            fig, ax1 = plt.subplots(figsize=(1.26, 2), dpi=300)
            ax1.set_ylim([0.122, 0.128])
            ax1.set_xlim([0.122, 0.128])
            ax1.set_yticks([0.123, 0.127])
            ax1.set_xticks([0.123, 0.127])
            ax1.set_yticklabels([])
            ms1 = 6
            ms2 = 4
        elif indx == 4:
            # fig, ax1 = plt.subplots(figsize=(1.63, 2), dpi=300)
            fig, ax1 = plt.subplots(figsize=(1.32, 2), dpi=300)
            ax1.set_yticks([0.185, 0.192])
            ax1.set_xticks([0.185, 0.192])
            ax1.set_yticklabels([])
            ms1 = 4
            ms2 = 4
        fig.patch.set_facecolor('None')
        fig.patch.set_alpha(0)
        for num_iterations in range(args.num_iterations//2-1, args.num_iterations-1):
            ax1.plot(length1s_proj[0, epoch, num_iterations, image_num],
                     length1s_proj[0, epoch, num_iterations+1, image_num], 'o', color='C0', ms=0.5)
        ax1.plot(length1s_proj[0, epoch, args.num_iterations-2, image_num],
                 length1s_proj[0, epoch, args.num_iterations-1, image_num], 's', color='C1', ms=ms1)
        ax1.plot(length2s_proj[0, epoch, args.num_iterations-2, image_num],
                 length2s_proj[0, epoch, args.num_iterations-1, image_num], 'D', color='C2', ms=ms2)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_xlabel(r'$\overline{X}_t$', fontsize=12)
        ax1.set_title('Epoch {}'.format(epoch), fontsize=13)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
            os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
        plt.savefig('results/{}_{}/poincare_{}'.format(args.dir, args.num_repeats, epoch),
                    bbox_inches='tight', facecolor=fig.get_facecolor())


def plot_lyapunov1s(args):
    """Plot mutual information and etc.

    Args:
        args: configuration options dictionary
    """
    lyapunovs = read_lyapunov1s(args)
    losses = read_losses(args)

    # Plot loss and lyapunovs
    fig, ax1 = plt.subplots(figsize=(9.2, 6))
    x = np.arange(args.epochs//1)
    val_loss = np.array([[losses[num_repeat]['val'][epoch][0]
                          for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    lyapunovss = np.array([[lyapunovs[num_repeat][epoch]
                            for epoch in args.log_epochs] for num_repeat in range(args.num_repeats)])
    # ax1.plot(args.log_epochs[1:201], lyapunovss.mean(axis=(0, 2))[1:201], 'o-', c='C3', ms=2.5, lw=1.2,
    #          label=r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$')
    ax1.errorbar(args.log_epochs, lyapunovss.mean(axis=(0, 2)), yerr=lyapunovss.std(axis=(0, 2)),
                 fmt='o-', c='C3', ms=2.5, lw=1.2, capsize=1, elinewidth=0.4,
                 label=r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$')
    # ax1.set_yscale('symlog')
    ax1.axhline(y=1, color='k', linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_xticks([0, 40, 80, 120, 160, 200])
    ax1.tick_params(axis='y', colors='C3')
    ax1.yaxis.label.set_color('C3')
    ax1.set_yticks([0, 1, 2, 3])
    # ax1.set_yticks([0, 4, 8])
    ax1.set_ylabel(
        r'$\frac{1}{\sqrt{N}} \overline{\Vert \mathtt{J}^{\ast} \Vert}$', fontsize=25)
    if args.architecture == 'densenet':
        ax1.set_ylim(-0.6, 3.9)
        # ax1.set_ylim(-0.5, 3.9)
    elif args.architecture == 'densenet_without_aug':
        ax1.set_ylim(0, 12)

    ax2 = ax1.twinx()
    ax2.plot(args.log_epochs[1:201], val_loss.mean(axis=0)[1:201], 'o-',
             c='royalblue', ms=2.5, lw=1.2, label='Test loss')
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.set_ylabel('Test loss', fontsize=25)
    ax2.set_yticks([0.5, 1, 1.5])
    ax2.tick_params(axis='y', colors='royalblue')
    ax2.yaxis.label.set_color('royalblue')
    ax2.spines['left'].set_color('C3')
    ax2.spines['right'].set_color('royalblue')
    if args.architecture == 'densenet':
        ax2.set_ylim(0.35, 1.7)
    elif args.architecture == 'densenet_without_aug':
        ax2.set_ylim(0, 2.3)

    fig.legend(fontsize=23, loc='upper right', bbox_to_anchor=(0.88, 0.935))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists('results/{}_{}'.format(args.dir, args.num_repeats)):
        os.makedirs('results/{}_{}'.format(args.dir, args.num_repeats))
    plt.savefig('results/{}_{}/lyapunov1s'.format(args.dir, args.num_repeats),
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

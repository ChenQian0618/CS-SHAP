'''
2025/04/13
Created by ChenQian-SJTU
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 1))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from SHAPs.plot_func import setdefault,ticklabel_format

setdefault()


def attr_visualization(savedir, mode, data, value,plot_params,
                       label, predict, dpi=500,):
    '''
    :param savedir: str
    :param mode: str
    :param data: (N,d1,d2,...) object
    :param value: (K,N,d1,d2,...) object
    :param plot_params: dict {x, x_label, y_label} for 1D | {x, y, x_label, y_label} for 2D
    :param dpi: int
    :return:
    '''
    mode = mode.split('_')[0]
    if not os.path.exists(os.path.split(savedir)[0]):
        os.makedirs(os.path.split(savedir)[0])

    # 1) prepare value xyz
    if mode in ['time', 'frequency', 'envelope']:  # 1D
        dimension = 1  # 1D
        plot_params.update({'Y_base': data, 'Ys': value})
    elif mode in ['STFT', 'CS']:  # 2D
        dimension = 2  # 2D
        if mode == 'CS': data[..., 0] *= 0.3  # reduce the DC component in axis-alpha for better visualization
        plot_params.update({'Z_base': data, 'Zs': value})
    else:
        raise ValueError(f'mode should be in [time, frequency, envelope, STFT, CS]')

    # 2.1) define the GridSpec
    N, K = data.shape[0], value.shape[0] # N is the number of samples, K is the number of classes
    dimension_scale = 1.8 if dimension == 1 else 2.5
    figsize = ((K + 1) * 2.5 / 2.54, N * dimension_scale / 2.54)
    gs = mpl.gridspec.GridSpec(N, K + 1, hspace=1)

    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=200)
    Axes = [[] for _ in range(N)]
    storage = {}
    for n in range(N):
        for k in range(K + 1):
            item = fig.add_subplot(gs[n, k])
            if item not in storage:
                storage[item] = fig.add_subplot(item) if item else None
            Axes[n].append(storage[item])

    # 2.2) plot
    if dimension == 1:
        x, Y_base, Ys = plot_params['x'], plot_params['Y_base'], plot_params['Ys']
        xlabel, ylabel = plot_params['x_label'], plot_params['y_label']
        # base
        for n in range(N):
            ax = Axes[n][0]
            ax.plot(x, Y_base[n])
            ax.margins(x=0.001)
            ax.set_xlabel(xlabel + f' | sample#{n} - label#{label[n]:d}')
            ax.set_ylabel('Amp.')

        for n in range(N):
            ys = Ys[:, n, ...]
            ylim = (1.1 * ys.min() - 0.1 * ys.max(), 1.1 * ys.max() - 0.1 * ys.min())
            for k in range(K):
                ax = Axes[n][k + 1]
                ax.plot(x, ys[k], color='C1', alpha=0.8)  # mpl.rcParams['axes.prop_cycle'].by_key()['color'][1]
                ax.margins(x=0.001)
                ax.set_xlabel(xlabel + f' | $y_{k}$={predict[n, k] * 100:.1f}%')
                ax.set_ylim(ylim)
                ax.set_ylabel(ylabel)
                # value
                ax2 = ax.twinx()
                ax2.plot(x, Y_base[n], color='gray', alpha=0.25)
                ax2.set_yticks([])
                ax2.set_xticks(Axes[n][0].get_xticks())
                ax2.set_xlim(Axes[n][0].get_xlim())
                ax2.set_ylim(Axes[n][0].get_ylim())

    else:
        x, y, Z_base, Zs = plot_params['x'], plot_params['y'], plot_params['Z_base'], plot_params['Zs']
        xlabel, ylabel = plot_params['x_label'], plot_params['y_label']
        # base
        for n in range(N):
            ax = Axes[n][0]
            ax.pcolormesh(x, y, Z_base[n], cmap='Blues', shading='gouraud', rasterized=True)
            ax.set_xlabel(xlabel + f' | sample#{n} - label#{label[n]:d}')
            ax.set_ylabel(ylabel)

        for n in range(N):
            bound = max(abs(Zs[:, n].min()), abs(Zs[:, n].max()))
            for k in range(K):
                ax = Axes[n][k + 1]
                ax.pcolormesh(x, y, Zs[k, n], cmap='bwr', shading='auto',
                              vmin=-bound, vmax=bound, rasterized=True)
                ax.set_xlabel(xlabel + f'| $y_{k}$={predict[n, k] * 100:.2f}%')
                ax.set_ylabel(ylabel)
                ax.set_xticks(Axes[n][0].get_xticks())
                ax.set_yticks(Axes[n][0].get_yticks())
                ax.set_xlim(Axes[n][0].get_xlim())
                ax.set_ylim(Axes[n][0].get_ylim())

    # 3) add title
    for k in range(K + 1):
        ax = Axes[0][k]
        if k == 0:
            ax.set_title('Data presentation', fontweight='bold')
        else:
            ax.set_title(f'Predicted class #{k - 1:d}', fontweight='bold')

    # 4) uniform the ticks and save
    gs.tight_layout(fig, pad=1, h_pad=0.5, w_pad=1.5, rect=None)
    for n in range(N):
        for k in range(K + 1):
            ticklabel_format(Axes[n][k])
            if k > 0:
                Axes[n][k].set_xticks(Axes[n][0].get_xticks())
                Axes[n][k].set_xlim(Axes[n][0].get_xlim())
            start = 0 if dimension == 2 else 1
            if k > start:
                Axes[n][k].set_yticks(Axes[n][start].get_yticks())
                Axes[n][k].set_ylim(Axes[n][start].get_ylim())
    savepath = os.path.join(savedir, f'{mode:s}_SHAP_visualization')
    fig.savefig(savepath+'.jpg', dpi=dpi)
    # fig.savefig(savepath+'.svg')
    print(f'SHAP visualization saved in:\n "{savepath.replace('\\','/')+'.jpg':s}"!\n' )
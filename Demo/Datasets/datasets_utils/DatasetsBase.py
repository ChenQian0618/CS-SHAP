#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 3))


from scipy.io import loadmat, savemat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from SHAPs.plot_func import setdefault
from SHAPs.DomainTransform import func_trans_CS,func_trans_envelope


class dataset(Dataset):

    def __init__(self, list_data, transform=None):
        self.has_infos = 'infos' in list_data.columns
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        if self.has_infos:
            self.infos = list_data['infos'].tolist()
        self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        if self.transforms:
            seq = self.transforms(seq)
        if self.has_infos:
            return seq, self.labels[item], self.infos[item]
        else:
            return seq, self.labels[item], 0  # keep the same format for all datasets


def signal_plot(data, label, Fs, save_path, dpi=600):
    if os.path.split(save_path)[0] and not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    setdefault()
    func_returns = func_trans_CS()
    trans_series = func_returns[-1]['trans_series']
    out = trans_series.forward(data, verbose=True)

    func_env = func_trans_envelope()[-1]['trans_series']
    out_env = func_env.forward(data, verbose=True)


    # obtain params
    mfft, hop = trans_series[0].STFT.mfft, trans_series[0].STFT.hop
    signals =data  # (N,t)
    temp_fft = lambda x: np.abs(np.fft.fft(x - x.mean(), axis=-1)[..., :(x.shape[-1] // 2 + 1)] / x.shape[-1])
    signals_freq = temp_fft(signals)  # (N,f)
    signals_STFT = np.abs(out[0])  # (N,f2,t2)
    signals_CSCoh = np.abs(out[3][-1])  # (N,f2,a)
    signals_CSCoh[:,:,0] *= 0.3 # for better visualization
    signals_env = np.abs(out_env[4][-1])  # (N,t)

    # prepare data
    t = np.arange(signals.shape[-1]) / Fs
    f = np.arange(signals_freq.shape[-1]) * Fs / signals.shape[-1] * 1e-3
    f1 = np.arange(signals_env.shape[1]) * Fs / signals.shape[-1]
    f2 = np.arange(signals_STFT.shape[1]) * Fs / mfft * 1e-3
    pad = signals_STFT.shape[-1] - signals.shape[-1] // hop
    t2 = (np.arange(signals_STFT.shape[-1]) - (pad + 1) // 2) * hop / Fs
    a = np.arange(signals_CSCoh.shape[-1]) / signals_STFT.shape[-1] * Fs / hop

    # plot
    K = signals.shape[0]
    plt.close('all')
    fig = plt.figure(figsize=[10 / 2.54, K * 3.5 / 2.54], dpi=200)
    gs = mpl.gridspec.GridSpec(3 * K, 3)
    for k in range(K):
        ax1 = fig.add_subplot(gs[3 * k:3 * k + 1, 0])
        ax2 = fig.add_subplot(gs[3 * k + 1:3 * k + 2, 0])
        ax3 = fig.add_subplot(gs[3 * k + 2:3 * k + 3, 0])
        ax4 = fig.add_subplot(gs[3 * k:3 * (k + 1), 1])
        ax5 = fig.add_subplot(gs[3 * k:3 * (k + 1), 2])
        ax1.plot(t, signals[k])
        ax1.set_xlabel('Time $t$ (s)')
        ax1.set_ylabel('Amp.')
        ax1.margins(x=0.01)
        ax2.plot(f, signals_freq[k])
        ax2.set_xlabel('Freq. $f$ (kHz)')
        ax2.set_ylabel('Amp.')
        ax2.margins(x=0.01)
        ax3.plot(f1, signals_env[k])
        ax3.set_xlabel('Freq. $f$ (Hz)')
        ax3.set_ylabel('Amp.')
        ax3.margins(x=0.01)
        ax4.pcolormesh(t2, f2, signals_STFT[k], cmap='Blues', shading='gouraud', rasterized=True)
        ax4.set_xlabel('Time $t$ (s)')
        ax4.set_ylabel('Freq. $f$ (kHz)')
        ax4.set_title(label[k], pad=2)
        ax5.pcolormesh(a, f2, signals_CSCoh[k], cmap='Blues', shading='gouraud', rasterized=True)
        ax5.set_xlabel(r'Cyclic feq. $\alpha$ (Hz)')
        ax5.set_ylabel('Freq. $f$ (kHz)')
    fig.tight_layout(h_pad=0.5)

    # save
    fig.savefig(save_path + '.jpg', dpi=dpi)
    fig.savefig(save_path + '.pdf', dpi=dpi)
    savemat(save_path + '.mat', {'signals': signals, 'signals_freq': signals_freq, 'signals_STFT': signals_STFT,
                                    'signals_CSCoh': signals_CSCoh, 'signals_env': signals_env, 't': t, 'f': f, 'f1': f1,
                                    'f2': f2, 't2': t2, 'a': a, 'label': label, 'Fs': Fs})


import random

import numpy as np
import pandas as pd
from scipy import signal
from torch.utils.data import Dataset


seed = 999
np.random.seed(seed)


def add_noise(sig, SNR):  # add noise to sig
    noise = np.random.randn(*sig.shape)
    noise_var = sig.var() / np.power(10, (SNR / 20))
    noise = noise / noise.std() * np.sqrt(noise_var)
    return sig + noise


def check_datatype1d(type: str) -> bool:
    if type in ['time', 'fft']:
        return True
    else:
        return False


def data_transforms1d(aug, dataset_type="train", normlize_type="1-1", aug_flag=False):
    if aug_flag:
        train_trans = aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.RandomAddGaussian(),
            aug.RandomScale(),
            aug.RandomStretch(),
            aug.RandomCrop(),
            aug.Retype()
        ])
    else:
        train_trans = aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    transforms = {
        'train': train_trans,
        'val': aug.Compose([
            aug.to_numpy(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms[dataset_type]


class sig_process(object):
    nperseg = 30
    adjust_flag = False

    def __init__(self):
        super().__init__()

    @classmethod
    def time(cls, x):
        return x

    @classmethod
    def fft(cls, x):
        if len(x.shape) == 1:
            x = x - np.mean(x)
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x[1:-1] = 2 * x[1:-1]
        else:
            x = x - np.mean(x,-1)
            x = np.fft.fft(x)
            x = np.abs(x) / x.shape[-1] * 2
            x = x[:,:x.shape[-1]//2]
        return x

    @classmethod
    def slice(cls, x):
        w = int(np.sqrt(len(x)))
        img = x[:w ** 2].reshape(w, w)
        return img

    @classmethod
    def STFT(cls, x, verbose=False):
        while not cls.adjust_flag:
            _, _, Zxx = signal.stft(x, nperseg=cls.nperseg)
            if abs(Zxx.shape[0] - Zxx.shape[1]) < 2:
                cls.adjust_flag = True
            elif Zxx.shape[0] > Zxx.shape[1]:
                cls.nperseg -= 1
            else:
                cls.nperseg += 1
        f, t, Zxx = signal.stft(x, nperseg=cls.nperseg)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img

    @classmethod
    def STFT8(cls, x, Nc=8):
        f, t, Zxx = signal.stft(x, nperseg=Nc * 2 - 1, noverlap=Nc * 2 - 2)
        img = np.abs(Zxx) / len(Zxx)
        return img

    @classmethod
    def STFT16(cls, x):
        return sig_process.STFT8(x, Nc=16)

    @classmethod
    def STFT32(cls, x):
        return sig_process.STFT8(x, Nc=32)

    @classmethod
    def STFT64(cls, x):
        return sig_process.STFT8(x, Nc=64)

    @classmethod
    def STFT128(cls, x):
        return sig_process.STFT8(x, Nc=128)

    @classmethod
    def mySTFT(cls, x, verbose=False, nperseg=256, noverlap=None):
        if not noverlap:
            noverlap = nperseg // 2
        f, t, Zxx = signal.stft(x, nperseg=nperseg, noverlap=noverlap)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img

def balance_label(df, labelcol='label', number=None):
    value_count = df[labelcol].value_counts()
    if number == None or number > value_count.to_numpy().min():
        number = value_count.to_numpy().min()
    new_df = pd.concat([df.loc[df.index[df[labelcol].apply(lambda x:x == lab)].to_numpy()[:number]] for lab in value_count.index.sort_values().to_list()])
    new_df.reset_index(inplace=True)
    return new_df,number

def GetIndex_fromdataset(base_dataset, Target_number=None):  # Obtain the index where each class has the same number of samples
    Label = []
    for item, label, *info in base_dataset:
        Label.append(label)
    Label = pd.Series(Label)
    value_count = Label.value_counts()
    if Target_number == None:
        Target_number = value_count.to_numpy().min()
    else:
        Target_number = min(value_count.to_numpy().min(), Target_number)
    Index = np.concatenate([pd.array(np.arange(len(Label)))[Label.apply(lambda x:x==lab)].to_numpy()[:Target_number] for lab in
                            value_count.index.sort_values().to_list()])
    return Index, Target_number


class indexed_dataset(Dataset):  # Create a new dataset with the same number of samples for each class
    def __init__(self, base_dataset, Target_number=None):
        self.index, self.number_each_class = GetIndex_fromdataset(base_dataset, Target_number)
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        return self.base_dataset[self.index[item]]

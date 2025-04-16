import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
projecht_dir = root_path_k(__file__, 3)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)

import numpy as np
import random
from scipy.signal import resample
from Demo.Datasets.datasets_utils.generalfunciton import sig_process

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Transpose(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()

class to_numpy(object):
    def __call__(self, seq):
        #print(seq.shape)
        return np.array(seq,dtype=np.float32)

class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            sig_std = np.std(seq)*10
            return seq + np.random.normal(loc=0, scale=self.sigma*sig_std, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if self.sigma == 0:
            return seq
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if self.crop_len == 0:
            return seq
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std",'none'
        self.type = type
    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min(-1,keepdims=True))/(seq.max(-1,keepdims=True)-seq.min(-1,keepdims=True))
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min(-1,keepdims=True))/(seq.max(-1,keepdims=True)-seq.min(-1,keepdims=True)) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean(-1,keepdims=True))/seq.std(-1,keepdims=True)
        elif self.type == "none":
            pass
        else:
            raise NameError('This normalization is not included!')

        return seq

class Data_type(object):
    def __init__(self, type="time"):  # "time","fft"
        self.type = type
        self.func =getattr(sig_process,type)
    def __call__(self, seq):
        return self.func(seq)

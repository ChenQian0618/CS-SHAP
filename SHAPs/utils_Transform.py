import warnings

import numpy as np
from scipy.signal import ShortTimeFFT, hilbert
from scipy.signal.windows import gaussian


class transform(object): # the base class of all transforms
    def __init__(self):
        pass
    def forward(self, x):
        '''
        The forward transform process of the transform class
        :param x: np.ndarray
        :return: np.ndarray
        '''
        return x

    def backward(self, x):
        '''
        The backward transform process of the transform class
        :param x: np.ndarray
        :return: np.ndarray
        '''
        return x

class trans_STFT(transform):
    def __init__(self, SampFreq, hop, win_len=50, win_g=10, mfft=None, keep_N=True): # keep_N=False
        super().__init__()
        self.SampFreq = SampFreq
        self.win_len = win_len
        self.win_g = win_g
        self.keep_N = keep_N
        window = gaussian(win_len, std=win_g, sym=True)  # symmetric Gaussian window
        self.STFT = ShortTimeFFT(window, hop=hop, fs=SampFreq, mfft=mfft, scale_to='magnitude')
        self.N = None

    def forward(self, x):
        '''
        The forward transform process of the STFT class
        :param x: np.ndarray
        :return: np.ndarray
        '''
        self.N = x.shape[-1]
        if self.keep_N:
            return self.STFT.stft(x, p0=0, p1=self.STFT.p_max(self.N)+self.STFT.p_min)
        else:
            return self.STFT.stft(x)

    def backward(self, x):
        '''
        The backward transform process of the STFT class
        :param x: np.ndarray
        :return: np.ndarray
        '''
        if self.keep_N:
            offset = self.STFT.p_min * self.STFT.hop
            k0,k1 = offset, offset+self.N
        else:
            k0,k1 = 0, self.N
        # To avoid the bug (if N>ShortTimeFFT.m_num, it will be truncated and cause a bug, the bug location is in ShortTimeFFT._ifft_func return)
        res = []
        for i in range(0,x.shape[0],self.STFT.m_num):
            res.append(self.STFT.istft(x[i:i+self.STFT.m_num,...], k0=k0, k1=k1))
        return np.concatenate(res, axis=0)


class trans_De_angle(transform):
    def __init__(self, de_mean=False):
        super().__init__()
        self.de_mean = de_mean

    def forward(self, X):
        '''
        The de_angle's forward transform process
        :param X: np.ndarray or tuple
        :return: tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        result = np.abs(x)
        if self.de_mean:
            out = (np.angle(x), result.mean(-1, keepdims=True), result - result.mean(-1, keepdims=True))
        else:
            out = (np.angle(x), result)

        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        The de_angle's backward transform process
        :param X: tuple
        :return: np.ndarray or tuple
        '''
        if self.de_mean:
            remain = X[:-3]
            angle, mean, result = X[-3:]
            out = (result + mean) * np.exp(1j * angle)

        else:
            remain = X[:-2]
            angle, result = X[-2:]
            out = result * np.exp(1j * angle)
        if remain:
            return *remain, out
        else:
            return out


class trans_De_angle_pow(transform):
    def __init__(self, de_mean=False):
        super().__init__()
        self.de_mean = de_mean

    def forward(self, X):
        '''
        the de_angle_pow's forward transform process
        :param X: np.ndarray or tuple
        :return: tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        result = np.abs(x) ** 2
        if self.de_mean:
            out = np.angle(x), result.mean(-1, keepdims=True), result - result.mean(-1, keepdims=True)
        else:
            out = np.angle(x), result
        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        The de_angle_pow's backward transform process
        :param X: tuple
        :return: np.ndarray or tuple
        '''
        if self.de_mean:
            remain = X[:-3]
            angle, mean, result = X[-3:]
            out = np.sqrt(np.abs(result + mean)) * np.exp(1j * angle)
        else:
            remain = X[:-2]
            angle, result = X[-2:]
            out = np.sqrt(np.abs(result)) * np.exp(1j * angle)
        if remain:
            return *remain, out
        else:
            return out
        
class trans_Hilbert(transform):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        The Hilbert's forward transform process
        :param X: np.ndarray or tuple
        :return: np.ndarray or tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # data processing
        out = hilbert(x)
        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], out
        else:
            return out

    def backward(self, X):
        '''
        The Hilbert's backward transform process
        :return:
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # data processing
        result = x.real
        if type(X) == tuple: # return the same type as input
            return *X[:-1], np.real(result)
        else:
            return np.real(result)


class trans_FFT(transform):
    def __init__(self, one_side=True):
        '''
        applying FFT to the time domain signal, output as complex array
        :one_side: only keep the positive frequency part, which means the output is half of the input, to reduce the feature dimensions in SHAP calculation
        '''
        super().__init__()
        self.one_side = one_side
        self.odd = True

    def forward(self, X):
        '''
        The forward transform process of the FFT class: for tuple input, only the last element is transformed
        :param X: np.ndarray or tuple
        :return: np.ndarray or tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # data processing
        result = np.fft.fft(x, axis=-1) / x.shape[-1]
        self.odd = bool(x.shape[-1]%2)
        out = result[..., :result.shape[-1] // 2 + 1] if self.one_side else result
        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], out
        else:
            return out

    def backward(self, X):
        '''
        The backward transform process of the FFT class: for tuple input, only the last element is transformed
        :param X: np.ndarray or tuple
        :return: np.ndarray or tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X
        # data processing
        if self.one_side:
            temp = np.conj(x[...,-1:0:-1]) if self.odd else np.conj(x[...,-2:0:-1])
            x = np.concatenate([x, temp], axis=-1)
        result = np.fft.ifft(x, axis=-1) * x.shape[-1]
        # check the imaginary part, if it is too large, raise an error
        if np.max(np.abs(np.imag(result))) > 1e-6:
            raise ValueError('The result is not real, please check the input data')
        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], np.real(result)
        else:
            return np.real(result)


class trans_Apart(transform):
    def __init__(self, apart_combine_func=None, n=0.25):
        '''
        Split the input into two parts, and only apply SHAP to the last part to reduce the computational cost
        :param apart_combine_func: None, (func,func) respectively correspond to the transformation functions of the front and back parts
        :param n: int or float, the ratio of the first part to the whole, or the number of elements in the first part
        '''
        super().__init__()
        if apart_combine_func is None:
            if n <= 1: # n is the ratio of the first part to the whole
                self.apart_func = lambda x_temp: (
                x_temp[..., int(x_temp.shape[-1] * n):], x_temp[..., :int(x_temp.shape[-1] * n)])  # remain,x
            else: # n is the number of elements in the first part
                self.apart_func = lambda x_temp: (x_temp[..., n:], x_temp[..., :n])  # remain,x
            self.combine_func = lambda remain_temp, x_temp: np.concatenate([x_temp, remain_temp], axis=-1)  # remain
        else:
            self.apart_func, self.combine_func = apart_combine_func

    def forward(self, X):
        '''
        The apart's forward transform process: for tuple input, only the last element is transformed
        :param X: np.ndarray or tuple
        :return: tuple
        '''
        # check the input type
        if type(X) == tuple:
            x = X[-1]
        else:
            x = X

        out = self.apart_func(x)
        # return the same type as input
        if type(X) == tuple:
            return *X[:-1], *out
        else:
            return out

    def backward(self, X):
        '''
        The apart's backward transform process: for tuple input, only the last element is transformed
        :param X: tuple
        :return: tuple
        '''
        # check the input type
        remain, x = X[-2:]
        result = self.combine_func(remain, x)
        return *X[:-2], result


class trans_Patch(transform):
    def __init__(self, p=(2,)):
        '''
        Split the input into patches of size p, and regrad each patch as an individual feature to calculate contribution. Thus, the input feature dimension is reduced.
        :param p: int or (int,int), the size of the patch
        '''
        super().__init__()
        self.p = p
        self.shape = None  # record the shape of input, to be used in backward process

    def forward(self, X, verbose=False):
        '''
        The patch's forward transform process: for tuple input, only the last element is transformed
        :param X: np.ndarray=(N,...) or (A,B,...,np.ndarray)
        :return: np.ndarray = N*[p1,p2,...,pn] or np.ndarray = N*[A,B,p1,p2,...,pn]
        '''
        # check the input type
        if type(X) in [tuple,list]:
            x = X[-1]
        else:
            x = X
        # Data processing
        out = self.patch(x, p=self.p)
        self.shape = out.shape[1:] # record the shape of input, to be used in backward process
        if verbose:
            print(f"<forward>: input shape: {x.shape}\n "
                  f"output shape: {out.shape}\n output-reshape: {out.reshape(out.shape[0], -1).shape}")
        out = out.reshape(out.shape[0], -1)
        # return the same type as input
        if type(X) in [tuple, list]:
            return *X[:-1], out
        else:
            return out

    def backward(self, X, verbose=False):
        '''
        The patch's backward transform process: for tuple input, only the last element is transformed
        :param X: np.ndarray=N*(...,np.ndarray) or N*(A,B,...,np.ndarray)
        :return:
        '''
        # check the input type
        if type(X) in [tuple, list]:
            x = X[-1]
        else:
            x = X
        # Data processing
        out = self.unpatch(x.reshape(-1, *self.shape))
        if verbose:
            print(f"<backward>: input shape: {x.shape}\n input reshape: {x.reshape(-1, *self.shape).shape}\n " +
                  f"output-reshape: {out.shape}")
        # return the same type as input
        if type(X) in [tuple, list]:
            return *X[:-1], out
        else:
            return out

    @staticmethod
    def patch(X:np.ndarray, p=2, axis=-1):
        '''
        Split the input into patches of size p
        if the length of the input is n, the output length is np.ceil(n/p)
        :X: input data np.ndarray
        :p: split length int or (int,int), default works on the last N dimensions
        :axis: split dimension int or (int,int)
        :return: np.ndarray with dtype=object
        e.g., X.shape=(n,10,20), p=(5,2) -> out.shape=(n,2,10), out[0,0,0].shape=(5,2)
        '''
        # if p is a tuple or list, then reuse the patch function
        if type(p) in [tuple, list]:
            if type(axis)==int: # if axis is int, then axis is set to the last len(p) dimensions
                axis = np.arange(-len(p),0).tolist()
            if (type(axis) not in [tuple, list]) or len(axis) != len(p): # check the length of p and axis
                raise ValueError('The length of p and axis should be equal')
            # reverse the order of p and axis, to make the patch function work in the right order
            for item_p,item_a in zip(p[::-1], axis[::-1]): # sorted(list(zip(p, axis)),key=lambda x:x[-1],reverse=True)
                X = trans_Patch.patch(X, item_p, item_a)
            return X
        # switch axis, to make the patch function work in the right order
        if axis != -1:
            X = np.swapaxes(X, axis, -1)
        # Data processing
        out = np.zeros(X.shape[:-1] + (int(np.ceil(X.shape[-1]/p)),),dtype=object)
        for i in np.arange(0, X.shape[-1], p):
            item = X[..., i:i + p]
            out[..., i // p] = convert_last_Ndim_to_object(item) # convert the last N dimensions to object type
        # restore axis
        if axis != -1:
            out = np.swapaxes(out, axis, -1)
        return out

    @staticmethod
    def unpatch(X:np.ndarray, axis=None):
        '''
        Unpatch the input data, and convert the object type to np.ndarray
        :X: input data np.ndarray
        :axis: int or None, default works on the last dimensions
        :return: np.ndarray
        e.g., X.shape=(n,2,10), X[0,0,0].shape=(5,2) -> out.shape=(n,10,20)
        '''
        # default works on the last n dimensions, where n is the dimensions of the object element
        if axis is None:
            n = X.flat[0].ndim # the dimensions of the object element
            axis = (np.arange(n) - n).tolist()
        if type(axis) in [tuple, list]: # if axis is a tuple or list, then reuse the unpatch function
            for item_a in axis:
                X = trans_Patch.unpatch(X, item_a)
            return X
        # switch axis, to make the unpatch function work in the right order
        if axis != -1:
            X = np.swapaxes(X, axis, -1)
        # Data processing
        out = []
        for i in range(X.shape[-1]):
            temp = convert_last_Ndim_to_object_reverse(X[..., i], N=1)
            out.append(temp)
        out = np.concatenate(out, axis=-1)
        # restore axis
        if axis != -1:
            out = np.swapaxes(out, axis, -1)
        return out



def convert_last_Ndim_to_object(array,N:int=1):
    '''
    Convert the last N dimensions of the array to object type
    :param array: input, np.ndarray
    :param N: the number of dimensions to be converted, int
    :return:
    '''
    # record the shape of the input array
    shape = array.shape
    # the new shape is the original shape minus the last N dimensions
    new_shape = shape[:-N]
    # create a new array with the same shape as the original array, but with the last N dimensions replaced by object type
    new_array = np.empty(new_shape, dtype=object)
    # use np.ndenumerate to iterate over the new array, and convert the last N dimensions of the original array to object type
    for index, _ in np.ndenumerate(new_array):
        item = array[index]
        if type(item) == np.ndarray and item.dtype == np.object_: # if the item is already an object type, then convert it to np.ndarray
            item = np.array([np.array(temp) for temp in item]) # convert the object type to np.ndarray
        new_array[index] = item # assign the object type to the new array
    return new_array

def convert_last_Ndim_to_object_reverse(array,N=None):
    '''
    The inverse process of convert_last_Ndim_to_object, convert the object type to np.ndarray
    :param array: input, np.ndarray with object type
    :param N: int or None, the number of dimensions to be converted, default is None
    :return:
    '''
    # check the input type
    if type(array.flat[0]) != np.ndarray:
        raise ValueError('The element of input array should be ndarray')
    if N is None:
        N = array.flat[0].ndim
    if N > array.flat[0].ndim:
        raise ValueError('The N should be less than the dimension of the element of input array')
    # the new shape is the original shape plus the last N dimensions
    new_shape = array.shape + array.flat[0].shape[:N]
    # create a new array, if N is less than the dimension of the element of input array, then the new array is of type object, otherwise it is of type array.flat[0].dtype
    new_array = np.empty(new_shape, dtype=object) if N < array.flat[0].ndim else np.empty(new_shape, dtype=array.flat[0].dtype)
    # use np.ndenumerate to iterate over the new array, and convert the object type to np.ndarray
    for index, _ in np.ndenumerate(array):
        item = array[index]
        if N < item.ndim: # convert item into the same shape as the new array
            new_array[index] = convert_last_Ndim_to_object(item, item.ndim-N)
        else: # new_array[index].shape==item.shape, then assign the item to the new array
            new_array[index] = item
    return new_array


class trans_Object_Combine(transform):
    '''
    For Tuple/List input, convert the first N-1 elements to object type, and combine them with the last element
    :return: np.ndarray with dtype=object
    '''
    def __init__(self):
        super().__init__()
        self.num_Prepart = 0 # the number of elements to be converted to object type, default is 0

    def forward(self, X):
        '''
        the  forward transform process of the Object_Combine class
        :param X: np.ndarray or tuple
        '''
        self.num_Prepart = len(X) - 1 if type(X) in [tuple, list] else 0
        # check the input type
        if self.num_Prepart > 0:
            out = X[-1]
            Pre_part = X[:-1]
            Pre_part = [convert_last_Ndim_to_object(item,N=item.ndim-1)[..., np.newaxis] for item in Pre_part] # Convert to object array [[N,1],[N,1],...]
            return np.concatenate(Pre_part + [out,], axis=-1)
        else:
            return X

    def backward(self, X, num_Prepart=False):
        '''
        The backward transform process of the Object_Combine class
        :param X: np.ndarray or tuple
        :param num_Prepart: int or False, the number of elements to be converted to object type, default is False
        :return: np.ndarray or tuple
        '''
        if num_Prepart is not False:
            self.num_Prepart = num_Prepart
        # check the input type
        if self.num_Prepart > 0:
            x = X[:, self.num_Prepart:]
        else:
            x = X
        # Data processing
        if self.num_Prepart > 0:
            Pre_part = X[:, :self.num_Prepart]
            Pre_part = [convert_last_Ndim_to_object_reverse(Pre_part[:, i]) for i in range(Pre_part.shape[-1])]
            return tuple(Pre_part + [X[:, self.num_Prepart:],])
        else:
            return X


class trans_Series(transform):
    '''
    combine multiple transforms into a series, and apply them in order
    :param args: list of transforms
    '''
    def __init__(self,*args):
        super().__init__()
        if len(args) == 1 and type(args[0]) in [tuple, list]:
            self.trans_list = args[0]
        else:
            self.trans_list = args
    def __len__(self):
        return len(self.trans_list)

    def __getitem__ (self,item):
        return self.trans_list[item]

    def show(self): 
        '''
        show the names of the transforms in the series
        :return: list of transform names
        '''
        names = []
        for trans in self.trans_list:
            names.append(trans.__class__.__name__)
        return names

    def forward(self, x, N=None,verbose=False):
        '''
        The forward transform process of the Series class
        :param x: np.ndarray or tuple
        :param N: int or None, the number of transforms to be applied, default is None, which means all transforms will be applied
        :param verbose: bool, if True, print the output shape and dtype of each transform
        '''
        N = N if N is not None and type(N) == int else len(self.trans_list)
        X = []
        for i,trans in enumerate(self.trans_list[:N]):
            x = trans.forward(x)
            if verbose:
                shape = x.shape if hasattr(x,'shape') else len(x)
                dtype = x.dtype if hasattr(x,'dtype') else x[0].dtype
                print(f"<forward> {i}-th layer({trans.__class__.__name__}): output shape= {shape}, dtype= {dtype}")
                if type(x) == tuple:
                    print(' | '.join([f'output[{j:d}] shape: {x[j].shape}' for j in [0,1,-1]]), 'dtype: ', x[-1].dtype)
            X.append(x)
        return X if verbose else x
    
    def backward(self, x, N=None, verbose=False):
        '''
        The backward transform process of the Series class
        :param x: np.ndarray or tuple
        :param N: int or None, the number of transforms to be applied, default is None, which means all transforms will be applied
        :param verbose: bool, if True, print the output shape and dtype of each transform
        '''
        N = N if N is not None and type(N) == int else len(self.trans_list)
        X = []
        for i,trans in enumerate(self.trans_list[::-1][:N]):
            x = trans.backward(x)
            if verbose:
                shape = x.shape if hasattr(x,'shape') else len(x)
                dtype = x.dtype if hasattr(x, 'dtype') else x[0].dtype
                print(
                    f"<backward> {len(self.trans_list) - 1 - i}-th layer({trans.__class__.__name__}): output shape= {shape}, dtype= {dtype}")
                if type(x) == tuple:
                    print(' | '.join([f'output[{i:d}] shape: {x[i].shape}' for i in [0, 1, -1]]), 'dtype: ', x[-1].dtype)
                X.append(x)
        return tuple(X) if verbose else x

    def __getitem__(self, item):
        return self.trans_list[item]

    def __len__(self):
        return len(self.trans_list)

    def __repr__(self):
        res = []
        for i, trans in enumerate(self.trans_list):
            res.append(f'{i + 1:2d}-th layer: {trans.__class__.__name__:s}')
        return '\n'.join(res)


if __name__ == '__main__':
    # test STFT
    x = np.random.randn(60, 2000)
    trans = trans_STFT(10000, 10, keep_N=True)  # keep_N=True
    y = trans.forward(x)
    x_recon = trans.backward(y)
    print(f"------------test STFT:---------------------")
    print('<without margin>:')
    print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x-x_recon)):g} | mean error: {np.mean(np.abs(x-x_recon)):g}")
    trans_stft_noKeep = trans_STFT(10000, 10, keep_N=False) # keep_N=True
    y_noKeep = trans_stft_noKeep.forward(x)
    x_recon_noKeep = trans_stft_noKeep.backward(y_noKeep)
    print('<with margin>:')
    print(f"input shape: {x.shape}\n output shape: {y_noKeep.shape}\n recon shape: {x_recon_noKeep.shape}")
    print(f"max error: {np.max(np.abs(x-x_recon_noKeep)):g} | mean error: {np.mean(np.abs(x-x_recon_noKeep)):g}")


    # test de_angle
    x = np.array([1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]).reshape(2, -1)
    trans_de_angle = trans_De_angle(de_mean=True)  # de_mean=True
    y = trans_de_angle.forward(x)
    x_recon = trans_de_angle.backward(y)
    print(f"------------test de_angle:---------------------")
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]) +
          f"\nrecon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test de_angle_pow
    x = np.array([1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]).reshape(2, -1)
    trans_de_angle_pow = trans_De_angle_pow(de_mean=True)  # de_mean=True
    y = trans_de_angle_pow.forward(x)
    x_recon = trans_de_angle_pow.backward(y)
    print(f"------------test de_angle_pow:---------------------")
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]) +
          f"\nrecon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")


    # test hilbert
    x = np.random.randn(2, 2000)
    trans_hilbert = trans_Hilbert()
    y = trans_hilbert.forward(x)
    x_recon = trans_hilbert.backward(y)
    print(f"------------test hilbert:---------------------")
    print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test fft
    x = np.random.randn(2, 26, 101)
    # x = ([1],x)
    trans_fft = trans_FFT(one_side=True)  # one_side=True
    y = trans_fft.forward(x)
    x_recon = trans_fft.backward(y)
    print(f"------------test fft:---------------------")
    if type(x) == tuple:
        print(f"input shape: {x[-1].shape}\n output shape: {y[-1].shape}\n recon shape: {x_recon[-1].shape}")
        print(f"max error: {np.max(np.abs(x[-1] - x_recon[-1])):g}")
    else:
        print(f"input shape: {x.shape}\n output shape: {y.shape}\n recon shape: {x_recon.shape}")
        print(f"max error: {np.max(np.abs(x - x_recon)):g}")

    # test apart
    x = (np.random.randn(2, 26, 101), np.random.randn(2, 26, 1), np.random.randn(2, 26, 51))
    trans_apart = trans_Apart(n=0.5)
    y = trans_apart.forward(x)
    x_recon = trans_apart.backward(y)
    print(f"------------test apart:---------------------")
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]))
    print(f'output[-1][0,0] shape: {y[-1][0, 0].shape} | output[-1][0,-1] shape: {y[-1][0, -1].shape}')
    print(f'max error: {[np.max(np.abs(x[i] - x_recon[i])) for i in range(len(x))]}')

    # test Patch
    print(f"------------test Patch:---------------------")
    x = (np.random.randn(2, 26, 101), np.random.randn(2, 26, 1), np.random.randn(2, 26, 51))
    p = (5, 5)
    trans_patch = trans_Patch(p=p)
    y = trans_patch.forward(x, verbose=True)
    x_recon = trans_patch.backward(y, verbose=True)
    print(f"--- with patch: p={p}")
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(' | '.join([f'output-{i:d} shape: {item.shape}' for i, item in enumerate(y)]))
    print(f'output[-1][0,0] shape: {y[-1][0,0].shape} | output[-1][0,-1] shape: {y[-1][0,-1].shape}')
    print(f'max error: {[np.max(np.abs(x[i]-x_recon[i])) for i in range(len(x))]}')

    print(f"------------test Object Combine:---------------------")
    trans_object_combine =trans_Object_Combine()
    x = (np.random.randn(2, 26, 101),np.random.randn(2, 26, 1),convert_last_Ndim_to_object(np.random.randn(2, 66, 2),1))
    y2 = trans_object_combine.forward(x)
    x_recon = trans_object_combine.backward(y2)
    print(' | '.join([f'input-{i:d} shape: {item.shape}' for i, item in enumerate(x)]))
    print(f'output shape: {y2.shape}')
    print(' | '.join([f'output[0,{i:d}] shape: {y2[0,i].shape}' for i in [0,1,-1]]))
    # print(f'max error: {[np.max(np.abs(x[i] - x_recon[i])) for i in range(len(x))]}')

    # test Series
    print(f"------------test Series:---------------------")
    trans_series = trans_Series(trans,trans_de_angle,trans_fft, trans_patch,trans_Object_Combine())
    x = np.random.randn(2,int(1e3))
    y = trans_series.forward(x,verbose=True)
    print('-'*30)
    x_recon = trans_series.backward(y[-1],verbose=True)
    print(f"input shape: {x.shape}\n" +
          ' | '.join([f'output-{i:d} shape: {item.shape}' for i,item in enumerate(y[-1])]) +
          f'\nrecon shape: {x_recon[-1].shape}')
    print(f"max error: {np.max(np.abs(x-x_recon[-1])):g} | mean error: {np.mean(np.abs(x-x_recon[-1])):g}")


    #-----------------------------test object function and path ---------------------------------
    # test convert_last_Ndim_to_object
    x = np.random.randn(2,10,20)
    y1 = convert_last_Ndim_to_object(x)
    y2 = convert_last_Ndim_to_object(y1)
    y1_recon = convert_last_Ndim_to_object_reverse(y2,N=1)
    x_recon = convert_last_Ndim_to_object_reverse(y1_recon)
    print(f"------------test convert_last_Ndim_to_object:---------------------")
    print(f"input shape: {x.shape}\n" +
          f"output-1 shape: {y1.shape} | output-1[0,0] shape: {y1[0,0].shape}" +
          f"\n output-2 shape: {y2.shape} | output-2[0] shape: {y2[0].shape}")
    print(f"recon_output-1 shape: {y1_recon.shape} | recon_output-1[0,0] shape: {y1_recon[0,0].shape}\n"+
          f"recon input shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x-x_recon)):g}")

    # test trans_Patch.patch
    # x = np.random.randn(2,4,5)
    x = np.arange(2*3*5).reshape(2,3,-1)
    trans_patch = trans_Patch()
    p = (2,2)
    y1 = trans_patch.patch(x, p,)
    x_recon = trans_patch.unpatch(y1)
    print(f"------------test trans_Patch.patch:---------------------")
    print(f"input shape: {x.shape} with patch :{p} \n" +
          f"output-1 shape: {y1.shape} \n" +
          f" output-1[0,0,0] shape: {y1[0,0,0].shape} | output-1[0,-1,0] shape: {y1[0,-1,0].shape} | output-1[0,-1,-1] shape: {y1[0,-1,-1].shape}")
    print(f"recon input shape: {x_recon.shape}")
    print(f"max error: {np.max(np.abs(x-x_recon)):g}")
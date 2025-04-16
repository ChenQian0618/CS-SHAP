import sys,os
def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 1))

import SHAPs.utils_Transform as transform
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

stft_params = {'SampFreq': 10000, 'hop': 10, 'win_len': 50, 'keep_N': False}
trans_stft = transform.trans_STFT(stft_params['SampFreq'], stft_params['hop'],
                                  win_len=stft_params['win_len'], keep_N=stft_params['keep_N'])

def pick(x): return x[-1] if type(x) == tuple else x

def func_unpatch_base(data, shap_value=None, trans_series=None, back_N=0):
    '''
    Unpatch the data (and shap_value) to the target domain. (avoid the effect of transform.patch & transform.combine)
    :param data: [N,Feature] object
    :param shap_value: [N,Feature] object
    :trans_series: transform series object
    :param back_N: the number of backward transform
    :return: [N,feature,....]  in the target domain
    '''
    out_data = trans_series.backward(data, N=back_N)
    if shap_value is not None:
        # [N,Feature,y] float -> [y,N,Feature] object
        new_shap_value = ShapValue_reshape(shap_value, data)

        # [y,N,Feature] object ->  [y*N,Feature] object
        temp = new_shap_value.reshape(-1, new_shap_value.shape[-1])

        # [y*N,Feature] object ->k * [y*N,Feature] float, k is the number of [phase, mean, values]
        out_shap_value = trans_series.backward(temp, N=back_N)

        # for special case: [phase, mean, values]
        if type(out_shap_value) == tuple:
            # k * [y,N,Feature] object -> k * [y,N,d1,d2...] float
            out_shap_value = tuple(item.reshape(new_shap_value.shape[:2] + out_shap_value[i].shape[1:]) for i, item in
                                   enumerate(out_shap_value))
        else:  # for common case: values
            # [y,N,Feature] object -> [y,N,d1,d2...] float
            out_shap_value = out_shap_value.reshape(
                new_shap_value.shape[:2] + out_shap_value.shape[1:])
        return out_data, out_shap_value
    else:
        return out_data, None


def ShapValue_reshape(shap_values, data):
    '''
    Reshape the SHAP values to match the data shape by convert the float to multi-item-object.  (the dtype of shap_values is converted from float to object!)
    :param shap_values: [N,Feature,y] float
    :param data: [N,Feature] object
    :return: [y,N,Feature] object
    '''
    shape = shap_values.shape
    out = np.zeros(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                out[i, j, k] = np.ones_like(
                    abs(data[i, j])) * shap_values[i, j, k]
    out = np.moveaxis(out, -1, 0)
    return out


# ------------------------------    time domain  ------------------------------
def func_trans_time(p = (3,),):
    '''
    :param p: the patch size used in transform.patch, caculate the total contribution of multyple features (i.e., one patch) to facilitate the speed. 
    :return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    name = 'time'
    trans_series = transform.trans_Series(
        [transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value=None): return func_unpatch_base(
        data, shap_value=shap_value, trans_series=trans_series, back_N=2)
    
    def get_plot_params(signal_len, Fs=1):
        x = np.arange(signal_len) / Fs
        return {'x': x, 'Fs': Fs, 'x_label': 'Time $t$ (s)', 'y_label': 'Value'}
    
    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series, 'get_plot_params': get_plot_params}


# ------------------------------ frequency  domain  ------------------------------
def func_trans_frequency(p = (3,),):
    '''
    obtain the frequency domain representation of the signal ( |FFT| )
    :param p: the patch size used in transform.patch, caculate the total contribution of multyple features (i.e., one patch) to facilitate the speed.
    :return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    name = 'frequency'
    trans_series = transform.trans_Series(
        [transform.trans_FFT(one_side=True),
         transform.trans_De_angle(),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value=None):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        return pick(out_data), pick(out_value)

    def get_plot_params(signal_len, Fs=1):
        signal = np.ones([1, signal_len],)
        out = func_unpatch(trans_series.forward(signal))[0]
        x = np.arange(out.shape[-1]) / signal_len * Fs * 1e-3
        return {'x': x, 'Fs': Fs, 'x_label': 'Spectral freq. $f$ (kHz)', 'y_label': 'Value'}
    
    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series, 'get_plot_params': get_plot_params}


# ------------------------------ envelope  domain  ------------------------------
def func_trans_envelope(p = (1,),):
    '''
    obtain the envelope domain representation of the signal ( Hilbert -> de_angle -> FFT -> apart -> de_angle )
    :param p: the patch size used in transform.patch, caculate the total contribution of multyple features (i.e., one patch) to facilitate the speed.
    :return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    name = 'envelope'
    trans_series = transform.trans_Series(
        [transform.trans_Hilbert(),
         transform.trans_De_angle(de_mean=True),
         transform.trans_FFT(one_side=True),
         transform.trans_Apart(n=0.12),# only consider low frequency for analyzing the envelope (n=0.12 is the percentage)
         transform.trans_De_angle(),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value=None):
        out_data, out_value = func_unpatch_base(
            data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        return pick(out_data), pick(out_value)
    
    def get_plot_params(signal_len, Fs=1):
        signal = np.ones([1, signal_len],)
        out = func_unpatch(trans_series.forward(signal))[0]
        x = np.arange(out.shape[-1]) / signal_len * Fs
        return {'x': x, 'Fs': Fs, 'x_label': r'Cyclic freq. $\alpha$ (Hz)', 'y_label': 'Value'}

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series, 'get_plot_params': get_plot_params}


# ------------------------------    STFT  domain ------------------------------
def func_trans_STFT(p = (1, 5),):
    '''
    obtain the Time-Frequency domain representation of the signal ( |STFT|)
    :param p: the patch size used in transform.patch, caculate the total contribution of multyple features (i.e., one patch) to facilitate the speed. # (1,5) | (1,10)
    :return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    name = 'STFT'
    trans_series = transform.trans_Series(
        [trans_stft,
         transform.trans_De_angle(de_mean=False),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value=None):
        out_data, out_value = func_unpatch_base(
            data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        return pick(out_data), pick(out_value)
    
    def get_plot_params(signal_len, Fs=1):
        mfft, hop = trans_series[0].STFT.mfft, trans_series[0].STFT.hop
        signal = np.ones([1, signal_len],)
        out = func_unpatch(trans_series.forward(signal))[0]
        y = np.arange(out.shape[-2]) / mfft * Fs * 1e-3
        pad = out.shape[-1] - signal_len / hop
        x = (np.arange(out.shape[-1]) - pad // 2)  / Fs * hop
        return {'x': x, 'y': y, 'Fs': Fs, 'x_label': 'Time $t$ (s)', 'y_label': 'Spectral freq. $f$ (kHz)'}
    
    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series, 'get_plot_params': get_plot_params}


# ------------------------------   CSCoh  domain ------------------------------
def func_trans_CS(p = (1, 3)):
    '''
    obtain the CS domain representation of the signal ( |STFT|^2 -> FFT+de_angle -> CSCoh)
    :param p: the patch size used in transform.patch, caculate the total contribution of multyple features (i.e., one patch) to facilitate the speed.
    :return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''

    name = 'CS'
    trans_series = transform.trans_Series(
        [trans_stft,
         transform.trans_De_angle_pow(de_mean=False),
         transform.trans_FFT(one_side=True),
         transform.trans_De_angle(de_mean=False),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])

    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value=None):
        out_data, out_value = func_unpatch_base(
            data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        return pick(out_data), pick(out_value)

    def get_plot_params(signal_len, Fs=1):
        mfft, hop = trans_series[0].STFT.mfft, trans_series[0].STFT.hop
        signal = np.ones([1, signal_len],)
        stft_shape = trans_series[0].forward(signal).shape
        out = func_unpatch(trans_series.forward(signal))[0]
        y = np.arange(out.shape[-2]) / mfft * Fs * 1e-3
        x = np.arange(out.shape[-1]) / stft_shape[-1] * Fs / hop
        return {'x': x, 'y': y, 'Fs': Fs, 'x_label': r'Cyclic freq. $\alpha$ (Hz)', 'y_label': 'Spectral freq. $f$ (kHz)'}
    
    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series, 'get_plot_params': get_plot_params}




if __name__ == '__main__':
    # 1) data
    inputs_ana = np.random.randn(*[2, 2000])
    # 2) test each transform function
     # func_trans_time, func_trans_frequency,func_trans_envelope, func_trans_STFT, func_trans_CSCoh
    func_lists = [func_trans_time, func_trans_frequency, func_trans_envelope, func_trans_STFT, func_trans_CS]
    
    for func in func_lists:
        func_Z, func_Z_inv, func_unpatch, trans_dict = func()
        print('\n'*2, '-' * 50, ' ' * 2 + trans_dict['name'] + ' ' * 2, '-' * 50)
        print('inputs.shape: ', inputs_ana.shape)
        y = func_Z(inputs_ana, verbose=True)
        print('')
        y_inv = func_Z_inv(y[-1], verbose=True)
        print('mean error: ', abs(y_inv[-1] - inputs_ana).mean())

        print('\n','<test unpatch function to obtain the presentation in the target domain>:')
        data = y[-1]
        values = np.random.randn(*data.shape, 3)
        out_data, out_value = func_unpatch(data, values)
        out_data, out_value = pick(out_data), pick(out_value)
        print(f'out_data.shape: {out_data.shape}, out_value.shape: {out_value.shape}')

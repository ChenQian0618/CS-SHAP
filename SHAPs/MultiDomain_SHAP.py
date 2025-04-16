'''
2025/04/13
Created by ChenQian-SJTU 
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 1))

import numpy as np
import copy, pickle, time
import torch
import shap

from SHAPs.DomainTransform import func_trans_time, func_trans_frequency, func_trans_envelope, func_trans_STFT, func_trans_CS
from SHAPs.utils_SHAP_MyIndependent import MyIndependent
from SHAPs.utils_Visualization import attr_visualization

class MultiDomain_SHAP(object):
    def __init__(self, func_predict, background_data, save_dir):
        self.func_predict = func_predict
        self.background_data = background_data
        self.save_dir = save_dir 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def explain(self, input_data, input_label, mode='CS', preload=True, Fs=1):
        # deepcopy the input data
        input_data = copy.deepcopy(input_data) 
        # preparation
        func_Z, func_Z_inv, func_unpatch, trans_dict = self.mode_select(mode)
        func_predict_z = lambda x: self.func_predict(func_Z_inv(x))
        savedir = self.save_dir
        rawfile_savepath = os.path.join(savedir, f'{mode:s}_SHAP_values_raw.pkl')

        # 1) test
        z = func_Z(input_data, verbose=True)
        print('',end='\n')
        z_inv = func_Z_inv(z[-1], verbose=True)
        print('mean error: ', abs(z_inv[-1] - input_data).mean())
        predict_ana = func_predict_z(z[-1])
        print('label of input sample:', np.argmax(predict_ana, -1))

        # 2) Try to preload the SHAP values
        preload_OK = False
        if os.path.exists(rawfile_savepath) and preload:
            with open(rawfile_savepath, 'rb') as f:
                save_dict = pickle.load(f)
            raw_shap_values = save_dict['raw_shap_values']
            inputs_load = save_dict['input_data']
            print('inputs_load[0,0]: ', inputs_load[0][0])
            preload_OK = True if inputs_load.shape == input_data.shape and \
                                 np.equal(inputs_load, input_data).all() and \
                                 (func_Z(inputs_load).shape == raw_shap_values.data.shape) else False
            plot_params = trans_dict['get_plot_params'](input_data.shape[-1], Fs=Fs)
            save_dict['plot_params'] = plot_params
            with open(rawfile_savepath, 'wb') as f:
                pickle.dump(save_dict, f)
        print(f'Preload: True | analyze time of per sample: {save_dict["analyse_time"]/input_data.shape[0]:.1f}s'
              if preload_OK else 'Preload: False')
        # 3) if preload failed, calculate the SHAP values
        if not preload_OK:
            plot_params = trans_dict['get_plot_params'](input_data.shape[-1], Fs=Fs)
            # 3.1) conduct SHAP analysis
            Z = func_Z(self.background_data) # preprocessing the background data
            background_masker = MyIndependent(Z) 
            explainer = shap.Explainer(func_predict_z, background_masker, algorithm='permutation') # algorithm='exact' is impossible for large data
            start_time = time.time()
            print('Start SHAP analysis, please wait...(several minutes / hours):')
            raw_shap_values = explainer(func_Z(input_data),
                                    max_evals=max(int(1e3), int(Z.shape[-1] * 10) // 2))  # the calculation of SHAP
            analyse_time = time.time() - start_time
            print(f'Analyse time of {input_data.shape[0]:d} samples: {analyse_time:.1f}s')

            # 3.2) unpatch the SHAP values to the original domain
            domain_data, domain_shap_value = func_unpatch(raw_shap_values.data, raw_shap_values.values)
            domain_data = np.abs(
                domain_data) if 'omplex' in domain_data.dtype.__class__.__name__ else domain_data  # avoid complex situation
            domain_shap_value = np.real(
                domain_shap_value) if 'omplex' in domain_shap_value.dtype.__class__.__name__ else domain_shap_value  # avoid complex situation

            # 3.3) save SHAP result
            predict_logit = func_predict_z(func_Z(input_data))
            predict_prob = torch.nn.functional.softmax(torch.tensor(predict_logit), -1).numpy()
            print('(saved) input_data[0,0]: ', input_data[0][0])
            save_dict = {'raw_shap_values': raw_shap_values, 
                          'domain_shap_value': domain_shap_value, 'domain_data': domain_data,
                          'plot_params': plot_params, 'mode': mode,
                         'input_data': input_data, 'input_label': input_label,
                         'predict_logit': predict_logit, 'predict_prob': predict_prob,
                         'analyse_time': analyse_time, }
            with open(rawfile_savepath, 'wb') as f:
                pickle.dump(save_dict, f)

        # 4) save visualization data
        attr_visualization(savedir=savedir,mode=mode, data= save_dict['domain_data'],
                           value= save_dict['domain_shap_value'],
                           plot_params=save_dict['plot_params'],
                           label= save_dict['input_label'],predict=save_dict['predict_prob'])

    def mode_select(self, mode):
        mode_map = {'time': func_trans_time,
                    'frequency': func_trans_frequency,
                    'envelope': func_trans_envelope,
                    'STFT': func_trans_STFT,
                    'CS': func_trans_CS, }
        if mode not in mode_map.keys():
            raise ValueError('mode should be in ', mode_map.keys())
        func_trans = mode_map[mode]
        func_Z, func_Z_inv, func_unpatch, trans_dict = func_trans()
        '''
        func_Z: z=func_Z(x), the transform function
        func_Z_inv: x=func_Z_inv(z), the inverse transform function
        func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
        trans_dict: {'name':.,'trans_series':.}                
        '''
        return func_Z, func_Z_inv, func_unpatch, trans_dict
    
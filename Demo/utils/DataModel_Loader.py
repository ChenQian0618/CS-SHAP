import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
projecht_dir = root_path_k(__file__, 2)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)


import pickle
import numpy as np
import torch

import Demo.Datasets as datasets
import Demo.Models as models
from Demo.utils.general_func import find_all_index_N
from Demo.utils.PostProcessLib import ExtractInfo
from SHAPs.plot_func import setdefault

class DataModel_Loader(object):
    def __init__(self, dir=None, flag_preload_dataset=True):
        '''
        :param dir: the directory of the model
        :param flag_preload_dataset: whether to preload the saved dataset during training, if not, reload the new dataset
        '''
        setdefault()

        self.flag_load_model_weight = True  # whether load the model weight
        self.flag_preload_dataset = flag_preload_dataset  # whether load the model weight
        self.data_train_not_val = False  # whether use the training data or the validation data

        self.device = torch.device('cpu')
        self.dir = dir
        self.args, Dict = ExtractInfo(os.path.join(self.dir, "training.log"))
        # args process
        self.args['SNR'] = None if self.args['SNR'] == 'None' else float(self.args['SNR'])
        self.args['test_size'] = None if self.args['test_size'] == 'None' else float(self.args['test_size'])
        # set postprocess dir
        self.save_dir = os.path.join(self.dir, 'PostProcess_of_SHAP_Analysis')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # load data and model
        self._get_all_data(self.flag_preload_dataset)  # self.data, self.label, self.label_name, self.Fs
        self._model_load()  # self.model

    def _get_all_data(self, flag_preload_dataset, Target_number=100):
        print('------begin data load--------')
        phase = 'train' if self.data_train_not_val else 'val'

        data_path = os.path.join(self.dir, f'all_dataset_{phase:s}_SavedInTraining.pkl')

        # try to load the data
        if flag_preload_dataset and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                temp = pickle.load(f)
            print('data load done! (by preload)')
        else:
            # Load the datasets

            subargs = {k: self.args[k] for k in ['data_dir', 'data_type', 'normlizetype', 'test_size'] +
                       ['data_name', 'data_signalsize', 'SNR']}
            Dataset = getattr(datasets, subargs['data_name'])
            (train_dataset, val_dataset), label_name = Dataset(subargs).data_preprare(
                signal_size=int(subargs['data_signalsize']), SNR=subargs['SNR'], try_preload=True)

            # collect data
            phases = ['val', 'train'] if self.data_train_not_val else ['train',
                                                                       'val']  # do both, but select train/val in the end
            for temp_phase in phases:  # do both
                ToBeCollect = train_dataset if temp_phase == 'train' else val_dataset
                temp_data_path = os.path.join(os.path.split(data_path)[0], f'all_dataset_{temp_phase:s}.pkl')
                data, label, infos = [], [], []
                for i in range(len(ToBeCollect)):
                    item, label_, *info = ToBeCollect[i]
                    data.append(item)
                    label.append(label_)
                    infos.append(info)
                data, label, infos = np.array(data).squeeze(), np.array(label).squeeze(), np.array(infos)
                Index = find_all_index_N(label, Target_number)

                temp = {'data': data[Index, ...], 'label': label[Index, ...], 'infos': infos[Index, ...],
                        'label_name': [item.strip().capitalize() for item in label_name],
                        'Fs': Dataset.Fs, 'subargs': subargs}
                with open(temp_data_path, 'wb') as f:
                    pickle.dump(temp, f)
            print('data load done! (by dataset)')
        self.data, self.label, self.label_name = temp['data'], temp['label'].squeeze(), temp['label_name']
        self.infos = temp['infos']
        print('data[0,0]: ', temp['data'][0][0])
        self.Fs = temp['Fs']


    def _model_load(self):
        # search
        print('------begin model load--------')
        model_name = self.args['model_name']
        # Define the models
        Dataset = getattr(datasets, self.args['data_name'])
        self.model = getattr(models, model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        # Invert the models and define the loss
        self.model.to(self.device)
        if self.flag_load_model_weight:
            tempfiles = next(os.walk(self.dir))[2]
            chose_one = [i for i in tempfiles if i.endswith('.pth')]  # choose .pth
            if len(chose_one) == 0:
                raise ValueError('No model weight file found!')
            chose_one = [item for item in chose_one if 'best' in item][0]  # choose best or final
            storage = torch.load(os.path.join(self.dir, chose_one))
            self.model.load_state_dict(storage['state_dict'])
        print('model load done!')
        self.model.eval()  # set the model to evaluation mode

    def get_fuc_data(self):
        # determine the analysis data
        index2 = find_all_index_N(self.label, 1)
        print('The index of selected data for analysis', index2.tolist())
        input_data = self.data[index2, ...]
        input_label = self.label[index2, ...]
        # infos_ana = self.infos[index2, ...]

        # determine the background data
        Index = find_all_index_N(self.label, squeeze=False)[:, -5:].reshape(-1)
        background_data = self.data[Index, ...]  # for SHAP analysis (as background)
        background_label = self.label[Index, ...]

        # determine the prediction function
        def func_predict(input, device=self.device):
            self.model.eval()  # set the model to evaluation mode
            input = input if len(input.shape) == 3 else input[:, np.newaxis, ...]
            input = torch.tensor(input, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = self.model(input)
            return logits.detach().cpu().numpy()
        
        return func_predict, background_data, background_label, input_data, input_label

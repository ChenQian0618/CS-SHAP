import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
projecht_dir = root_path_k(__file__, 2)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)


import pickle
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import Demo.Datasets.datasets_utils.sequence_aug as aug1d
from Demo.utils.general_func import find_all_index_N
from Demo.Datasets.datasets_utils.DatasetsBase import dataset, signal_plot
from Demo.Datasets.get_file.Simulation_get_files import generate_simu_signals_all

seed = 999
np.random.seed(seed)
random.seed(seed)


class Simulation(object):
    num_classes = 3
    inputchannel = 1
    Fs = 1e4
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _preload(pre_file_dir, args, try_preload=True):
        if os.path.exists(pre_file_dir) and try_preload:
            with open(pre_file_dir, 'rb') as f:
                data_pd, storage_args, label_name = pickle.load(f)
                if storage_args == args:
                    print('load data from buffer')
                    return data_pd, label_name
        print('generate data')
        list_data, list_labels, label_name, infos = generate_simu_signals_all \
            (L=args['signal_size'], SNR=args['SNR'], N_sample=args['N_sample'])
        # infos: [N_sample,(f_c,f_f,A.phi),N_sub]
        data_pd = pd.DataFrame({"data": [item[np.newaxis, :] for item in list_data],
                                "label": list_labels, "infos": [item for item in infos]})
        data_pd['label'] = data_pd['label'].astype(int)
        if os.path.split(pre_file_dir)[0] and not os.path.exists(os.path.split(pre_file_dir)[0]):
            os.makedirs(os.path.split(pre_file_dir)[0])

        with open(pre_file_dir, 'wb') as f:
            pickle.dump((data_pd, args, label_name), f)
            print('save data to buffer')
        return data_pd, label_name

    def data_preprare(self, signal_size=2000, SNR=None, try_preload=True, plot=False):
        test_size = self.args['test_size'] if 'test_size' in self.args.keys() else 0.3
        args = {'signal_size': signal_size, 'SNR': SNR, 'N_sample': 5000}  # params for data generation
        data_pd, label_name = self._preload \
            (os.path.join(self.args['data_dir'], 'data_buffer.pkl'), args, try_preload=try_preload)

        if plot:
            index_plot = find_all_index_N(data_pd['label'].to_numpy(), 1)
            signal_plot(data=np.vstack(data_pd['data'][index_plot].values.tolist()),
                        label=[label_name[item] for item in data_pd['label'][index_plot].values.tolist()],
                        Fs=self.Fs, save_path=os.path.join(self.args['data_dir'], 'signal_plot'))

        index_list = list(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0).split(np.ones(data_pd.shape[0]), data_pd["label"].to_list()))
        train_pd, val_pd = data_pd.iloc[index_list[0][0]], data_pd.iloc[index_list[0][1]]

        transform = {
            'train': aug1d.Compose([aug1d.to_numpy(), aug1d.Data_type(self.args['data_type']),
                                    aug1d.Normalize(self.args['normlizetype']), aug1d.Retype()]),
            'val': aug1d.Compose([aug1d.to_numpy(), aug1d.Data_type(self.args['data_type']),
                                  aug1d.Normalize(self.args['normlizetype']), aug1d.Retype()]), }
        train_dataset = dataset(list_data=train_pd, transform=transform['train'])
        val_dataset = dataset(list_data=val_pd, transform=transform['val'])
        return (train_dataset, val_dataset), label_name



if __name__ == '__main__':
    args = {'data_type': 'time', 'normlizetype': 'mean-std', 'test_size': 0.3,
            'data_dir': os.path.join(projecht_dir,'Demo','Datasets','Buffer-SimulationDataset'), }
    simulation = Simulation(args)

    a, b = simulation.data_preprare(plot=True,SNR=0)
    print('train:', len(a[0]), 'val:', len(a[1]))

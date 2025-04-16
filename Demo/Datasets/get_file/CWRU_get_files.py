import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 3))

import random
import numpy as np
from scipy.io import loadmat

from Demo.Datasets.datasets_utils.generalfunciton import add_noise

seed = 999
np.random.seed(seed)
random.seed(seed)

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"] # 48K healthy data: 1797rpm,1772rpm,1750rpm,1730rpm
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat", "174.mat", "189.mat", "201.mat", "213.mat", "226.mat",
             "238.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "227.mat",
              "239.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "228.mat",
              "240.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "229.mat",
              "241.mat"]  # 1730rpm
dataname_12k_DE = [dataname1, dataname2, dataname3, dataname4]
dataname_48k_DE = [dataname9, dataname10, dataname11, dataname12]
# label
# label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
lab_name = ['N', 'I1', 'B1', 'O1', 'I2', 'B2', 'O2', 'I3', 'B3', 'O3']
indexs = [1, 2, 3]  # do not consider normal class, in [1,9] | [3,6,9] | list(range(1,10))
lab_name = ['H'] + [lab_name[i] for i in indexs]
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, signal_size=1024, SNR=None, load_condition=1):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    types: 'time'| 'fft', depends on the signal types, and 'time' is used in this article
    signal_size: length size of signal sample
    downsample_rate: to downsample signal in a specific rate, 1 is used in this article
    SNR: if SNR == None, no noise is added into dataset. else a specific noise is added to dataset
    load_condition: 0|1|2|3, choose the load condition of signals.
    '''
    signal_size = 2*signal_size if type == 'fft' else signal_size
    data_root1 = os.path.join(root, datasetname[3]) # For normal data
    data_root2 = os.path.join(root, datasetname[0]) # For 12k Drive End Bearing Fault Data

    path1 = os.path.join(data_root1, normalname[load_condition])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data, lab = data_load(path1, signal_size=signal_size,label=0,downsample_rate=4,SNR=SNR)  # Extract normal data, whose label is 0

    for i, item in enumerate(indexs): # Extract fault data, whose label is 1-9
        path2 = os.path.join(data_root2, dataname_12k_DE[load_condition][item - 1])
        data1, lab1 = data_load(path2, signal_size=signal_size, label=i + 1, downsample_rate=1, SNR=SNR)
        data += data1
        lab += lab1
    return [data, lab], lab_name


def data_load(filename,signal_size, label,downsample_rate=1,SNR=None):
    '''
    This function is mainly used to generate test data and training data.
    filename: Data location
    types: 'time'| 'fft' | 'slice' | 'CWT' | 'STFT', depends on the signal types, and 'time' is used in this article
    signal_size: length size of signal sample
    label: assigned to data label
    downsample_rate: to downsample signal in a specific rate, 1 is used in this article
    SNR: if SNR == None, no noise is added into dataset. else a specific noise is added to dataset
    '''
    downsample_rate = max(int(downsample_rate), 1)
    fl = loadmat(filename.replace('\\','/'))
    for i,item in enumerate(fl.keys()): # find axis[0] of mat file
        if axis[0] in item: # drive end
            fl = fl[item]
            break
        if i == len(fl)-1:
            raise ValueError("target item didn't found in mat file")

    fl = fl.squeeze()[::downsample_rate]
    if SNR is not None:
        fl = add_noise(fl,SNR)
    data = []
    lab = []
    start, end = 0, signal_size
    step = int(signal_size // 2)
    while end <= fl.shape[0]:
        x = fl[start:end]
        x = x.reshape([1] + list(x.shape)) #  get channel dimension
        data.append(x)
        lab.append(label)
        start += step
        end += step
    return data, lab

if __name__ == '__main__':
    root = r'$CWRU_dir$'
    (data,lab), name = get_files(root, signal_size=1024, SNR=None, load_condition=1)
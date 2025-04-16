import os
import numpy as np
from datetime import datetime

def ExtractInfo(filepath,append_acc = True):
    # try:
    start_time, record_time = None, None
    print(filepath)
    Dict = {'current_lr': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    Params = {'filename': os.path.split(os.path.split(filepath)[0])[-1]}
    with open(filepath, 'r') as f:
        temp = f.readline()
        flag_temp = True
        while '-----Epoch' not in temp:  # read params util find '-----Epoch'
            if ": " in temp[15:] and flag_temp:
                while ": " in temp:
                    key = temp.split(": ")[0].split(" ")[-1].strip(' ')
                    value = temp.split(": ")[1].split("|")[0].replace("\n", '').strip(' ')
                    Params[key] = value
                    temp = temp[len(temp.split("|")[0])+1:]
            if '---------------------------------' in temp:
                flag_temp = False
            temp = f.readline()
        while temp != '':  # enter the loop
            if '-----Epoch' in temp:  # beginning of a new epoch
                temp = f.readline().split('current lr: ')[1].split(':')[-1].strip('\n[]')
                Dict['current_lr'].append([float(item.strip(' ')) for item in temp.split(',')])
                temp = f.readline()
                if not start_time:
                    start_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
                while "<tempinfo>" in temp:  # skip the tempinfo
                    temp = f.readline()
                Dict['train_loss'].append([float(item.split(':')[1].strip(', \n')) for item in temp.replace('train Loss','train-Loss').split('train-Loss')[1:]])
                Dict['train_acc'].append(float(temp.split('train-Acc: ')[1].split(' ')[0].strip(', ')))
                temp = f.readline()
                Dict['val_loss'].append([float(item.split(',')[0].split(':')[1].strip(', \n')) for item in temp.replace('val Loss','val-Loss').split('val-Loss')[1:]])
                Dict['val_acc'].append(float(temp.split('val-Acc: ')[1].split(' ')[0].strip(', ')))
                end_time = datetime.strptime(temp[:14], '%m-%d %H:%M:%S')
            if '<training time>: ' in temp:  # obtain the training time
                record_time = float(temp.split('<training time>: ')[1].split(' ')[0].replace(',', ''))
            temp = f.readline()

    for key in Dict.keys():  # convert the list to numpy array
        if key != 'params':
            Dict[key] = np.array(Dict[key])
    Params['train_time'] = record_time if record_time else (end_time - start_time).total_seconds()
    if append_acc:
        Params['max acc'] = max(Dict['val_acc'])
        Params['final acc'] = Dict['val_acc'][-5:].mean() if len(Dict['val_acc'])>5 else Dict['val_acc'][-1]
    if 'model_name' in Params.keys() and 'Proto' in Params['model_name']:
        Params['model_name'] += '-'+Params['proto_M']
    if 'val_v' in Params.keys() and 'val_d' in Params.keys():
        Params['v_d'] = Params['val_v']+'-'+Params['val_d']
    # except:
    #     print(filepath)
    return Params,Dict


if __name__ == '__main__':
    a = ExtractInfo(r'XXXXXXXXXXXX', append_acc=True) # replace with your checkpoint file path, e.g., 'checkpoint/CNN-Simulation-time-SNRNone-0413-191146'
    print(1)

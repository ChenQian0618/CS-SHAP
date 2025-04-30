'''
2025/04/13
Created by ChenQian-SJTU (chenqian2020@sjtu.edu.cn)
This script is used to analyze the SHAP values of the models under different situations (e.g., model, transform function, etc.)
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
os.chdir(root_path_k(__file__, 0)) # change the current working directory to the root path
projecht_dir = root_path_k(__file__, 1)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)

import argparse
from SHAPs.MultiDomain_SHAP import MultiDomain_SHAP
from Demo.utils.DataModel_Loader import DataModel_Loader


def parse_args():
    parser = argparse.ArgumentParser(description='Demo_analysis')

    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint', help='the name of the data')
    parser.add_argument('--checkpoint_name', type=str, default='None', help='specify the checkpoint name,  e.g., "CNN-Simulation-time-SNRNone-0413-191146"')
    parser.add_argument('--domain_mode', type=str, default='all', choices=['all','time', 'frequency', 'envelope', 'STFT', 'CS'], help='the name of the data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ----------------------------parse args------------------------------------------------
    args = parse_args()
    checkpoint_root = args.checkpoint_root
    checkpoint_name = args.checkpoint_name
    domain_modes = args.domain_mode
    # process checkpoint_name
    if checkpoint_name == 'None': # if None, use the first one
        checkpoint_name = next(os.walk(checkpoint_root))[1][0]
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    print(f'checkpoint_name: "{checkpoint_dir.replace('\\','/'):s}"')
    # process domain_mode
    if domain_modes == 'all':
        domain_modes = ['time', 'frequency', 'envelope', 'STFT', 'CS']
    else:
        domain_modes = [domain_modes]

    # ----------------------------load model and data------------------------------------------------
    loader = DataModel_Loader(dir=checkpoint_dir, flag_preload_dataset=True)
    func_predict, background_data, _, input_data, input_label = loader.get_fuc_data()
    save_dir = loader.save_dir

    #  ----------------------------shap analysis------------------------------------------------
    MD_SHAP = MultiDomain_SHAP(func_predict, background_data, save_dir)
    for i,mode in enumerate(domain_modes):
        print('\n'*2, '-'*30, f'({i+1:d}/{len(domain_modes):d}) begin {mode:s}-SHAP analysis', '-'*30)
        MD_SHAP.explain(input_data, input_label, mode=mode, preload=True, Fs=loader.Fs)

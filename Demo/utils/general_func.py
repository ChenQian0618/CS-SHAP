from collections import Counter
import numpy as np

def find_all_index_N(label, Target_number=0, squeeze=True):
    '''
    :param label: (N,) int,ndarray
    :param Target_number: int
    :param squeeze: boll
    :return:
    '''
    if Target_number:
        Target_number = min(min(Counter(label).values()), Target_number)
    else:
        Target_number = min(Counter(label).values())
    Index = np.array([np.where(label == temp_l)[0][:Target_number] for temp_l in set(label)])
    if squeeze:
        return Index.reshape(-1)
    else:
        return Index  # (c,n)
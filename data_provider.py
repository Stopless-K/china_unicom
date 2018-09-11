import pandas as pd
import numpy as np
from collections import Counter
import os
from config import Config
config = Config()

def clean_n(item):

    if item == r'\N':return 0
    else: return item

def load(type_ = 'train'):
    raw = pd.read_csv(config.data_path+ type_+ r'\\'+type_+'.csv', sep=',', header=0, index_col=None)

    if config.drop_feature:
        dropped = raw.drop(config.drop_feature, axis=1)

    dropped = dropped.applymap(clean_n)
    if type_ == 'train':
        label = dropped[config.label_name].values
        x = dropped.drop(config.label_name, 1).values
    # label_counter = Counter(raw['current_service'].values.tolist())

    # print(label_counter.most_common())

    # check for \N values
    # tmp = x[:, 4]
    # for i, each in enumerate(tmp):
    #     if isinstance(each, str):
    #         print(each)
    #         print(i)
    #         print( each == r'\N')
    #         print( float(each))
        label = process_label(label)
        return x, label
    else:
        x = dropped.values
        user_id = raw['user_id']
        return x, user_id

def process_label(y):
    counter = Counter(y.tolist())

    keys_by_count = [kv[0] for kv in counter.most_common()]

    idx = np.arange(len(keys_by_count))

    service_2_index = dict(zip(keys_by_count, idx))
    if not os.path.exists(config.out_path+ config.idx_2_service):
        os.mkdir(config.out_path)
        index_2_service = dict(zip(idx, keys_by_count))
        pd.to_pickle(index_2_service, config.out_path+ config.idx_2_service)
    label = np.array([service_2_index[k] for k in y])

    return label


if __name__ == '__main__':
    load('train')
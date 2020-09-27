import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import load_npz

from dataset import MultiDataset, MultiTestDataset
from dataset import MultiClassDataset, MultiClassTestDataset
from dataset import Regression_MultiDataset, Regression_MultiTestDataset


def scatter_path_array(path_data, size, rank):
    all_lst = []

    for row, item in path_data.iterrows():
        path, num = item['path'], int(item['res_num'])
        all_lst.extend([[path, i, row] for i in range(num)])
    all_lst = np.array(all_lst, dtype=object)
    all_lst = np.random.permutation(all_lst)
    all_lst = all_lst[int(len(all_lst) / size) * rank:int(len(all_lst) / size) * (rank + 1):]
    return all_lst[:, 0], all_lst[:, 1], all_lst[:, 2]


class Dataproc():
    def __init__(self, size, rank, config):
        self.config = config
        label_name = self.config['label'][0]
        voxel_path = Path(self.config['voxel_path'])
        label_path = Path(self.config['label_path'])

        # size = 32
        train_data_df = pd.read_csv(self.config['train_data'], index_col=0)
        path, resid, protein_id = scatter_path_array(train_data_df, size, rank)
        train_df = pd.DataFrame({'path': path, 'i': resid}, index=protein_id)
        train_groupby = train_df.groupby('path')


        def load_train(path, group):
            voxel = load_npz(voxel_path / path)
            label = np.load(label_path / path)[label_name]
            return [[voxel[index], float(label[index])] for index in group['i']]


        sub = Parallel(n_jobs=14)([delayed(load_train)(path, group) for path, group in train_groupby])
        train_voxel = [y[0] for x in sub for y in x]
        train_label = [y[1] for x in sub for y in x]

        del train_df, train_groupby, sub
        gc.collect()
        
        test_data_df = pd.read_csv(self.config['test_data'], index_col=0)
        path, resid, protein_id = scatter_path_array(test_data_df, size, rank)
        test_df = pd.DataFrame({'path': path, 'i': resid}, index=protein_id)
        test_groupby = test_df.groupby('path')

        def load_test(path, group):
            voxel = load_npz(voxel_path/path)
            label = np.load(label_path/path)[label_name]
            protein_id = group.index[0]
            return [[voxel[index], float(label[index]), protein_id] for index in group['i']]
        
        sub = Parallel(n_jobs=14)([delayed(load_test)(path, group) for path, group in test_groupby])
        test_voxel = [y[0] for x in sub for y in x]
        test_label = [y[1] for x in sub for y in x]
        test_protein_id = [y[2] for x in sub for y in x]

        del test_df, test_groupby, sub
        gc.collect()

        train_voxel = np.random.RandomState(7).permutation(train_voxel)
        train_label = np.random.RandomState(7).permutation(train_label)

        self.train_voxel = train_voxel
        self.train_label = train_label
        self.test_voxel = test_voxel
        self.test_label = test_label
        self.test_protein_id = test_protein_id


    def get_test_path_data(self):
        test_path_df = pd.read_csv(self.config['test_data'], index_col=0)
        return test_path_df


    def get_classification_dataset(self, key):
        self.train_label = (self.train_label > self.config['local_threshold']).astype(np.int).reshape(-1, 1)
        self.test_label = (np.array(self.test_label) > self.config['local_threshold']).astype(np.int).reshape(-1, 1)
        if key == 'train':
            dataset = MultiDataset(self.train_voxel, self.train_label, config=self.config)
        elif key == 'test':
            dataset = MultiTestDataset(self.test_voxel, self.test_label, self.test_protein_id, config=self.config)
        return dataset


    def get_multiclassification_dataset(self, key):
        if key == 'train':
            dataset = MultiClassDataset(self.train_voxel, self.train_label, config=self.config)
        elif key == 'test':
            dataset = MultiClassTestDataset(self.test_voxel, self.test_label, self.test_protein_id, config=self.config)
        return dataset


    def get_regression_dataset(self, key):
        if key == 'train':
            dataset = Regression_MultiDataset(self.train_voxel, self.train_label, config=self.config)
        elif key == 'test':
            dataset = Regression_MultiTestDataset(self.test_voxel, self.test_label, self.test_protein_id, config=self.config)
        return dataset


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


if __name__ == '__main__':
    f = open('./data/config_multi_label.json', 'r')
    config = json.load(f, object_hook=hinted_tuple_hook)[0]
    d = Dataproc(32, 0, config)
    train = d.get_classification_dataset('test')
    #train = d.get_regression_dataset('test')
    example = train.get_example(1)
    print(example)
    print(example[0].shape)
    if len(example) > 4:
        print(example[:-3])
        print(example[:-3][0].shape)
        print(example[:-3][1].shape)
        print(example[-3])
        print(example[-3].shape)
        print(example[-2])
        print(example[-2].shape)
        print(example[-1])
        print(example[-1].shape)

import os

import numpy as np
import six
from chainer.dataset import DatasetMixin
from numba import jit
from scipy.sparse import load_npz


class Regression_MultiDataset(DatasetMixin):
    def __init__(self, voxel, label, config):
        super(Regression_MultiDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        #voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        data_width = voxel.shape[1]
        b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        voxel = voxel[:, b:e, b:e, b:e]
        local_label = self.label[i]
        return voxel, local_label

    def get_example(self, i):
        voxel, label = self.get_data(i)
        return voxel, label


class Regression_MultiTestDataset(DatasetMixin):
    def __init__(self, voxel, label, protein_id, config):
        super(Regression_MultiTestDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.protein_id = protein_id
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        #voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        data_width = voxel.shape[1]
        b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        voxel = voxel[:, b:e, b:e, b:e]
        label = self.label[i]
        protein_id = self.protein_id[i]
        return voxel, local_label, protein_id

    def get_example(self, i):
        voxel, label, protein_id = self.get_data(i)
        protein_id = np.array([protein_id])
        return voxel, label, protein_id


class MultiDataset(DatasetMixin):
    def __init__(self, voxel, label, config):
        super(MultiDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        # voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        # voxel = voxel[:38]
        #voxel = voxel[:38, 2:30, 2:30, 2:30]
        data_width = voxel.shape[1]
        b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        voxel = voxel[:, b:e, b:e, b:e]
        local_label = self.label[i]
        return voxel, local_label

    def get_example(self, i):
        voxel, label = self.get_data(i)
        return voxel, label


class MultiTestDataset(DatasetMixin):
    def __init__(self, voxel, label, protein_id, config):
        super(MultiTestDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.protein_id = protein_id
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        # voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        # voxel = voxel[:38]
        #voxel = voxel[:38, 2:30, 2:30, 2:30]
        data_width = voxel.shape[1]
        b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        voxel = voxel[:, b:e, b:e, b:e]
        local_label = self.label[i]
        protein_id = self.protein_id[i]
        return voxel, local_label, protein_id

    def get_example(self, i):
        voxel, label, protein_id = self.get_data(i)
        protein_id = np.array([protein_id])
        return voxel, label, protein_id


class MultiClassDataset(DatasetMixin):
    def __init__(self, voxel, label, config):
        super(MultiClassDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.config = config
        self.class_range = np.linspace(0, 1, self.config['class_num']+1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        # voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        #data_width = voxel.shape[1]
        #b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        #voxel = voxel[:, b:e, b:e, b:e]
        # voxel = voxel[:38]
        local_label = self.label[i]
        return voxel, local_label

    def get_example(self, i):
        voxel, label = self.get_data(i)
        label = max(0, np.where(label <= self.class_range)[0][0] - 1)
        label = np.array(label, dtype=np.int)
        return voxel, label


class MultiClassTestDataset(DatasetMixin):
    def __init__(self, voxel, label, protein_id, config):
        super(MultiClassTestDataset, self).__init__()
        self.voxel = voxel
        self.label = label
        self.protein_id = protein_id
        self.config = config
        self.class_range = np.linspace(0, 1, self.config['class_num']+1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        return len(self.voxel)

    @jit
    def get_data(self, i):
        voxel = self.voxel[i]
        # voxel = np.reshape(voxel.toarray(), (-1, self.config['box_width'], self.config['box_width'], self.config['box_width'])).astype(np.float32)
        voxel = np.reshape(voxel.toarray(), (-1, 32, 32, 32)).astype(np.float32)
        data_width = voxel.shape[1]
        b, e = (data_width - self.config['box_width']) // 2, (data_width + self.config['box_width']) // 2
        voxel = voxel[:, b:e, b:e, b:e]
        # voxel = voxel[:38]
        local_label = self.label[i]
        protein_id = self.protein_id[i]
        return voxel, local_label, protein_id

    def get_example(self, i):
        voxel, label, protein_id = self.get_data(i)
        label = max(0, np.where(label <= self.class_range)[0][0] - 1)
        label = np.array(label, dtype=np.int)
        protein_id = np.array(protein_id, dtype=np.int)
        return voxel, label, protein_id

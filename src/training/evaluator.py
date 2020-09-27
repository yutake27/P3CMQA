import copy
import os
from functools import reduce
from pathlib import Path

import chainer
import chainer.functions as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
from chainer import configuration, cuda, function
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.training.extensions import Evaluator
from chainermn import CommunicatorBase
from sklearn import metrics
from tqdm import tqdm


def _to_list(a):
    """convert value `a` to list
    Args:
        a: value to be convert to `list`
    Returns (list):
    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


def plot_roc(y_true, y_score, out_name):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
    auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % auc)
    plt.legend()
    plt.title('ROC curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(True)
    plt.savefig(out_name)


class Classification_Evaluator(Evaluator):
    """Evaluator which calculates auc and correlation
    Note that this Evaluator is only applicable to binary classification task.
    Args:
        iterator: Dataset iterator for the dataset to calculate pearson.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, comm, label_name, converter=convert.concat_examples,
                    device=None, eval_hook=None, eval_func=None, name=None,
                    pos_labels=1, ignore_labels=None, path_data=None):
        super(Classification_Evaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.rank = comm.rank
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm
        self.label_name = label_name
        self.path_data = path_data

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name, target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_roc_corr(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_roc_corr(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = np.array([]).reshape([0, len(self.label_name)])
        t_total = np.array([]).reshape([0, len(self.label_name)])
        protein_id_total = np.array([]).reshape([0, len(self.label_name)])
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-2])
                t = in_arrays[-2]
                protein_id = in_arrays[-1]
            # y = F.sigmoid(y)
            y_data = cuda.to_cpu(y.data)
            t_data = cuda.to_cpu(t)
            protein_id = cuda.to_cpu(protein_id)
            y_total = np.vstack([y_total, y_data])
            t_total = np.vstack([t_total, t_data])
            protein_id_total = np.vstack([protein_id_total, protein_id])
        
        updater = trainer.updater
        epoch = str(updater.epoch)
        out_dir = Path(trainer.out)

        observation = {}
        for label_index, label in enumerate(self.label_name):
            y = y_total[:, label_index]
            t = t_total[:, label_index]
            protein_id = protein_id_total[:, label_index]
            index = np.where(t != -1)[0]
            y = y[index]
            t = t[index]
            protein_id = protein_id[index]
            gather_data = self.comm.gather(np.vstack([t, y, protein_id]))
            if self.rank == 0:
                gather_data = np.concatenate(gather_data, axis=1)
                gather_t = np.array(gather_data[0], dtype=np.int)
                gather_y = np.array(gather_data[1], dtype=np.float32)
                gather_protein_id = np.array(gather_data[2], dtype=np.int)

                global_score = []
                global_label = []
                target_name = []
                model_path = []
                for row, item in self.path_data.iterrows():
                    model_index = np.where(gather_protein_id==row)[0]
                    if len(model_index) > 0:
                        global_score.append(np.mean(F.sigmoid(gather_y[model_index]).data))
                        global_label.append(item['gdtts'])
                        target_name.append(item['dir_name'])
                        model_path.append(item['path'])
                df = pd.DataFrame({'global_score':global_score, 'global_label':global_label, 'target_name':target_name, 'model_path': model_path})
                pearson = df.groupby('target_name').corr(method='pearson')['global_score'].mean(level=1)['global_label']
                spearman = df.groupby('target_name').corr(method='spearman')['global_score'].mean(level=1)['global_label']
                csv_out_name = out_dir/(epoch+label+'_df.csv')
                df.to_csv(csv_out_name)

                roc_out_name = out_dir/(epoch+'iteration_'+label+'_roc.png')
                y_score = F.sigmoid(gather_y).data
                plot_roc(y_true=gather_t, y_score=y_score, out_name=roc_out_name)
                roc_auc = metrics.roc_auc_score(gather_t, y_score)
                with reporter.report_scope(observation):
                    reporter.report({'roc_auc_'+label: roc_auc}, self._targets['main'])
                    reporter.report({'loss': F.sigmoid_cross_entropy(gather_y, gather_t).data},
                                    self._targets['main'])
                    reporter.report({'accuracy': F.binary_accuracy(gather_y, gather_t).data}, self._targets['main'])
                    reporter.report({'pearson': pearson}, self._targets['main'])
                    reporter.report({'spearman': spearman}, self._targets['main'])
        return observation


class MultiClassification_Evaluator(Evaluator):
    """Evaluator which calculates auc and correlation
    Note that this Evaluator is only applicable to binary classification task.
    Args:
        iterator: Dataset iterator for the dataset to calculate pearson.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, comm, label_name, class_num,
                    converter=convert.concat_examples,
                    device=None, eval_hook=None, eval_func=None, name=None,
                    pos_labels=1, ignore_labels=None, path_data=None):
        super(MultiClassification_Evaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.rank = comm.rank
        self.class_num = class_num
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm
        self.label_name = label_name
        self.path_data = path_data

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name, target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_corr(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_corr(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = np.array([]).reshape([0, self.class_num])
        t_total = np.array([], dtype=np.int)
        protein_id_total = np.array([], dtype=np.int)
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-2])
                t = in_arrays[-2]
                protein_id = in_arrays[-1]
            # y = F.sigmoid(y)
            y_data = cuda.to_cpu(y.data)
            t_data = cuda.to_cpu(t)
            protein_id = cuda.to_cpu(protein_id)
            y_total = np.vstack([y_total, y_data])
            t_total = np.concatenate([t_total, t_data])
            protein_id_total = np.concatenate([protein_id_total, protein_id])
        
        updater = trainer.updater
        epoch = str(updater.epoch)
        out_dir = Path(trainer.out)

        observation = {}
        gather_data = self.comm.gather(np.hstack([t_total.reshape(-1,1), y_total, protein_id_total.reshape(-1,1)]))
        if self.rank == 0:
            gather_data = np.concatenate(gather_data)
            gather_t = gather_data[:, 0].astype(np.int)
            gather_y = gather_data[:, 1:-1].astype(np.float32)
            gather_protein_id = gather_data[:, -1].astype(np.int)

            global_score = []
            global_label = []
            target_name = []
            model_path = []
            for row, item in self.path_data.iterrows():
                model_index = np.where(gather_protein_id==row)[0]
                if len(model_index) > 0:
                    local_score = np.argmax(gather_y[model_index], axis=1)/self.class_num
                    global_score.append(np.mean(local_score))
                    global_label.append(item['gdtts'])
                    target_name.append(item['dir_name'])
                    model_path.append(item['path'])
            df = pd.DataFrame({'global_score':global_score, 'global_label':global_label, 'target_name':target_name, 'model_path': model_path})
            pearson = df.groupby('target_name').corr(method='pearson')['global_score'].mean(level=1)['global_label']
            spearman = df.groupby('target_name').corr(method='spearman')['global_score'].mean(level=1)['global_label']
            csv_out_name = out_dir/(epoch+'_df.csv')
            df.to_csv(csv_out_name)
            with reporter.report_scope(observation):
                reporter.report({'loss': F.softmax_cross_entropy(gather_y, gather_t).data},
                                    self._targets['main'])
                reporter.report({'accuracy': F.accuracy(gather_y, gather_t).data}, self._targets['main'])
                reporter.report({'pearson': pearson}, self._targets['main'])
                reporter.report({'spearman': spearman}, self._targets['main'])
        return observation



class Regression_Evaluator(Evaluator):
    """Evaluator which calculates correlation
        Args:
        iterator: Dataset iterator for the dataset to calculate pearson.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, comm, label_name, converter=convert.concat_examples,
                    device=None, eval_hook=None, eval_func=None, name=None,
                    pos_labels=1, ignore_labels=None, path_data=None):
        super(Regression_Evaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.rank = comm.rank
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm
        self.label_name = label_name
        self.path_data = path_data

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name, target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_corr(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_corr(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = np.array([]).reshape([0, len(self.label_name)])
        t_total = np.array([]).reshape([0, len(self.label_name)])
        protein_id_total = np.array([]).reshape([0, len(self.label_name)])
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-2])
                t = in_arrays[-2]
                protein_id = in_arrays[-1]
            # y = F.sigmoid(y)
            y_data = cuda.to_cpu(y.data)
            t_data = cuda.to_cpu(t)
            protein_id = cuda.to_cpu(protein_id)
            y_total = np.vstack([y_total, y_data])
            t_total = np.vstack([t_total, t_data])
            protein_id_total = np.vstack([protein_id_total, protein_id])
        
        updater = trainer.updater
        epoch = str(updater.epoch)
        out_dir = Path(trainer.out)

        observation = {}
        for label_index, label in enumerate(self.label_name):
            y = y_total[:, label_index]
            t = t_total[:, label_index]
            protein_id = protein_id_total[:, label_index]
            index = np.where(t != -1)[0]
            y = y[index]
            t = t[index]
            protein_id = protein_id[index]
            gather_data = self.comm.gather(np.vstack([t, y, protein_id]))
            if self.rank == 0:
                gather_data = np.concatenate(gather_data, axis=1)
                gather_t = np.array(gather_data[0], dtype=np.float32)
                gather_y = np.array(gather_data[1], dtype=np.float32)
                gather_protein_id = np.array(gather_data[2], dtype=np.int)

                global_score = []
                global_label = []
                target_name = []
                model_path = []
                for row, item in self.path_data.iterrows():
                    model_index = np.where(gather_protein_id==row)[0]
                    if len(model_index) > 0:
                        global_score.append(np.mean(gather_y[model_index]))
                        global_label.append(item['gdtts'])
                        target_name.append(item['dir_name'])
                        model_path.append(item['path'])
                df = pd.DataFrame({'global_score':global_score, 'global_label':global_label, 'target_name':target_name, 'model_path': model_path})
                pearson = df.groupby('target_name').corr(method='pearson')['global_score'].mean(level=1)['global_label']
                spearman = df.groupby('target_name').corr(method='spearman')['global_score'].mean(level=1)['global_label']
                csv_out_name = out_dir/(epoch+label+'_df.csv')
                df.to_csv(csv_out_name)

                with reporter.report_scope(observation):
                    reporter.report({'loss': F.mean_squared_error(gather_y, gather_t).data},
                                    self._targets['main'])
                    reporter.report({'accuracy': F.r2_score(gather_y, gather_t).data}, self._targets['main'])
                    reporter.report({'pearson': pearson}, self._targets['main'])
                    reporter.report({'spearman': spearman}, self._targets['main'])
        return observation

from __future__ import print_function

import argparse
import copy
import json
import os
import re
import sys
from functools import partial

import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.links as L
import matplotlib
import numpy as np
from chainer import training
from chainer.sequential import Sequential
from chainer.training import extensions
from chainermn import links as MNL

from dataproc import Dataproc
from evaluator import Classification_Evaluator, MultiClassification_Evaluator
# from resnet import ResNet18
# from resnet_before import ResNet18_half

matplotlib.use('Agg')



def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


def schedule_optimizer_value(epoch_list, value_list, optimizer_name='main', attr_name='lr'):
    """Set optimizer's hyperparameter according to value_list, scheduled on epoch_list.

    Example usage:
    trainer.extend(schedule_optimizer_value([2, 4, 7], [0.008, 0.006, 0.002]))
    """
    if isinstance(epoch_list, list):
        assert len(epoch_list) == len(value_list)
    else:
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        assert isinstance(value_list, float) or isinstance(value_list, int)
        epoch_list = [epoch_list, ]
        value_list = [value_list, ]

    trigger = chainer.training.triggers.ManualScheduleTrigger(epoch_list, 'epoch')
    count = 0

    @chainer.training.extension.make_extension(trigger=trigger)
    def set_value(trainer: chainer.training.Trainer):
        nonlocal count
        optimizer = trainer.updater.get_optimizer(optimizer_name)
        setattr(optimizer, attr_name, value_list[count])
        count += 1

    return set_value


def get_model(config, comm, predict=False):
    model = Sequential()
    W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
    bias = chainer.initializers.Zero(dtype=np.float32)
    layers = config['model']
    for layer in layers:
        name = layer['name']
        parameter = copy.deepcopy(layer['parameter'])
        if name[0] == 'L':
            if 'Conv' in name or 'Linear' in name:
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name[0] == 'O':
            if predict == False:
                parameter.update({'initialW': W, 'initial_bias': bias, 'bn_kwargs': {'comm': comm}})
            else:
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name[0] == 'F':
            if len(parameter) == 0:
                add_layer = partial(eval(name))
            else:
                add_layer = partial(eval(name), **parameter)
        elif name[0] == 'M':
            if predict:
                add_layer = L.BatchNormalization(size=parameter['size'])
            else:
                add_layer = MNL.MultiNodeBatchNormalization(size=parameter['size'], comm=comm)
        model.append(add_layer)
    return model


def main():
    import chainermn
    chainer.global_config.autotune = True

    parser = argparse.ArgumentParser(description='ChainerMN example: Train MQAP using 3DCNN')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch', '-b', type=int, default=32,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume the training from snapshot')
    parser.add_argument('--config', '-c', type=int, default=0,
                        help='Number of config')
    parser.add_argument('--regression', '-reg', action='store_true',
                        help='Regression')
    parser.add_argument('--opt', type=str,
                        default='adam', help='Optimizer',
                        choices=['adam', 'adabound', 'sgd', 'smorms3'])
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate(alpha)')
    parser.add_argument('--cos', action='store_true',
                        help='Cosine annealing')
    parser.add_argument('--weight', action='store_true',
                        help='Weight Decay')
    parser.add_argument('--only_weight', '-w', action='store_true',
                        help='Resume only weight')
    parser.add_argument('--multi_class', '-m', action='store_true',
                        help='multi class classification')
    args = parser.parse_args()
    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator, allreduce_grad_dtype='float16')
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    f = open('./data/config_multi_label.json', 'r')
    config = json.load(f, object_hook=hinted_tuple_hook)[args.config]

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num epoch: {}'.format(args.epoch))
        print('Config Num:  {}'.format(args.config))
        print('Train_rate  {}'.format(config['train_rate']))
        print('Data frac  {}'.format(config['data_frac']))
        print('Box_width:  {}'.format(config['box_width']))
        print('Channel:  {}'.format(config['channel']))
        print('Label:  {}'.format(config['label']))
        print('Local_threshold:  {}'.format(config['local_threshold']))
        print('Batch size:  {}'.format(args.batch))
        print('Optimizer:  {}'.format(args.opt))
        print('Learning Rate:  {}'.format(args.lr))
        print('Out Directory:  {}'.format(args.out))
        if args.regression:
            print('Regression')
        if args.cos:
            print('Cosine Annealing')
        if args.weight:
            print('Weight Decay')
        if args.multi_class:
            print('Multi Class classification: {}'.format(config['class_num']))
        print('==========================================')
    if comm.rank == 0:
        print('data load start!!')
    d = Dataproc(size=comm.size, rank=comm.rank, config=config)
    if device >= 0:
        chainer.cuda.get_device(device).use()
    sub_comm = comm.split(comm.rank // comm.intra_size, comm.rank)
    model = get_model(config=config, comm=sub_comm)
    # if args.multi_class:
    #     model = ResNet18(config['class_num'], bn_kwargs={'comm': sub_comm})
    # else:
    #     model = ResNet18_half(1, bn_kwargs={'comm': sub_comm})
    if comm.rank == 0:
        print('data load end!!')
        print(model)
    if args.regression:
        model = F.sigmoid(model)
        model = L.Classifier(model, lossfun=F.mean_squared_error, accfun=F.r2_score)
        train, test = d.get_regression_dataset(key='train'), d.get_regression_dataset(key='test')
    elif args.multi_class:
        model = L.Classifier(model ,lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
        train, test = d.get_multiclassification_dataset(key='train'), d.get_multiclassification_dataset(key='test')
    else:
        model = L.Classifier(model, lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)
        train, test = d.get_classification_dataset(key='train'), d.get_classification_dataset(key='test')

    #train_iter = I.MultiprocessIterator(dataset=train, batch_size=args.batch, repeat=True, shuffle=True, n_processes=5)
    train_iter = I.MultithreadIterator(dataset=train, batch_size=args.batch, repeat=True, shuffle=True, n_threads=14)
    #train_iter = I.SerialIterator(dataset=train, batch_size=args.batch, repeat=True, shuffle=True)
    #test_iter = I.MultiprocessIterator(dataset=test, batch_size=args.batch, repeat=False, shuffle=False, n_processes=14)
    test_iter = I.MultithreadIterator(dataset=test, batch_size=args.batch, repeat=False, shuffle=False, n_threads=14)
    #test_iter = I.SerialIterator(dataset=test, batch_size=args.batch, repeat=False, shuffle=False)
    if device >= 0:
        model.to_gpu()
    if args.opt == 'adam':
        opt = chainer.optimizers.Adam(amsgrad=True, alpha=args.lr)
        optimizer = chainermn.create_multi_node_optimizer(opt, comm, double_buffering=False)
        optimizer.setup(model)
    elif args.opt == 'adabound':
        opt = chainer.optimizers.Adam(adabound=True, alpha=args.lr)
        optimizer = chainermn.create_multi_node_optimizer(opt, comm, double_buffering=False)
        optimizer.setup(model)
    elif args.opt == 'sgd':
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.MomentumSGD(lr=args.lr), comm, double_buffering=False)
        optimizer.setup(model)
    elif args.opt == 'smorms3':
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.SMORMS3(lr=args.lr), comm, double_buffering=False)
        optimizer.setup(model)

    val_interval = 1, 'epoch'
    log_interval = 0.5, 'epoch'
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    if args.regression:
        evaluator = Regression_Evaluator(iterator=test_iter, target=model, device=device,
                                         eval_func=model.predictor, name='val', comm=comm,  label_name=config['label'], path_data=d.get_test_path_data())
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
        trainer.extend(evaluator, trigger=val_interval)
    elif args.multi_class:
        evaluator = MultiClassification_Evaluator(iterator=test_iter, target=model, device=device, eval_func=model.predictor, name='val', comm=comm,  label_name=config['label'], path_data=d.get_test_path_data(), class_num=config['class_num'])
        trainer.extend(evaluator, trigger=val_interval)
    else:
        evaluator = Classification_Evaluator(iterator=test_iter, target=model, device=device,
                                         eval_func=model.predictor, name='val', comm=comm,  label_name=config['label'], path_data=d.get_test_path_data())
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
        trainer.extend(evaluator, trigger=val_interval)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))

        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(), trigger=val_interval)
        # Be careful to pass the interval directly to LogReport
        # (it determines when to emit log rather than when to read observations)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        if args.regression:
            trainer.extend(extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss', 'val/main/loss', 'main/accuracy', 'val/main/accuracy', 'val/main/pearson', 'val/main/spearman',
                 'elapsed_time']), trigger=log_interval)
            trainer.extend(
                extensions.PlotReport(['main/loss', 'val/main/loss'],
                                      'epoch', file_name='loss.png'), trigger=val_interval)
            trainer.extend(
                extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'),
                trigger=val_interval)
        elif args.multi_class:
            trainer.extend(extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss', 'val/main/loss', 'main/accuracy', 'val/main/accuracy',
                 'val/main/pearson', 'val/main/spearman', 'elapsed_time']), trigger=log_interval)
            trainer.extend(
                extensions.PlotReport(['main/loss', 'val/main/loss'],
                                      'epoch', file_name='loss.png'), trigger=val_interval)
            trainer.extend(
                extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'),
                trigger=val_interval)
        else:
            trainer.extend(extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss', 'val/main/loss', 'main/accuracy', 'val/main/accuracy',
                 'val/main/roc_auc_local_lddt', 'val/main/pearson', 'val/main/spearman', 'elapsed_time']), trigger=log_interval)
            trainer.extend(
                extensions.PlotReport(['main/loss', 'val/main/loss'],
                                      'epoch', file_name='loss.png'), trigger=val_interval)
            trainer.extend(
                extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'),
                trigger=val_interval)

            trainer.extend(
                extensions.PlotReport(
                    ['val/main/local_lddt'],
                    'epoch', file_name='roc_auc.png'), trigger=val_interval)

        trainer.extend(extensions.ProgressBar(update_interval=10))
    # trainer.extend(extensions.ExponentialShift('alpha', 0.1),
    #                trigger=chainer.training.triggers.ManualScheduleTrigger([4000, 6000, 10000], 'iteration'))
    if args.resume:
        snap_list = [p for p in os.listdir(args.out) if 'snapshot' in p]
        snap_num = np.array([int(re.findall("[+-]?[0-9]+[\.]?[0-9]*[eE]?[+-]?[0-9]*", p)[0]) for p in snap_list])
        path = snap_list[np.argmax(snap_num)]
        path = os.path.join(args.out, path)
        if args.only_weight:
            obj_path = 'updater/model:main/predictor/'
            chainer.serializers.load_npz(path, model.predictor, obj_path)
        else:
            chainer.serializers.load_npz(path, trainer)
    if comm.rank == 0:
        #protein_name_dict = d.get_protein_name_dict()
        from pathlib import Path
        out_path = Path(args.out)
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        #np.savez(os.path.join(args.out, 'protein_name'), **protein_name_dict)
        f = open(os.path.join(args.out, 'config.json'), 'w')
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
        f.close()
        f = open(os.path.join(args.out, 'args.json'), 'w')
        json.dump(vars(args), f)
        f.close()
    if comm.rank == 0:
        print('train start!!!')
    trainer.run()


def global_except_hook(exctype, value, traceback):
    import sys
    from traceback import print_exception
    print_exception(exctype, value, traceback)
    sys.stderr.flush()

    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)


if __name__ == '__main__':
    sys.excepthook = global_except_hook
    import multiprocessing as mp

    tmp = os.environ['TMPDIR']
    os.environ.update({'CUPY_CACHE_DIR': os.path.join(tmp, '.cupy/kernel_cache')})
    mp.set_start_method('forkserver', force=True)
    main()

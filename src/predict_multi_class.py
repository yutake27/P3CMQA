import argparse
import copy
import gc
import glob
import json
import os
from functools import partial
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pandas as pd
from chainer import Variable
from chainer.function import no_backprop_mode
from chainer.sequential import Sequential
from chainer.serializers import load_npz
from make_voxel import get_voxel_predict
from training.resnet import ResNet18


def get_predict_value(data, model, gpu, class_num):
    data = data.astype(np.float32)
    batch_size = 16
    i = 0
    out_rep_list = []
    out_exp_list = []
    while i * batch_size < data.shape[0]:
        voxel = data[i * batch_size:(i + 1) * batch_size]
        voxel = Variable(voxel)
        if gpu >= 0:
            voxel.to_gpu()
        with no_backprop_mode(), chainer.using_config('train', False):
            pred_score = model(voxel)
        pred_score = chainer.cuda.to_cpu(pred_score.data)
        rep_score = np.argmax(pred_score, axis=1) / class_num
        pred_score = F.softmax(pred_score).data
        exp_score = np.sum(pred_score * np.arange(class_num) / class_num, axis=1)
        out_rep_list.extend(rep_score)
        out_exp_list.extend(exp_score)
        i += 1
    return np.array(out_rep_list), np.array(out_exp_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict ')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input_path', '-i', help='Input data path')
    parser.add_argument('--input_dir_path','-d', help='Input dir data path')
    parser.add_argument('--fasta_path', '-f', help='Reference FASTA Sequence path')
    parser.add_argument('--model_path', '-m', help='Pre-trained model path')
    parser.add_argument('--preprocess_dir', '-p', help='Input preprocess directory')
    parser.add_argument('--output_dir', '-o', help='Output directory')
    args = parser.parse_args()

    # 4class
    model4 = ResNet18(4)
    load_npz(file='../trained_model_4class.npz', obj=model4, path='updater/model:main/predictor/')
    model10 = ResNet18(10)
    load_npz(file='../trained_model_10class.npz', obj=model10, path='updater/model:main/predictor/')
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model4.to_gpu()  # Copy the model to the GPU
        model10.to_gpu()

    target = Path(args.fasta_path).stem
    # preprocess directory
    if not args.preprocess_dir:
        args.preprocess_dir = Path('../data/profile')/target
    preprocess_dir = Path(args.preprocess_dir)
    pssm_path = (preprocess_dir/target).with_suffix('.pssm')
    ss_path = (preprocess_dir/target).with_suffix('.ss')
    acc20_path = (preprocess_dir/target).with_suffix('.acc20')
    # output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('../data/score')/target
    output_dir.mkdir(exist_ok=True, parents=True)


    if args.input_path:
        data, _, _ = get_voxel_predict(input_path=str(input_path), fasta_path=args.fasta_path , pssm_path=pssm_path , ss_path = ss_path, acc20_path = acc20_path, buffer=32, width=1)
        predict_value4 = get_predict_value(data=data, model=model4, gpu=args.gpu, class_num=4)
        predict_value10 = get_predict_value(data=data, model=model10, gpu=args.gpu, class_num=10)
        print('Input Data Path : {}'.format(args.input_path))
        print(' 4 class representative value : {:.5f}'.format(np.mean(predict_value4[:, 0])))
        print(' 4 class expected value : {:.5f}'.format(np.mean(predict_value4[:, 1])))
        print('10 class representative value : {:.5f}'.format(np.mean(predict_value10[:, 0])))
        print('10 class expected value : {:.5f}'.format(np.mean(predict_value10[:, 1])))

    elif args.input_dir_path:
        input_dir_path = Path(args.input_dir_path)
        model_array = []
        global_score_array = []
        for input_path in input_dir_path.iterdir():
            model_array.append(input_path.stem)
            try:
                data, _, _ = get_voxel_predict(input_path=str(input_path), fasta_path=args.fasta_path , pssm_path=pssm_path , ss_path = ss_path, acc20_path = acc20_path, buffer=32, width=1)
            except Exception as e:
                global_score_array.append([None, None, None, None])
                print('make_voxel error:'+str(input_path), e)
            else:
                predict_value4R_array, predict_value4E_array = get_predict_value(data=data, model=model4, gpu=args.gpu, class_num=4)
                predict_value10R_array, predict_value10E_array = get_predict_value(data=data, model=model10, gpu=args.gpu, class_num=10)
                predict_value4R = np.mean(predict_value4R_array)
                predict_value4E = np.mean(predict_value4E_array)
                predict_value10R = np.mean(predict_value10R_array)
                predict_value10E = np.mean(predict_value10E_array)
                global_score_array.append([predict_value4R, predict_value4E, predict_value10R, predict_value10E])
                print(input_path.stem, predict_value4R, predict_value4E, predict_value10R, predict_value10E)
                del data
                gc.collect()
        global_score_array = np.array(global_score_array)
        df = pd.DataFrame({'4R': global_score_array[:, 0], '4E': global_score_array[:, 1],
                            '10R': global_score_array[:, 2], '10E': global_score_array[:, 3]}, index=model_array)
        output_path = (output_dir / target).with_suffix('.csv')
        df.to_csv(output_path)


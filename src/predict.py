import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import json
import argparse
import copy
from chainer.function import no_backprop_mode
from chainer.sequential import Sequential
from chainer.serializers import load_npz
from chainer import Variable
from functools import partial
from make_voxel import get_voxel_predict
import pandas as pd
import os, gc
from pathlib import Path

def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


def get_model(config):
    model = Sequential()
    W = chainer.initializers.HeNormal(1 / np.sqrt(1 / 2), dtype=np.float32)
    bias = chainer.initializers.Zero(dtype=np.float32)
    layers = config[0]['model']
    for layer in layers:
        name = layer['name']
        parameter = copy.deepcopy(layer['parameter'])
        if name[0] == 'L':
            if 'Conv' in name or 'Linear' in name:
                parameter.update({'initialW': W, 'initial_bias': bias})
            add_layer = eval(name)(**parameter)
        elif name[0] == 'F':
            if len(parameter) == 0:
                add_layer = partial(eval(name))
            else:
                add_layer = partial(eval(name), **parameter)
        elif name[0] == 'M':
            add_layer = L.BatchNormalization(size=parameter['size'])
        model.append(add_layer)
    return model


def get_predict_value(data, model, gpu):
    data = data.astype(np.float32)
    batch_size = 16
    i = 0
    out_list = []
    while i * batch_size < data.shape[0]:
        voxel = data[i * batch_size:(i + 1) * batch_size]
        voxel = Variable(voxel)
        if gpu >= 0:
            voxel.to_gpu()
        with no_backprop_mode(), chainer.using_config('train', False):
            pred_score = F.sigmoid(model(voxel))
        pred_score = chainer.cuda.to_cpu(pred_score.data).ravel()
        out_list.extend(pred_score)
        i += 1
    return np.array(out_list) 


def output_score(input_path, output_path, predict_value, resid, resname):
    global_score = np.mean(predict_value)
    with open(output_path, 'w') as f:
        f.write('# Model name : {}\n'.format(input_path))
        f.write('# Model Quality Score : {:.5f}\n'.format(global_score))
        f.write('Resid\tResname\tScore\n')
        for i in range(len(resname)):
            f.write('{}\t{}\t{:.5f}\n'.format(resid[i], resname[i], predict_value[i]))


def output_score_stdout(input_path, global_score):
    print('# Model name : {}'.format(input_path))
    print('# Model Quality Score: {:.5g}'.format(global_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict ')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input_path', '-i', help='Input data path')
    parser.add_argument('--input_dir_path','-d', help='Input dir data path')
    parser.add_argument('--fasta_path', '-f', required=True, help='Reference FASTA Sequence path')
    parser.add_argument('--model_path', '-m', help='Pre-trained model path', default='../trained_model.npz')
    parser.add_argument('--preprocess_dir', '-p', help='Input preprocess directory')
    parser.add_argument('--output_dir', '-o', help='Output directory')
    parser.add_argument('--save_res', '-s', action='store_true', help='save score of each residue')
    args = parser.parse_args()
    model = get_model(json.load(open('./config.json', 'r'), object_hook=hinted_tuple_hook))
    load_npz(file=args.model_path, obj=model, path='updater/model:main/predictor/')
    if args.gpu >= 0:
        ws_size = 512 * 1024 * 1024 # 512 MB
        chainer.cuda.set_max_workspace_size(ws_size) # set large workspace size
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    
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
        input_path = args.input_path
        data, resname, resid = get_voxel_predict(input_path=input_path, fasta_path=args.fasta_path, pssm_path=pssm_path, ss_path=ss_path, acc20_path=acc20_path, buffer=28, width=1)
        predict_value = get_predict_value(data=data, model=model, gpu=args.gpu)
        global_score = np.mean(predict_value)
        output_score_stdout(input_path, global_score)
        output_path = (output_dir / Path(input_path).stem).with_suffix('.txt')
        output_score(input_path, output_path, predict_value, resid, resname)
        
    elif args.input_dir_path:
        input_dir_path = Path(args.input_dir_path)
        model_array = []
        global_score_array = []
        for input_path in input_dir_path.iterdir():
            model_name = input_path.stem
           
            try:
                data, _, resid = get_voxel_predict(input_path=str(input_path), fasta_path=args.fasta_path, pssm_path=pssm_path, ss_path=ss_path, acc20_path=acc20_path, buffer=28, width=1)
            except Exception as e:
                model_array.append(model_name)
                global_score_array.append(None)
                print('make_voxel error:'+str(input_path), e)
                import traceback
                traceback.print_exc()
            else:
                predict_value = get_predict_value(data=data, model=model, gpu=args.gpu)
                model_array.append(model_name)
                global_score = np.mean(predict_value)
                global_score_array.append(global_score)
                output_score_stdout(input_path, global_score)
                if args.save_res:
                    output_path = (output_dir/model_name).with_suffix('.txt')
                    output_score(input_path, output_path, predict_value, resid, resname)
                del data, resid
                gc.collect()
        
        df = pd.DataFrame({'Score': global_score_array, 'Model_name': model_array}).set_index('Model_name')
        df = df.sort_index()
        output_path = (output_dir/target).with_suffix('.csv')
        df.to_csv(output_path)

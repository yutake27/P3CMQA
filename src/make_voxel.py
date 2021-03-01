import os
import subprocess
import sys
import tempfile
from functools import reduce
from pathlib import Path

import numpy as np
from Bio import AlignIO, SeqIO
from prody import LOGGER, parseMMCIF, parsePDB

from amino import get_atom_type_array
from calculate_axis import get_axis

LOGGER.verbosity = 'none'


def align_fasta(input_mol, target_fasta_path):
    _, input_fasta_path = tempfile.mkstemp(suffix='.fasta')
    f = open(input_fasta_path, 'w')
    f.write('>temp\n')
    if len(input_mol.select('name CA').getSequence()) < 25:
        return None, None, None
    else:
        f.write(reduce(lambda a, b: a + b, input_mol.select('name CA').getSequence()))
        f.close()
        _, needle_path = tempfile.mkstemp(suffix='.needle')
        cmd = ['needle', '-outfile', needle_path, '-asequence', input_fasta_path, '-bsequence', target_fasta_path,
               '-gapopen', '10', '-gapextend', '0.5']
        subprocess.call(cmd)
        needle_result = list(AlignIO.parse(needle_path, 'emboss'))[0]
        input_seq, target_seq = np.array(list(str(needle_result[0].seq))), np.array(list(str(needle_result[1].seq)))
        input_seq, target_seq = input_seq[np.where(target_seq != '-')], target_seq[np.where(input_seq != '-')]
        input_align_indices = np.where(target_seq != '-')[0]
        target_align_indices = np.where(input_seq != '-')[0]
        align_pdb = input_mol.select('resindex ' + reduce(lambda a, b: str(a) + ' ' + str(b), input_align_indices))
        os.remove(input_fasta_path)
        os.remove(needle_path)
        return align_pdb, target_align_indices, input_align_indices


def calc_occupancy_bool(atom_coord, channel, buffer, width, axis):
    atom_coord = np.dot(atom_coord, np.linalg.inv(axis))
    atom_coord += np.array([buffer // 2, buffer // 2, buffer // 2])
    index = np.where(np.all(atom_coord >= 0, 1) * np.all(atom_coord < buffer, 1))
    atom_coord = (atom_coord / width).astype(np.int)
    atom_coord, channel = atom_coord[index], channel[index]
    length = int(buffer / width)
    occus = np.zeros([length, length, length, channel.shape[1]])
    for i in range(len(atom_coord)):
        h = channel[i]
        occus[atom_coord[i][0]][atom_coord[i][1]][atom_coord[i][2]] += h
    occus = occus.transpose([3, 0, 1, 2])
    index = occus[13,:,:,:] > 1
    occus[:,index] = np.concatenate((np.logical_or(occus[:14,index],0), occus[14:34,index]/occus[13,index],
                        np.logical_or(occus[34:37,index],0), np.array([occus[37,index]/occus[13,index]])))
    return occus


def get_atom_target_index(target_index, input_index, resindices):
    """get target indices of input atom
    
    Arguments:
        target_index {array} -- target(native) residue index
        input_index {array} -- input residue index
        resindices {array} -- target residue index of input atoms
    
    Returns:
        array -- target indices of input atom
    """
    dic = {}
    for i,j in zip(input_index, target_index):
        dic[i] = j
    atom_target_residue_index = []
    for index in resindices:
        atom_target_residue_index.append(dic[index])

    return atom_target_residue_index


def make_voxel_train(input_mol, target_mol, pssm, predicted_ss, predicted_acc20, buffer, width):
    input_mol, target_index, input_index = align_by_resid(input_mol, target_mol)
    atom_coord = input_mol.getCoords()
    CA_list, C_list, N_list = input_mol.select('name CA').getCoords(), input_mol.select(
        'name C').getCoords(), input_mol.select('name N').getCoords()
    channel = get_atom_type_array(res_name=input_mol.getResnames(), atom_name=input_mol.getNames())
    atom_target_index = get_atom_target_index(target_index, input_index, input_mol.getResindices())
    residue_info = np.hstack((pssm, predicted_ss, predicted_acc20))
    channel = np.append(channel, residue_info[atom_target_index], axis=1)
    output = []
    for ca_coord, c_coord, n_coord in zip(CA_list, C_list, N_list):
        axis = get_axis(CA_coord=ca_coord, N_coord=n_coord, C_coord=c_coord)
        atom = atom_coord - ca_coord
        occus = calc_occupancy_bool(atom_coord=atom, channel=channel, buffer=buffer, width=width, axis=axis)
        output.append(occus)
    output = np.array(output, dtype=np.float32)
    return output, input_mol


def make_voxel_predict(input_mol, fasta_path, pssm, predicted_ss, predicted_acc20, buffer, width):
    input_mol, target_index, input_index = align_fasta(input_mol, fasta_path)
    atom_coord = input_mol.getCoords()
    CA_list, C_list, N_list = input_mol.select('name CA').getCoords(), input_mol.select(
        'name C').getCoords(), input_mol.select('name N').getCoords()
    channel = get_atom_type_array(res_name=input_mol.getResnames(), atom_name=input_mol.getNames())
    atom_target_index = get_atom_target_index(target_index, input_index, input_mol.getResindices())
    residue_info = np.hstack((pssm, predicted_ss, predicted_acc20))
    channel = np.append(channel, residue_info[atom_target_index], axis=1)
    output = []
    for ca_coord, c_coord, n_coord in zip(CA_list, C_list, N_list):
        axis = get_axis(CA_coord=ca_coord, N_coord=n_coord, C_coord=c_coord)
        atom = atom_coord - ca_coord
        occus = calc_occupancy_bool(atom_coord=atom, channel=channel, buffer=buffer, width=width, axis=axis)
        output.append(occus)
    output = np.array(output, dtype=np.float32)
    return output, input_mol


def get_voxel_train(input_path, target_path, pssm_path, ss_path, acc20_path, buffer, width):
    input_mol, target_mol = parsePDB(input_path), parsePDB(target_path)
    input_mol = input_mol.select('element C or element N or element O or element S')
    target_mol = target_mol.select('element C or element N or element O or element S')
    pssm = get_pssm(pssm_path)
    predicted_ss = get_predicted_ss(ss_path)
    predicted_acc20 = get_predicted_acc20(acc20_path)
    occus,input_mol = make_voxel_train(input_mol=input_mol, target_mol=target_mol, pssm=pssm,
            predicted_ss=predicted_ss, predicted_acc20=predicted_acc20, buffer=buffer, width=width)
    return occus, input_mol.select('name CA').getResnames(), input_mol.select('name CA').getResnums()


def get_voxel_predict(input_path, fasta_path, pssm_path, ss_path, acc20_path, buffer, width):
    input_mol = parse_input_file(input_path)
    input_mol = input_mol.select('element C or element N or element O or element S')
    pssm = get_pssm(pssm_path)
    predicted_ss = get_predicted_ss(ss_path)
    predicted_acc20 = get_predicted_acc20(acc20_path)
    occus,input_mol = make_voxel_predict(input_mol=input_mol, fasta_path=fasta_path, pssm=pssm,
                                predicted_ss=predicted_ss, predicted_acc20=predicted_acc20, buffer=buffer, width=width)
    return occus, input_mol.select('name CA').getResnames(), input_mol.select('name CA').getResnums()


def parse_input_file(input_path: str):
    input_path = Path(input_path)
    if input_path.suffix == '.pdb':
        input_mol = parsePDB(input_path)
    elif input_path.suffix == '.cif':
        input_mol = parseMMCIF(input_path)
    else:
        input_mol = parsePDB(input_path)
        if input_mol is None:
            input_mol = parseMMCIF(input_path)
    return input_mol


def get_pssm(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[3:]
        lines = [list(filter(lambda x: len(x) != 0, l.split(' ')))[2:22] for l in lines]
        lines = list(filter(lambda x: len(x) == 20, lines))
        pssm = np.array(lines, dtype=np.float32)
        pssm = (pssm+13)/26
        return pssm


def get_predicted_acc20(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1].strip().split()
        rsa = np.array(lines, dtype=np.float32)
        rsa = np.reshape(rsa, [rsa.shape[0], 1])
        return rsa/100

def get_predicted_ss(file_path):
    ss_dict = {'H': 0, 'G': 0, 'I': 0, 'E': 1, 'B': 1, 'b': 1, 'T': 2, 'C': 2}
    with open(file_path, 'r') as f:
        lines = f.readlines()[1].strip()
        ss = np.array([ss_dict[i] for i in lines], dtype=np.int32)
        ss = np.identity(3, dtype=np.bool)[ss]
        return ss

def align_by_resid(input_mol, target_mol):
    target_resid, input_resid = target_mol.select('calpha').getResnums(), input_mol.select('calpha').getResnums()
    aligned_input_index = np.where(np.in1d(input_resid, target_resid))[0]
    aligned_target_index = np.where(np.in1d(target_resid, input_resid))[0]

    if len(input_mol.select('name CA').getSequence()) < 25 or len(
            target_mol.select('name CA').getSequence()) < 25 or len(aligned_target_index) < 25:
        return None, None, None
    aligned_input_mol = input_mol.select('resindex ' + reduce(lambda a, b: str(a) + ' ' + str(b), aligned_input_index))
    return aligned_input_mol, aligned_target_index, aligned_input_index


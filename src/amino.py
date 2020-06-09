import numpy as np
from collections import OrderedDict


def flatten(array):
    return [e2 for e1 in array for e2 in e1]


class Atom():
    def __init__(self, resname, name, C_terminal=False):
        self.resname = resname
        self.name = name
        self.C_terminal = C_terminal

    def get_type_from_array(self, name_array):
        for resname, name in name_array:
            if self.get_type(resname, name) is True:
                return True
        return False

    def get_type(self, resname, name):
        return ((self.resname == resname) | (resname == 'ALL')) & (self.name == name)


def get_atom_type_array(atom_name, res_name):
    """原子ごとの特徴量を返す関数
    
    Arguments:
        atom_name {np.ndarray} -- 原子名の配列
        res_name {np.ndarray} -- 各原子ごとの残基名の配列
    
    Returns:
        np.ndarray -- 原子ごとの特徴が入った二次元配列
    """
    channels = []
    for i in range(len(atom_name)):
        if i == len(atom_name) - 1:
            channels.append(atom_type(resname=res_name[i], name=atom_name[i]))
        else:
            channels.append(atom_type(resname=res_name[i], name=atom_name[i], C_terminal=True))
    channels = np.array(channels)
    return channels


_order = ('Sulfur', 'N_amide', 'N_aromatic', 'N_guanidinium', 'N_ammonium', 'O_carbonyl', 'O_hydroxyl', 'O_carboxyl',
          'C_sp2', 'C_atomatic', 'C_sp3', 'main_chain', 'CA', 'Occupancy')


def atom_type(resname, name, C_terminal=False):
    if C_terminal is True and name == 'O':
        name = 'OXT'
    atom = Atom(resname=resname, name=name)
    props = OrderedDict()
    props['Sulfur'] = atom.get_type_from_array([('CYS', 'SG'), ('MET', 'SD'), ('MSE', 'SE')])
    props['N_amide'] = atom.get_type_from_array([('ASN', 'ND2'), ('GLN', 'NE2'), ('ALL', 'N')])
    props['N_aromatic'] = atom.get_type_from_array([('HIS', 'ND1'), ('HIS', 'NE1'), ('TRP', 'NE1')])
    props['N_guanidinium'] = atom.get_type_from_array([('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2')])
    props['N_ammonium'] = atom.get_type_from_array([('LYS', 'NZ')])
    props['O_carbonyl'] = atom.get_type_from_array([('ASN', 'OD1'), ('GLN', 'OE1'), ('ALL', 'O')])
    props['O_hydroxyl'] = atom.get_type_from_array([('SET', 'OG'), ('THR', 'OG1'), ('TYR', 'OH')])
    props['O_carboxyl'] = atom.get_type_from_array(
        [('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2'), ('ALL', 'OXT')])
    props['C_sp2'] = atom.get_type_from_array(
        [('ARG', 'CZ'), ('ASN', 'CG'), ('ASP', 'CG'), ('GLN', 'CD'), ('GLU', 'CD'), ('ALL', 'C')])
    props['C_atomatic'] = atom.get_type_from_array(
        [('HIS', 'CG'), ('HIS', 'CD2'), ('HIS', 'CE1'), ('PHE', 'CG'), ('PHE', 'CD1'), ('PHE', 'CD2'), ('PHE', 'CE1'),
         ('PHE', 'CE2'), ('PHE', 'CZ'), ('TRP', 'CD1'), ('TRP', 'CD2'), ('TRP', 'CE1'), ('TRP', 'CE2'), ('TRP', 'CZ1'),
         ('TRP', 'CZ2'), ('TRP', 'CH2'), ('TYR', 'CG'), ('TYR', 'CD1'), ('TYR', 'CD2'), ('TYR', 'CE1'), ('TYR', 'CE2'),
         ('TYR', 'CZ')])
    props['C_sp3'] = atom.get_type_from_array(
        [('ALA', 'CB'), ('ARG', 'CB'), ('ARG', 'CG'), ('ARG', 'CD'), ('ASN', 'CB'), ('ASP', 'CB'), ('CYS', 'CB'),
         ('GLN', 'CB'), ('GLN', 'CG'), ('GLU', 'CB'), ('GLU', 'CG'), ('HIS', 'CB'), ('ILE', 'CG1'), ('ILE', 'CG2'),
         ('ILE', 'CD1'), ('LEU', 'CB'), ('LEU', 'CG'), ('LEU', 'CD1'), ('LEU', 'CD2'), ('LYS', 'CB'), ('LYS', 'CG'),
         ('LYS', 'CD'), ('LYS', 'CE'), ('MET', 'CB'), ('MET', 'CG'), ('MET', 'CE'), ('MSE', 'CB'), ('MSE', 'CG'),
         ('MSE', 'CE'), ('PHE', 'CB'), ('PRO', 'CB'), ('PRO', 'CG'), ('PRO', 'CD'), ('SER', 'CB'), ('THR', 'CB'),
         ('THR', 'CG2'), ('TRP', 'CB'), ('TYR', 'CB'), ('VAL', 'CB'), ('VAL', 'CG1'), ('VAL', 'CG2'), ('ALL', 'CA')])
    props['main_chain'] = atom.get_type_from_array([('ALL', 'C'), ('ALL', 'CA'), ('ALL', 'N')])
    props['Occupancy'] = True
    props['CA'] = atom.get_type_from_array([('ALL', 'CA')])
    channels = np.zeros(len(_order), dtype=np.float32)

    for i, p in enumerate(_order):
        channels[i] = props[p]

    return channels

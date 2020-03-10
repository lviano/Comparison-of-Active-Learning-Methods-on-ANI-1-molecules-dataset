import numpy as np
from ase.io import read,write
from ase.visualize import view
import h5py, sys
#import quippy as qp
import ase
import os.path
import json
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
#sys.path.insert(0, 'ANI-1_release/')
sys.path.insert(0,'../ANI-1_release/readers/lib/')

#from ase.io import read,write
#from ase.visualize import view
import pyanitools as pya
sys.path.insert(0, './ml_tools/')  #/local/switchdrive/semester_project_viano/scripts
from ml_tools.descriptors import RawSoapInternal
from ml_tools.models.KRR import KRR,TrainerCholesky,KRRFastCV
from ml_tools.kernels import KernelPower,KernelSum
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score,load_pck,tqdm_cs
from ml_tools.split import KFold,LCSplit,ShuffleSplit
from ml_tools.compressor import FPSFilter


from glob import glob
from math_tools import *

z2symb = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
          8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
          14: 'Si', 15: 'P',
          16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc',
          22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co',
          28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As',
          34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',
          40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
          46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb',
          52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
          58: 'Ce', 59: 'Pr', 60: 'Nd', 62: 'Sm', 63: 'Eu', 64: 'Gd',
          65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
          71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os',
          77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
          83: 'Bi'
}
sym2z = {sym:z for z,sym in z2symb.iteritems()}

hartree2meV = 27.2114 * 1000
self_energy = {1:-0.500607632585*hartree2meV,6:-37.8302333826*hartree2meV,
               7:-54.5680045287*hartree2meV,8:-75.0362229210*hartree2meV}

def load_(frames_indices, path):
    fns = glob(path)
    print(fns)
    Nmol = []
    confstrides=[[] for _ in range(len(fns))]
    old_to_append = [0]
    for i,fn in enumerate(fns):
        Nconf = []
        with h5py.File(fn, 'r') as f:
            for key in f.keys():
                dset = f[key]
                Nmol.append(len(dset.keys()))
                for k,v in dset.items():
                    Nconf.append(v['coordinates'].shape[0])
                Nmol,Nconf
        to_append = np.cumsum(Nconf)
        confstrides[i] = np.concatenate([confstrides[i],to_append])
    file_sep = np.cumsum([confstrides[i][-1] for i in range(len(fns))])
    local_ids = [[[] for _ in range(Nmol[i])] for i in range(len(fns))]
    for idx in frames_indices:
        ff = file_sep - idx
        ifile = len(ff[ff<0])
        if ifile > 0:
            aa = confstrides[ifile] - idx + file_sep[ifile - 1]
        else:
            aa = confstrides[ifile] - idx
        imol = len(aa[aa<0])
        local_ids[ifile][imol].append(aa[imol])
    frames = []
    labels = []
    mol_indices = []
    for i,fn in enumerate(fns):
        #print(i,fn)
        with h5py.File(fn, 'r') as f:
            for key in f.keys():
                #print(key)
                dset = f[key]
                for N, [(k,v),ids] in enumerate(zip(dset.items(),local_ids[i])):
                    #print(N, k,v, ids)
                    for ind in ids:
                        pos = v['coordinates'][ind]
                        energy = v['energies'][ind]
                        S = v['species']
                        #sm = v['smiles']

                        numbers = np.array([sym2z[sym] for sym in S])
                        #print(numbers)
                        energy_components = 0
                        for _, num in enumerate(numbers.astype(int)):
                            energy_components += self_energy[num]
                        frame = ase.Atoms(      numbers=numbers[:].astype(int),
                                                positions=pos,
                                                pbc=False,cell=np.eye(3)*20,
                                                info=dict(E_unit='meV', pos_unit='angstrom'))
                        frames.append(frame)
                        labels.append((energy*hartree2meV - energy_components)/numbers.shape[0])
                        mol_indices.append(file_sep[np.heaviside(i-1,0).astype(int)]*np.heaviside(i,0) + confstrides[i][np.heaviside(N-1,0).astype(int)]*np.heaviside(N,0) + ind)
    return frames, np.array(labels).reshape(len(frames),1), frames_indices.reshape(len(frames),1) #np.array(mol_indices).reshape(len(frames),1)

def FPS_reduction(frames):
    X = compute_soap_matrix(frames)
    # set up the kernel parameters
    kernel = KernelPower(zeta = 1)

    Nselect = 250
    compressor = FPSFilter(Nselect,kernel,act_on='feature',precompute_kernel=True,disable_pbar=True)
    compressor.fit(X,dry_run=True)

    indices = compressor.selected_ids
    min_distance2 = compressor.min_distance2

    return indices,min_distance2,X

def load_COMP(frame_indices):

    number_of_frames = len(frame_indices)
    frames = []*number_of_frames
    labels = np.zeros([number_of_frames,1])
    mol_indices = np.zeros([number_of_frames,1])
    with h5py.File('../ANI-1_release/molecules_COMP_dataset_1.hdf5', 'r') as f:
        for ind, iframe in enumerate(frame_indices):
            dset_energies = f["/"+ str(iframe)]['energies']
            dset_indices = f["/"+ str(iframe)]['indices']
            dset_numbers = f["/"+ str(iframe)]['numbers']
            dset_positions = f["/"+ str(iframe)]['positions']
            dset_labels = f["/"+ str(iframe)]['labels']
            dset_formation_energies = f["/"+ str(iframe)]['formation_energies']

            numbers=dset_numbers[:]
            positions=dset_positions[:,:]
            frame = ase.Atoms(numbers=dset_numbers[:].astype(int),
                                positions=dset_positions[:,:],
                                pbc=False,cell=np.eye(3)*20,
                                info=dict(E_unit='meV', pos_unit='angstrom'))
            frames.append(frame)
            mol_indices[ind] = dset_indices
            labels[ind] = dset_formation_energies
    return frames, labels, mol_indices

def feature_reduction(X, Y, number_of_features, mode = 'VarianceThreshold'):
    if mode == 'VarianceThreshold':
        print(number_of_features)
        med_variance = np.median(np.std(X, axis = 0))

        sel = VarianceThreshold(threshold = med_variance )

        X_feature_matrix = sel.fit_transform(X)
        ind_selected = sel.get_support(indices = True)

    elif mode == 'Lasso':
        lasso = Lasso(alpha = 1e-7).fit(X, Y)
        model = SelectFromModel(lasso, prefit=True)
        X_feature_matrix = model.transform(X)
        ind_selected = model.get_support(indices = True)
    return X_feature_matrix, ind_selected

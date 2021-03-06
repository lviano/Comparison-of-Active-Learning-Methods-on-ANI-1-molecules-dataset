import sys,os
sys.path.insert(0,'ml_tools/')

from ml_tools.descriptors import RawSoapInternal
from ml_tools.models.KRR import KRR,TrainerCholesky,KRRFastCV
from ml_tools.kernels import KernelPower,KernelSum
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score,load_pck,tqdm_cs
from ml_tools.split import KFold,LCSplit,ShuffleSplit
from ml_tools.compressor import FPSFilter
from sklearn.decomposition import PCA

import numpy as np
from ase.io import read,write
from ase.visualize import view
import matplotlib.pyplot as plt


from glob import glob
from load_data import *
from math_tools import *
from active_learning import *
from plots import *

number_of_molecules = 22057374
 #61760 for first 2 dataset #22057374 for all dataset #864898 for first 4 dataset, 4360294 for 6 databases
number_of_molecules_COMP = 83670
number_of_data = 20000 #20000
number_of_data_COMP = 1499 #20000
number_of_seeds = 10
conformer_for_each_molecule = 80
proportion = 1./conformer_for_each_molecule
number_not_conformer = int(number_of_data*proportion)
# Prepare Active Learning dataset
#No permutation--create an nexcess of small molecules
with open('../ANI-1_release/frames21.json', 'r') as fp:
    frames_different_molecule = json.load(fp)
random_indices = np.random.RandomState(seed=882).permutation(np.array(frames_different_molecule['frames']))
print(random_indices.shape[0])
random_indices = random_indices[:number_not_conformer]
random_indices = random_indices.repeat(conformer_for_each_molecule) + np.tile(np.arange(2,conformer_for_each_molecule+2),number_not_conformer)
print(random_indices, random_indices.shape)
frames, Y, mol_indices = load_(random_indices, '../ANI-1_release/ani_gdb_s*.h5')
print(Y.shape, mol_indices.shape)

### Prepare Active learning with FPS ###
ind_selected,_,X = FPS_reduction(frames)
print(X.shape)
train_test_ratio = 0.00025
final_train_test_ratio = 0.015
X_feature_matrix = X[:,ind_selected[:500]]
print(X_feature_matrix.shape)

### Visualize samples ###
"""
plt.figure()
plt.scatter(np.sort(mol_indices, 0), Y[np.argsort(mol_indices.flatten())], c = 'grey', alpha = 0.4)

al = plt.scatter(np.sort(indices,0), Y_selected_AL[np.argsort(indices.flatten())], c = 'green', alpha = 0.4)
start = plt.scatter(np.sort(mol_label_train, 0), Y_train[np.argsort(mol_label_train.flatten())], c = 'red', alpha = 0.5)
rd = plt.scatter(np.sort(mol_label_train_comp, 0), Y_train_comp[np.argsort(mol_label_train_comp.flatten())], c = 'yellow', alpha = 0.4)
plt.legend((al, start, rd),
           ('AL sampling', 'Starting point', 'Random Sampling'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.xlabel('Molecule Index')
plt.ylabel('Energy [meV]')
plt.title('Number of data = ' + str(number_of_data) + ' train test ratio = ' + str(final_train_test_ratio))
"""

#Upload fresh data from COMP6 dataset for benchmarking
with open('../ANI-1_release/frames_gdb_789.json', 'r') as fp:
    indices_not_conformersCOMP = json.load(fp)
indicesCOMP = np.array(indices_not_conformersCOMP['frames'])[:number_of_data_COMP]
#indicesCOMP = np.arange(number_of_molecules_COMP, dtype = int) for all all the molecules
frames_benchmark, Y_benchmarkCOMP, mol_indices_comp = load_(indicesCOMP, '../ANI-1_release/gdb*.h5')
X_benchmark = compute_soap_matrix(frames_benchmark)

X_benchmarkCOMP = X_benchmark[:,ind_selected[:500]]

#Upload fresh data to benchmark on other molecules in ANI

random_indices_test = np.random.RandomState(seed=882).permutation(np.array(frames_different_molecule['frames_test']))

random_indices_test = random_indices[:number_of_data_COMP]  #number_of_data_COMP is the number of test data
frames_benchmark, Y_benchmarkANI, mol_indices_comp = load_(random_indices_test, '../ANI-1_release/ani_gdb_s*.h5')

X_benchmark = compute_soap_matrix(frames_benchmark)

X_benchmarkANI = X_benchmark[:,ind_selected[:500]]
full_errors = {}
full_errors['MAEANI'], full_errors['MSEANI'] = compute_loss(X_benchmarkANI, ridge_regression(Y,X_feature_matrix, np.ones_like(X_feature_matrix[:,0])), Y_benchmarkANI)
full_errors['MAECOMP'], full_errors['MSECOMP'] = compute_loss(X_benchmarkCOMP, ridge_regression(Y,X_feature_matrix, np.ones_like(X_feature_matrix[:,0])), Y_benchmarkCOMP)
print(full_errors.items())
matrices = {}
matrices['pool'] = X_feature_matrix.tolist()
matrices['labels'] = Y.tolist()
matrices['mol_indices'] = mol_indices.tolist()
matrices['ANI_bencjmark'] = X_benchmarkANI.tolist()
matrices['COMP_benchmark'] = X_benchmarkCOMP.tolist()
matrices['labels_ANI'] = Y_benchmarkANI.tolist()
matrices['labels_COMP'] =  Y_benchmarkCOMP.tolist()
with open('../results/matrix250x80.json','w') as fp:
    json.dump(matrices,fp)
### Draw learning curves #####
method_list = ['RAND','FPS2','ECLA', 'QBC', 'ECNA', 'EGAL']
ratios = np.logspace(0,3,10)*train_test_ratio 
compute_learning_curve(ratios, number_of_data, proportion, mol_indices, Y, X_feature_matrix, X_benchmarkCOMP, Y_benchmarkCOMP, X_benchmarkANI, Y_benchmarkANI,method_list, number_of_seeds, restart=False)
draw_learning_curve(ratios, method_list, number_of_data_COMP, number_of_data, number_of_seeds, full_errors)
visualize_PCA(number_of_seeds, method_list)


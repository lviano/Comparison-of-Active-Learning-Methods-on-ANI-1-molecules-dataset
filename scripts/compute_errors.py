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


with open('../results/matrix.json','r') as fp:
    matrices = json.load(fp)
matrices['ANI_bencjmark']

full_errors = {}
full_errors['MAEANI'], full_errors['MSEANI'] = compute_loss(X_benchmarkANI, ridge_regression(Y,X_feature_matrix, np.ones_like(X_feature_matrix[:,0])), Y_benchmarkANI)
full_errors['MAECOMP'], full_errors['MSECOMP'] = compute_loss(X_benchmarkCOMP, ridge_regression(Y,X_feature_matrix, np.ones_like(X_feature_matrix[:,0])), Y_benchmarkCOMP)
print(full_errors.items())
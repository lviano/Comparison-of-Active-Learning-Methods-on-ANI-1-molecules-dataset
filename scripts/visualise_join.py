import numpy as np
from ase.io import read,write
from ase.visualize import view

from glob import glob
from load_data import *
from math_tools import *
from active_learning import *
from plots import *

train_test_ratio = 0.00025

ratios = np.logspace(0,3,10)*train_test_ratio 

method_list = ['RAND', 'ECNA', 'QBC', 'FPS2', 'EGAL']
with open('../results/stateALL.json', 'r') as fp:
            state = json.load(fp)
vecMAE_ALCOMP = np.array(state['MAECOMP'])
vecMSE_ALCOMP = np.array(state['MSECOMP'])

vecMAE_ALANI = np.array(state['MAEANI'])
vecMSE_ALANI = np.array(state['MSEANI'])

vecMAE_ALU = np.array(state['MAEU'])
vecMSE_ALU = np.array(state['MSEU'])
with open('../results/stateEGALgood.json', 'r') as fp:
            state = json.load(fp)
vecMAE_ALCOMP_EGAL = np.array(state['MAECOMP'])
vecMSE_ALCOMP_EGAL = np.array(state['MSECOMP'])

vecMAE_ALANI_EGAL = np.array(state['MAEANI'])
vecMSE_ALANI_EGAL = np.array(state['MSEANI'])

vecMAE_ALU_EGAL = np.array(state['MAEU'])
vecMSE_ALU_EGAL = np.array(state['MSEU'])

vecMAE_ALCOMP = np.concatenate((vecMAE_ALCOMP ,vecMAE_ALCOMP_EGAL ), axis = 2)
vecMSE_ALCOMP = np.concatenate((vecMSE_ALCOMP ,vecMSE_ALCOMP_EGAL ), axis = 2)
vecMAE_ALANI = np.concatenate((vecMAE_ALANI ,vecMAE_ALANI_EGAL ), axis = 2)
vecMSE_ALANI = np.concatenate((vecMSE_ALANI ,vecMSE_ALANI_EGAL ), axis = 2)
vecMSE_ALU = np.concatenate((vecMSE_ALU ,vecMSE_ALU_EGAL ), axis = 2)
vecMAE_ALU = np.concatenate((vecMAE_ALU ,vecMAE_ALU_EGAL ), axis = 2)
state = {
                'MAECOMP': vecMAE_ALCOMP.tolist(),
                'MSECOMP': vecMSE_ALCOMP.tolist(),

                'MAEANI': vecMAE_ALANI.tolist(),
                'MSEANI': vecMSE_ALANI.tolist(),

                'MAEU': vecMAE_ALU.tolist(),
                'MSEU': vecMSE_ALU.tolist(),
            }
        
with open('../results/state.json', 'w') as fp:
    json.dump(state, fp)
number_of_data = 20000 #20000 #20000
number_of_data_COMP = 1499 #20000
number_of_seeds = 10

full_errors = {
    'MAEANI': 30.90291,
    'MSEANI': 33.41579,
    'MAECOMP': 34.75238,
    'MSECOMP': 31.05879,
}

draw_learning_curve(ratios, method_list, number_of_data_COMP, number_of_data, number_of_seeds, full_errors)
number_of_seeds = 1
#visualize_PCA(number_of_seeds, method_list)
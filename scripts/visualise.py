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
method_list = ['RAND', 'EGAL','FPS2', 'ECLA', 'ECNA', 'QBC']
number_of_data = 20000 #20000 #20000
number_of_data_COMP = 2999 #20000
number_of_seeds = 10

full_errors = {
    'MAEANI': 27.24284332837359, 
    'MSECOMP': 36.411158117837125, 
    'MAECOMP': 37.05505183314068, 
    'MSEANI': 36.710993884929017,
}
"""'MAEANI': 30.90291,
    'MSEANI': 33.41579,
    'MAECOMP': 34.75238,
    'MSECOMP': 31.05879, """

draw_learning_curve(ratios, method_list, number_of_data_COMP, number_of_data, number_of_seeds, full_errors)
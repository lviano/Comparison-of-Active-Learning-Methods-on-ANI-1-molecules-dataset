import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from active_learning import *
from sklearn.decomposition import PCA
import json

def compute_learning_curve(ratios, number_of_data, proportion, mol_indices, Y, X_feature_matrix, X_benchmarkCOMP, Y_benchmarkCOMP, X_benchmarkANI, Y_benchmarkANI,method_list, number_of_seeds, restart = False ):
    train_test_ratio = ratios[0]
    vecMAE_ALANI = np.zeros([len(ratios),number_of_seeds, len(method_list)])
    vecMSE_ALANI = np.zeros([len(ratios),number_of_seeds, len(method_list)])
    vecMAE_ALU = np.zeros([len(ratios),number_of_seeds, len(method_list)])
    vecMSE_ALU = np.zeros([len(ratios),number_of_seeds, len(method_list)])
    vecMAE_ALCOMP = np.zeros([len(ratios),number_of_seeds, len(method_list)])
    vecMSE_ALCOMP = np.zeros([len(ratios),number_of_seeds, len(method_list)])


    methods = {
        'QBC' : active_learning_iterate,
        'ECLA': expected_model_change_iterate_average_of_labels,
        'F'   : Fisher_optimality_iterate,
        'F WRONG':  Fisher_optimality_iterate_wrong,
        'ECNA': expected_model_change_iterate_average_of_norms,
        'RAND': random_sampling_iterate,
    }
    free_model_methods = {
        'EGAL': EGAL_iterate,
    }
    temp_method_list = method_list
    seeds = range(number_of_seeds)
    shift = 0
    
    map_ind = {}
    for i in range(len(mol_indices)):
	    map_ind[str(mol_indices[i])] = i
    
    if restart == True:
        with open('../results/state.json', 'r') as fp:
            state = json.load(fp)
        seed_start = state['seed index']
        method_start = state['method index']
        if method_start == len(method_list)-1:
            seeds = np.arange(seed_start + 1, number_of_seeds, dtype = int)
        else:
            seeds = np.arange(seed_start, number_of_seeds, dtype = int)
            temp_method_list = method_list[method_start+1:]
            shift = method_start + 1
            vecMAE_ALCOMP = np.array(state['MAECOMP'])
            vecMSE_ALCOMP = np.array(state['MSECOMP'])

            vecMAE_ALANI = np.array(state['MAEANI'])
            vecMSE_ALANI = np.array(state['MSEANI'])

            vecMAE_ALU = np.array(state['MAEU'])
            vecMSE_ALU = np.array(state['MSEU'])

    for ind, seed in enumerate(seeds):
        print('Seed number ' + str(seed))

        X_train_start, Y_train_start, mol_label_train_start, X_test_start, Y_test_start, mol_label_test_start, train_rows_start = split_data(   Y, X_feature_matrix,mol_indices, train_test_ratio,seed = seed)

        
        
        for in_method, method in enumerate(temp_method_list):
            picked_mol_ind = []
            in_method += shift
            X_train = X_train_start
            Y_train = Y_train_start
            X_test = X_test_start
            Y_test = Y_test_start
            indices = np.copy(mol_label_test_start.flatten())
            w = np.random.RandomState(seed = 3000).rand(X_train.shape[1],1)
            weights_AL = ridge_regression(Y_train, X_train, w)
            vecMAE_ALCOMP[0, ind, in_method], vecMSE_ALCOMP[0, ind, in_method] = compute_loss(X_benchmarkCOMP, weights_AL, Y_benchmarkCOMP) #compute_loss(X_test, weights_AL, Y_test)                                                                                                 
            vecMAE_ALANI[0, ind, in_method], vecMSE_ALANI[0, ind, in_method] = compute_loss(X_benchmarkANI, weights_AL, Y_benchmarkANI)
            vecMAE_ALU[0, ind, in_method], vecMSE_ALU[0, ind, in_method] = compute_loss(X_test, weights_AL, Y_test)
            if method in methods.keys():
                start = time.time()
                for i in range(len(ratios)-1):
                    numbers_of_iteractions = int((ratios[i+1] - ratios[i])*number_of_data)

                    selected_indices, X_train, Y_train, X_test, Y_test, selected_rows, weights_AL = methods[method](X_train, np.copy(Y_train), X_test, Y_test,
                                                                                                                    indices, weights_AL, subgroup_size_ratio = 0.66666,
                                                                                                                    models_in_the_committee = 3,
                                                                                                                    numbers_of_iteractions = numbers_of_iteractions)
                    
                    indices = np.delete(indices, np.where(np.in1d(indices, selected_indices.flatten()))[0])
                    picked_mol_ind = picked_mol_ind + selected_indices.tolist()
                    vecMAE_ALCOMP[i+1, ind, in_method], vecMSE_ALCOMP[i+1, ind, in_method] = compute_loss(X_benchmarkCOMP, weights_AL, Y_benchmarkCOMP)
                    vecMAE_ALANI[i+1, ind, in_method], vecMSE_ALANI[i+1, ind, in_method] = compute_loss(X_benchmarkANI, weights_AL, Y_benchmarkANI)  
                    vecMAE_ALU[i+1, ind, in_method], vecMSE_ALU[i+1, ind, in_method] = compute_loss(X_test, weights_AL, Y_test)                                                                                              
                end = time.time()
                #indices are the not selected ones. See how many conformers have been selected
                number_of_selected_not_conformers = np.delete(mol_indices.flatten()[:int(number_of_data*proportion)], 
                                                    np.where(np.in1d(mol_indices.flatten()[:int(number_of_data*proportion)],indices))).shape[0]
                print(number_of_selected_not_conformers)
                print (method +" Execution Time ")
                print(end - start)
                print (method +" Selected Not Conformers ")
                print(number_of_selected_not_conformers)
            elif method == 'EGAL':
                start = time.time()
                similitude_matrix, densities = prepare_EGAL(X_train, X_test, by_chunks = True)
                for i in range(len(ratios)-1):
                    numbers_of_iteractions = int((ratios[i+1] - ratios[i])*number_of_data)
                    selected_indices, X_train, Y_train, X_test, Y_test, selected_rows, weights_AL = EGAL_iterate(  X_train, np.copy(Y_train), X_test, Y_test, 
                                                                                                                    indices, weights_AL, similitude_matrix, densities, 
                                                                                                                    numbers_of_iteractions = numbers_of_iteractions,
                                                                                                                    by_chunks = True) 
                    
                    
                    indices = np.delete(indices, np.where(np.in1d(indices, selected_indices.flatten()))[0])
                    picked_mol_ind = picked_mol_ind + selected_indices.tolist()
                    vecMAE_ALCOMP[i+1, ind, in_method], vecMSE_ALCOMP[i+1, ind, in_method] = compute_loss(X_benchmarkCOMP, weights_AL, Y_benchmarkCOMP)
                    vecMAE_ALANI[i+1, ind, in_method], vecMSE_ALANI[i+1, ind, in_method] = compute_loss(X_benchmarkANI, weights_AL, Y_benchmarkANI)  
                    vecMAE_ALU[i+1, ind, in_method], vecMSE_ALU[i+1, ind, in_method] = compute_loss(X_test, weights_AL, Y_test) 
                end = time.time()
                print (method +" Execution Time ")
                print(end - start)
                number_of_selected_not_conformers = np.delete(mol_indices.flatten()[:int(number_of_data*proportion)], 
                                                    np.where(np.in1d(mol_indices.flatten()[:int(number_of_data*proportion)],indices))).shape[0]
                print (method +" Selected Not Conformers ")
                print(number_of_selected_not_conformers)
            elif method == 'FPS2':
                start = time.time()
                FPS_indices = prepare_FPS(X_train_start, X_test_start)
                XFPS = np.vstack([X_train_start,X_test_start])[FPS_indices]
                number_of_samples = X_train_start.shape[0]
                YFPS = np.vstack([Y_train_start,Y_test_start])[FPS_indices] 
                for i in range(len(ratios)-1):
                   
                    number_of_samples += int((ratios[i+1] - ratios[i])*number_of_data)
                    picked_mol_ind = mol_indices[FPS_indices[:number_of_samples]]
                    X_trainFPS = XFPS[:number_of_samples]
                    X_testFPS = XFPS[number_of_samples:]
                    Y_trainFPS = YFPS[:number_of_samples]
                    Y_testFPS = YFPS[number_of_samples:]
                    weights_ALFPS = ridge_regression( Y_trainFPS, X_trainFPS, w)
                    vecMAE_ALCOMP[i+1, ind, in_method], vecMSE_ALCOMP[i+1, ind, in_method] = compute_loss(X_benchmarkCOMP, weights_ALFPS, Y_benchmarkCOMP)
                    vecMAE_ALANI[i+1, ind, in_method], vecMSE_ALANI[i+1, ind, in_method] = compute_loss(X_benchmarkANI, weights_ALFPS, Y_benchmarkANI)  
                    vecMAE_ALU[i+1, ind, in_method], vecMSE_ALU[i+1, ind, in_method] = compute_loss(X_testFPS, weights_ALFPS, Y_testFPS) 
                selected_rows = FPS_indices[:number_of_samples]
                end = time.time()
                print (method +" Execution Time ")
                print(end - start)
                number_of_not_selected_not_conformers = np.delete(mol_indices.flatten()[:int(number_of_data*proportion)], 
                                                    np.where(np.in1d(mol_indices.flatten()[:int(number_of_data*proportion)],picked_mol_ind.flatten()))).shape[0]
                
                print (method +" Selected Not Conformers ")
                print(int(number_of_data*proportion) - number_of_not_selected_not_conformers)
            state = {
                'MAECOMP': vecMAE_ALCOMP.tolist(),
                'MSECOMP': vecMSE_ALCOMP.tolist(),

                'MAEANI': vecMAE_ALANI.tolist(),
                'MSEANI': vecMSE_ALANI.tolist(),

                'MAEU': vecMAE_ALU.tolist(),
                'MSEU': vecMSE_ALU.tolist(),
                'seed index': ind,
                'method index': in_method,
            }
        
            with open('../results/state.json', 'w') as fp:
                json.dump(state, fp)
            if method == 'FPS' or method == 'FPS2':
                picked_rows = selected_rows
            else:
                picked_rows = np.array([1])
                for i, picked_mol in enumerate(picked_mol_ind):
                    picked_rows = np.vstack([picked_rows,map_ind['['+str(picked_mol)+']']])
                picked_rows = picked_rows[1:]
            PCAstate = {
                'X_feature_matrix':X_feature_matrix.tolist(),
                'picked_rows': picked_rows.tolist(),
                'train_rows_start': train_rows_start.tolist(),
                'method':method
            }
            
            with open('../results/PCAdata'+method+str(seed)+'.json','w') as fp:
                json.dump(PCAstate,fp)
        #restore the full list of methods. Necessary in case of restart
        temp_method_list = method_list
        shift = 0

def draw_learning_curve(ratios, method_list, number_of_data_COMP, number_of_data, number_of_seeds, full_errors, show = False, format = 'png'):
    colors = {
        'QBC' : 'red',
        'ECLA': 'green',
        'F'   : 'orange',
        'F WRONG'   : 'purple',
        'ECNA': 'pink',
        'RAND': 'blue',
        'EGAL': 'orange',
        'FPS2' : 'grey',
    }

    with open('../results/state.json', 'r') as fp:
        state = json.load(fp)

    vecMSE_AL = {}
    vecMAE_AL = {}
    vecMSE_AL['COMP'] = np.array(state['MSECOMP'])
    vecMAE_AL['COMP'] = np.array(state['MAECOMP'])
    vecMSE_AL['ANI'] = np.array(state['MSEANI'])
    vecMAE_AL['ANI'] = np.array(state['MAEANI'])
    vecMSE_AL['U']= np.array(state['MSEU'])
    vecMAE_AL['U']= np.array(state['MAEU'])

    for test_set_name in vecMAE_AL.keys():
        plt.figure('MAE. Average of ' + str(number_of_seeds) + ' seeds.' + str(number_of_data)
               + ' points from ANI ' + str(number_of_data_COMP) + ' points from ' + test_set_name)

        for in_method, method in enumerate(method_list):
            
            MAE_AL_plot = plt.loglog(ratios*number_of_data,np.mean(vecMAE_AL[test_set_name], axis = 1)[:, in_method], color = colors[method], linestyle = '--', marker = 'x', label = method + ' MAE')
        if test_set_name == 'ANI' or test_set_name == 'COMP':
            plt.loglog(ratios*number_of_data, full_errors['MAE'+test_set_name]*np.ones_like(ratios), linestyle = '-', color = 'black', label = 'full')
        plt.legend()
        plt.xlabel('# Labelled molecules')
        plt.ylabel('Formation Energy MAE [meV/atom] ')
        plt.title('MAE Evolution')
        if (format == 'png' or format == 'both'):
            plt.savefig('../plot/MAEaverage_of_seeds'+test_set_name+'.png')
        if (format == 'eps' or format == 'both'):
            plt.savefig('../plot/MAEaverage_of_seeds'+test_set_name+'.eps')

    for test_set_name in vecMSE_AL.keys():
        plt.figure('MSE. Average of ' + str(number_of_seeds) + ' seeds.' + str(number_of_data)
                    + ' points from ANI ' + str(number_of_data_COMP) + ' points from ' + test_set_name)
        for in_method, method in enumerate(method_list):

            MSE_AL_plot = plt.loglog(ratios*number_of_data,np.mean(vecMSE_AL[test_set_name], axis = 1)[:, in_method], color = colors[method], marker = 'x', linestyle = '--', label = method + ' RMSE')
        if test_set_name == 'ANI' or test_set_name == 'COMP':
            plt.loglog(ratios*number_of_data, full_errors['MSE'+test_set_name]*np.ones_like(ratios), linestyle = '-', color = 'black', label = 'full')
        
        plt.legend(fontsize = 13)
        plt.xlabel('# Labelled molecules', fontsize = 13)
        plt.ylabel('Formation Energy RMSE [meV/atom] ', fontsize = 13)
        plt.title('RMSE Evolution', fontsize = 13)
        if (format == 'png' or format == 'both'):
            plt.savefig('../plot/MSEaverage_of_seeds'+test_set_name+'.png')
        if (format == 'eps' or format == 'both'):
            plt.savefig('../plot/MSEaverage_of_seeds'+test_set_name+'.eps')

    if show:
        plt.show()

def visualize_PCA(number_of_seeds, method_list, show = False, format = 'png'):
    for ind, seed in enumerate(range(number_of_seeds)):
        for in_method, method in enumerate(method_list):
            with open('../results/PCAdata'+method+str(seed)+'.json','r') as fp:
                PCAstate = json.load(fp)
            X_feature_matrix = np.array(PCAstate['X_feature_matrix'])
            selected_rows = np.array(PCAstate['picked_rows'])
            train_rows = np.array(PCAstate['train_rows_start'])
            x2d = PCA(n_components=2).fit_transform(X_feature_matrix)
            print(method + str(np.unique(selected_rows).shape))

            plt.figure()

            plt.scatter(x2d[:,0],x2d[:,1], facecolors = 'grey', alpha = 1)
            al_pca = plt.scatter(x2d[selected_rows, 0], x2d[selected_rows, 1], edgecolors = 'red' , facecolors = 'none')
            start_pca = plt.scatter(x2d[train_rows, 0], x2d[train_rows, 1], edgecolors = 'yellow', facecolors = 'none')

            plt.legend((al_pca, start_pca),
                    (method + ' sampling', 'Starting point'),
                    scatterpoints=1,
                    loc='lower left',
                    ncol=3,
                    fontsize=8)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA View. Number of data = ' + str(X_feature_matrix.shape[0]) + ' train test ratio = ' + str(len(selected_rows)/X_feature_matrix.shape[0]))

            if (format == 'png' or format == 'both'):
                plt.savefig('../plot/PCA'+method+str(seed)+'.png')
            if (format == 'eps' or format == 'both'):
                plt.savefig('../plot/PCA'+method+str(seed)+'.png')
            if show:
                plt.show()

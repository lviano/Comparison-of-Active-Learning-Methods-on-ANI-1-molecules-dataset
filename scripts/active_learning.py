import numpy as np
import os.path
from sklearn.neighbors import KNeighborsRegressor
from numba import jit, cuda
from scipy.sparse import linalg
from sklearn.metrics.pairwise import cosine_similarity
from ml_tools.compressor import FPSFilter
from ml_tools.kernels import KernelPower,KernelSum
import time
from parallel_sort import numpyParallelSort
FLOAT_MAX = 0.5*np.finfo(float).max
FLOAT_MAX_SQRT = np.sqrt(FLOAT_MAX)

def assert_y_tx(y, tx):
    """
        Checks if y and tx have a corresponding shape : (N,) and (N,D)
    """
    assert len(y) == tx.shape[0], "Shape of output vector y (" + str(len(y)) + ",) should correspond to the shape of the input matrix tx (" + str(tx.shape[0]) + "," + str(tx.shape[1]) + "). "

def split_data(y, tx, mol_labels, ratio, seed=1):
    """
    Splits (and shuffles) the data of the output vector y and input matrix tx into two set : training set and test set

    Args:
        y (array): output vector of shape (N,) (N is the data size)
        tx (array): input matrix of shape (N,D) (D is the number of features)
	    ratio (float): percentage of data to be used for training
	    seed (int or float): for random number generation

    Returns:
	    tx_training (array): contains the elements of the original array that have been assigned to the train subset
        y_training (array): contains the elements of the original array that have been assigned to the train subset
	    tx_testing (array): contains the elements of the original array that have been assigned to the test subset
	    y_testing (array): contains the elements of the original array that have been assigned to the test subset

        """
    # set seed
    np.random.seed(seed)
    length = len(tx)
    random_index = np.random.permutation(length)
    q = int(np.floor(length*ratio))

    training_index = random_index[0:q]
    testing_index = random_index[q:]

    return tx[training_index], y[training_index], mol_labels[training_index], tx[testing_index], y[testing_index], mol_labels[testing_index], training_index


def ridge_regression(y, tx, initial_w, max_iters = 16000, mode = 'invert'):
    """
        Ridge regression using normal equations

        Finds the weight vector w* which minimizes the cost function L(w) by solving the normal equations.
        The cost function L(w) is MSE(w) + lambda_*||w||^2 (this means large ||w|| are penalized with lambda_ > 0).

        Args:
            y (array): output vector of shape (N,) (N is the data size)
            tx (array): input matrix of shape (N,D) (D is the number of features)
            lambda_ (float): regularization term (should be > 0)

        Returns:
            array: minimum weight vector w*
            float: loss value associated with w*
    """

    param = {
            'invert' : 0.00000001,
            'iterate': 0.0001,

    }
    if mode == 'invert':
        y = np.array(y)
        tx = np.array(tx)

        assert_y_tx(y, tx)

        a = tx.T.dot(tx) + 2*tx.shape[0]*param[mode]*np.eye(tx.shape[1])
        b = tx.T.dot(y)
        w = np.linalg.solve(a, b)
    elif mode == 'iterate':
        w = ridge_sgd(y, tx, initial_w, max_iters, param[mode])

    return w

@jit(nopython = True, parallel = True)
def ridge_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 20
    old_delta = np.zeros_like(w)
    alpha = 0.9989
    for n in range(max_iters):
        rand_index = np.random.permutation(len(y)) #stochastic Gradient Descent with batch size = 1
        rand_index = rand_index[:batch_size]
        grad = -tx[rand_index].T.dot(y[rand_index] - tx[rand_index].dot(w))
        delta = alpha*old_delta - gamma * grad
        next_w = w + delta
        old_delta = delta
        w = next_w
    return w

@jit(nopython = True, parallel = True)
def ridge_sgd_no_momentum(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 10
    for n in range(max_iters):
        rand_index = np.random.permutation(len(y)) #stochastic Gradient Descent with batch size = 1
        rand_index = rand_index[:batch_size]
        grad = -tx[rand_index].T.dot(y[rand_index] - tx[rand_index].dot(w))

        next_w = w - gamma * grad
        w = next_w
    return w

def compute_ensemble_variance(predictions):

    predictions = np.array(predictions)
    variances = np.var(predictions, axis = 1)

    return variances


def compute_bootstrap_models_predictions(X_train, labels, X_test,  w, subgroup_size_ratio = 0.66666, models_in_the_committee = 3):
    """
    Function that, given the training set X_train, carries out a subsampling
    of different data subgroup. A linear model is trained on each of them so that
    the model commitee is generated

    """
    total_data = X_train.shape[0]
    indices = np.arange(total_data).reshape(total_data,1).repeat(models_in_the_committee, axis = 1)
    [np.random.shuffle(x) for x in indices.T]
    subgroup_size = int(np.floor(subgroup_size_ratio*total_data))
    indices = indices[:subgroup_size,:]
    predictions = np.zeros([X_test.shape[0], models_in_the_committee])

    for ind,one_group_indices in enumerate(indices.T):
        #Select only the rows and labels of the submodel
        labels_submodel = labels[one_group_indices]
        X_submodel = X_train[one_group_indices]
        # train and predict
        w = ridge_regression( labels_submodel, X_submodel, w)
        predictions[:, ind] = X_test.dot(w).flatten()

    return predictions

def random_sampling_iterate(X_train, labels, X_test, Y_test, mol_indices, w, subgroup_size_ratio = 0.66666,
                            models_in_the_committee = 3, numbers_of_iteractions = 10, sgd_iteraction = 10000, mode = 'invert'):
    
    indices = np.random.permutation(mol_indices.shape[0])
    indices = indices[:numbers_of_iteractions]
    selected_data = X_test[indices]
    indices_mol = mol_indices[indices]
    X_train = np.vstack([X_train,selected_data])

    X_test = np.delete(X_test, indices, 0)

    labels = np.vstack([labels,Y_test[indices]])
    Y_test = np.delete(Y_test, indices, 0)
    w = ridge_regression(labels, X_train, w , max_iters=sgd_iteraction, mode = mode)
    return indices_mol, X_train, labels, X_test, Y_test, indices, w

def active_learning_iterate(X_train, labels, X_test, Y_test, mol_indices, w, subgroup_size_ratio = 0.66666,
                            models_in_the_committee = 3, numbers_of_iteractions = 10):

    """
    labels is Y_train
    """
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int)
    print( 'QBC : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
        
    for it in range(numbers_of_iteractions):
        predictions = compute_bootstrap_models_predictions( X_train, labels, X_test, w, subgroup_size_ratio = subgroup_size_ratio,
                                                            models_in_the_committee = models_in_the_committee)
        variances = compute_ensemble_variance(predictions)
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices  = sample_best_scores( variances, X_train, labels, X_test, Y_test,
                                                                                        mol_indices, numbers_of_iteractions,
                                                                                        mode = 'highest')
    w = ridge_regression(labels, X_train, w )
    return indices, X_train, labels, X_test, Y_test, pos, w

def expected_model_change_iterate_average_of_labels(  X_train, labels, X_test, Y_test, mol_indices,  w, subgroup_size_ratio = 0.66666, models_in_the_committee = 3,
                                    numbers_of_iteractions = 10):
    indices = np.zeros(numbers_of_iteractions,dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int)
    print( 'ECLA : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
    for it in range(numbers_of_iteractions):
        # Compute by bootstrap the values will be considered as true labels
        label_prediction = compute_bootstrap_models_predictions(    X_train, labels, X_test, w, subgroup_size_ratio = subgroup_size_ratio,
                                                                    models_in_the_committee = models_in_the_committee)
        bootstrap_label = np.mean(label_prediction, axis = 1)
        # Use all the labelled dataset to compute the model output
        w = ridge_regression( labels, X_train, w)
        prediction_model = X_test.dot(w)

        # Compute active learning scores
        factors = (bootstrap_label.reshape(bootstrap_label.shape[0],1) - prediction_model).repeat(X_test.shape[1],axis = 1)

        partial_gradients = factors*X_test
        scores = np.linalg.norm(partial_gradients, axis = 1)

        # Add the highest scores data and return
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices  = sample_best_scores( scores, X_train, labels, X_test, Y_test,
                                                                                        mol_indices, numbers_of_iteractions,
                                                                                        mode = 'highest')
    return indices, X_train, labels, X_test, Y_test, pos, w

def expected_model_change_iterate_average_of_norms(  X_train, labels, X_test, Y_test, mol_indices,  w, subgroup_size_ratio = 0.66666, models_in_the_committee = 3,
                                    numbers_of_iteractions = 10):
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int)
    print( 'ECNA : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
    for it in range(numbers_of_iteractions):
        # Compute by bootstrap the values will be considered as true labels
        label_prediction = compute_bootstrap_models_predictions(    X_train, labels, X_test, w, subgroup_size_ratio = subgroup_size_ratio,
                                                                    models_in_the_committee = models_in_the_committee)
        # Use all the labelled dataset to compute the model output
        w = ridge_regression( labels, X_train, w)

        prediction_model = X_test.dot(w)

        # Compute active learning scores
        factors = (label_prediction - prediction_model.repeat(label_prediction.shape[1], axis = 1))
        scores = np.zeros([factors.shape[0],1])
        for i in range(factors.shape[1]):
            factor = factors[:,i].reshape(factors.shape[0],1).repeat(X_test.shape[1],axis = 1)
            scores += np.linalg.norm(factor*X_test, axis = 1).reshape(factors.shape[0],1)
        # Add the highest scores data and return
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices  = sample_best_scores( scores, X_train, labels, X_test, Y_test,
                                                                                        mol_indices, numbers_of_iteractions,
                                                                                        mode = 'highest')
    return indices, X_train, labels, X_test, Y_test, pos, w
@jit(nopython = True, parallel = True)
def D_optimality(matrix, eps):
    eig, _ = np.linalg.eigh(matrix)
    order_of_magnitude = np.sum(np.log2(eig + eps))
    #return 1/(np.linalg.det(matrix))  it works just adding an identity to the fisher matrix
    return -1*order_of_magnitude
#@jit(nopython = True, parallel = True)
def A_optimality(matrix, eps):
    eig, _ = np.linalg.eig(matrix)
    order_of_magnitude = np.log2(np.sum(np.abs(eig)))
    #return np.sum(np.ones_like(eig)/eig)
    return -1*order_of_magnitude

def E_optimality(matrix, eps):
    #eig, _ = linalg.eigs(matrix, k = 1, which = 'SM')
    #TODO: inverse power with numba
    eig, _ = np.linalg.eigh(matrix)
    return eig.max()

def D_approximated(matrix, eps):
    eig = np.diag(matrix)
    order_of_magnitude = np.sum(np.log2(np.abs(eig) + eps))
    return -1*order_of_magnitude

def Fisher_optimality_iterate(  X_train, labels, X_test, Y_test, mol_indices,  w, subgroup_size_ratio = 0.66666, models_in_the_committee = 3,
                                numbers_of_iteractions = 10, mode = 'Dapprox'):
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int)
    optimality_crit = { 'D' :   D_optimality,
                        'A' :   A_optimality,
                        'E' :   E_optimality,
                        'Dapprox' : D_approximated,

    }
    eps = np.finfo(np.float64).eps

    for it in range(numbers_of_iteractions):
        # Compute by bootstrap the values will be considered as true labels
        label_prediction = compute_bootstrap_models_predictions(    X_train, labels, X_test, w, subgroup_size_ratio = subgroup_size_ratio,
                                                                    models_in_the_committee = models_in_the_committee)
        print( 'F : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
        #Update the weights
        w = ridge_regression( labels, X_train, w)
        #Update the labelled set Fisher matrix after having added the best data at the previous iteraction
        prediction_model_train = X_train.dot(w)
        factors_train = (labels - prediction_model_train).repeat(X_train.shape[1],axis = 1)
        delta_x_matrix_train = factors_train*X_train
        delta_x_matrix_train_sum = np.sum(delta_x_matrix_train, axis = 0)
        print(labels.shape)
        print(prediction_model_train.shape)
        print('Check it is a column')
        bootstrap_label = np.mean(label_prediction, axis = 1)
        # Use all the labelled dataset to compute the model output
        prediction_model = X_test.dot(w)
        # Compute active learning scores inverting the Fisher information matrix for all the data
        factors = (bootstrap_label.reshape(bootstrap_label.shape[0],1) - prediction_model).repeat(X_test.shape[1],axis = 1)
        delta_x_matrix = factors*X_test
        scores = np.zeros([X_test.shape[0],1])
        for i,delta_x in enumerate(delta_x_matrix):
            delta_x = delta_x + delta_x_matrix_train_sum
            delta_x = delta_x.reshape(1,delta_x_matrix.shape[1])
            Fisher_matrix = delta_x.T.dot(delta_x) #+ np.eye(delta_x.shape[1])
            scores[i] = optimality_crit[mode](Fisher_matrix, eps)
        # Add the highest scores data and return
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices  = sample_best_scores( scores, X_train, labels, X_test, Y_test,
                                                                                        mol_indices, numbers_of_iteractions,
                                                                                        mode = 'lowest')
    return indices, X_train, labels, X_test, Y_test, pos, w

def Fisher_optimality_iterate_wrong(  X_train, labels, X_test, Y_test, mol_indices,  w, subgroup_size_ratio = 0.66666, models_in_the_committee = 3,
                                numbers_of_iteractions = 10, mode = 'Dapprox'):
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int)
    optimality_crit = { 'D' :   D_optimality,
                        'A' :   A_optimality,
                        'E' :   E_optimality,
                        'Dapprox' : D_approximated,

    }
    eps = np.finfo(np.float64).eps
    for it in range(numbers_of_iteractions):
        # Compute by bootstrap the values will be considered as true labels
        label_prediction = compute_bootstrap_models_predictions(    X_train, labels, X_test, w, subgroup_size_ratio = subgroup_size_ratio,
                                                                    models_in_the_committee = models_in_the_committee)
        print( 'F : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
        bootstrap_label = np.mean(label_prediction, axis = 1)
        # Use all the labelled dataset to compute the model output
        w = ridge_regression( labels, X_train, w)
        prediction_model = X_test.dot(w)
        # Compute active learning scores inverting the Fisher information matrix for all the data
        factors = (bootstrap_label.reshape(bootstrap_label.shape[0],1) - prediction_model).repeat(X_test.shape[1],axis = 1)
        delta_x_matrix = factors*X_test
        scores = np.zeros([X_test.shape[0],1])
        for i,delta_x in enumerate(delta_x_matrix):
            delta_x = delta_x.reshape(1,delta_x_matrix.shape[1])
            Fisher_matrix = delta_x.T.dot(delta_x) #+ np.eye(delta_x.shape[1])
            scores[i] = optimality_crit[mode](Fisher_matrix, eps)
        # Add the highest scores data and return
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices  = sample_best_scores( scores, X_train, labels, X_test, Y_test,
                                                                                        mol_indices, numbers_of_iteractions,
                                                                                        mode = 'lowest')
    return indices, X_train, labels, X_test, Y_test, pos, w

def calculate_mse(e):
    """Calculate the MSE for the error vector e."""
    e = np.clip(e, -FLOAT_MAX_SQRT, FLOAT_MAX_SQRT) #We want to avoid overflow errors
    return 0.5*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))

def compute_loss(X_test, weights, Y_test, method = 'ridge'):
    if (method == 'ridge'):
        if weights.shape[0] == 1:
            weights = weights.T
        individuals_error = Y_test - X_test.dot(weights)
        rmse = np.sqrt(calculate_mse(individuals_error))#/X_test.shape[0]
        mae = calculate_mae(individuals_error)#/X_test.shape[0]

    return mae, rmse

def sample_best_scores(scores, X_train, labels, X_test, Y_test, mol_indices, numbers_of_iteractions, mode):
    f = {   'lowest': np.argmin,
            'highest': np.argmax,}
    index_max_variance = f[mode](scores)
    index_max_variance = int (index_max_variance)
    selected_data = X_test[index_max_variance]
    indices = mol_indices[index_max_variance]
    pos = index_max_variance
    mol_indices = np.delete(mol_indices, index_max_variance, 0)
    X_train = np.vstack([X_train,selected_data])
    X_test = np.delete(X_test, index_max_variance, 0)
    labels = np.vstack([labels,Y_test[index_max_variance]])
    Y_test = np.delete(Y_test, index_max_variance, 0)
    #print( mode + ' score: ' + str(np.max(scores)))
    return indices, X_train, labels, X_test, Y_test, pos, mol_indices

def prepare_EGAL(X_train, X_test, by_chunks = False):
    if by_chunks:
        X_tot = np.vstack([X_train, X_test])
        step_size = 10
        mean = 0
        std_dev = 0
        for i in np.arange(0,X_tot.shape[0], step_size):
            sim = cosine_similarity_n_space(X_tot,X_tot[i:i+step_size])
            fp = np.memmap('../results/sim'+str(i), dtype = 'float32', mode = 'w+', shape = (X_tot.shape[0],step_size))
            fp[:] = sim[:]
            mean = mean + np.mean(sim)
            std_dev = std_dev + np.std(sim)
        std_dev = std_dev/X_tot.shape[0]*step_size
    else:
        sim = cosine_similarity(np.vstack([X_train, X_test]))
        mean = np.mean(sim)
        std_dev = np.std(sim)
    alpha = mean - 0.5*std_dev
    if by_chunks:
        densities = np.zeros([X_tot.shape[0],1])
        for i in np.arange(0,X_tot.shape[0], step_size):
            sim = np.memmap('../results/sim'+str(i), dtype='float32', mode='r', shape=(X_tot.shape[0],step_size))
            sim_thresholded = np.where(sim[:,:] >= alpha, sim[:,:], 0)
            new_densities = np.sum(sim_thresholded, axis = 1).reshape(sim[:,:].shape[0],1)
            densities = densities + new_densities
    else:
        sim_thresholded = np.where(sim >= alpha, sim, 0)
        densities = np.sum(sim_thresholded, axis = 0).reshape(sim.shape[0],1)
    return sim, densities
    
def EGAL_iterate(  X_train, labels, X_test, Y_test, mol_indices,  w, sim, densities, numbers_of_iteractions = 10, subgroup_size_ratio = 0.66666, models_in_the_committee = 3, by_chunks=True):
    #print( 'EGAL : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
    w = 0.25
    training_ind = np.arange(X_train.shape[0], dtype = int).reshape(X_train.shape[0],1)
    testing_ind = np.arange(X_train.shape[0], X_train.shape[0] + X_test.shape[0], dtype = int).reshape(X_test.shape[0],1)
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int) 
    print('EGAL: dimension unlabelled set '+ str(testing_ind.shape[0]))
    print('EGAL: dimension labelled set '+ str(training_ind.shape[0]))

    for it in range(numbers_of_iteractions):
        if by_chunks:
            X_tot = np.vstack([X_train, X_test])
            inv_diversities = np.zeros([X_test.shape[0],1])
            step_size = 10
            for i in np.arange(0,X_tot.shape[0], step_size):
                sim = np.memmap('../results/sim'+str(i), dtype='float32', mode='r', shape=(X_tot.shape[0],step_size))
                indices_ch = np.arange(i,i+step_size)
                training_ind_partial = np.intersect1d(training_ind.flatten(),indices_ch) - i
                sim_unlabelled = sim[np.ix_(testing_ind.flatten(),training_ind_partial)]
                if sim_unlabelled.shape[1] != 0:
                    inv_diversities_partial = np.max(sim_unlabelled, axis = 1)
                    inv_diversities = np.max(np.concatenate((inv_diversities.reshape(X_test.shape[0],1), inv_diversities_partial.reshape(X_test.shape[0],1)), axis = 1).reshape(X_test.shape[0],2),axis = 1)
        else:
            sim_unlabelled = sim[np.ix_(testing_ind.flatten(),training_ind.flatten())]
            inv_diversities = np.max(sim_unlabelled, axis = 1)
        copy_inv_diversities = np.copy(inv_diversities)
        numpyParallelSort(copy_inv_diversities)
        beta = copy_inv_diversities[int(np.floor(X_test.shape[0]*w))]
        candidate_indices = np.where(inv_diversities <= beta)
        #sort the candidates according to densities
        scores = np.zeros([X_test.shape[0],1])
        scores[candidate_indices] = densities[candidate_indices]
        # Sampling the highest density element inside the candidate dataset
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices = sample_best_scores( scores, X_train, labels, X_test, Y_test, 
                                                                                            mol_indices, numbers_of_iteractions,    
                                                                                            mode = 'highest')
        training_ind = np.vstack([training_ind,testing_ind[pos[it]]])
        testing_ind = np.delete(testing_ind, pos[it])
	
    w = ridge_regression( labels, X_train, w)                                                                                    
    return indices, X_train, labels, X_test, Y_test, pos, w
""" NOT CHUNK VERSION OF EGAL

def prepare_EGAL(X_train, X_test):
    sim = cosine_similarity(np.vstack([X_train, X_test]))
    mean = np.mean(sim)
    std_dev = np.std(sim)
    alpha = mean - 0.5*std_dev
    sim_thresholded = np.where(sim >= alpha, sim, 0)
    densities = np.sum(sim_thresholded, axis = 0).reshape(sim.shape[0],1)
    return sim, densities
    
def EGAL_iterate(  X_train, labels, X_test, Y_test, mol_indices,  w, sim, densities, numbers_of_iteractions = 10, subgroup_size_ratio = 0.66666, models_in_the_committee = 3):
    #print( 'EGAL : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
    w = 0.25
    training_ind = np.arange(X_train.shape[0], dtype = int).reshape(X_train.shape[0],1)
    testing_ind = np.arange(X_train.shape[0], X_train.shape[0] + X_test.shape[0], dtype = int).reshape(X_test.shape[0],1)
    indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int) 
    print('EGAL: dimension unlabelled set '+ str(training_ind.shape[0]))
    print('EGAL: dimension labelled set '+ str(testing_ind.shape[0]))
    for it in range(numbers_of_iteractions):
        #sim_unlabelled = sim[testing_ind.flatten(),:][:,training_ind.flatten()]
        sim_unlabelled = sim[np.ix_(testing_ind.flatten(),training_ind.flatten())]
        inv_diversities = np.max(sim_unlabelled, axis = 1)
        copy_inv_diversities = np.copy(inv_diversities)
        numpyParallelSort(copy_inv_diversities)
        beta = copy_inv_diversities[int(np.floor(X_test.shape[0]*w))]
        candidate_indices = np.where(inv_diversities <= beta)
        #sort the candidates according to densities
        scores = np.zeros([X_test.shape[0],1])
        scores[candidate_indices] = densities[candidate_indices]
        # Sampling the highest density element inside the candidate dataset
        indices[it], X_train, labels, X_test, Y_test, pos[it], mol_indices = sample_best_scores( scores, X_train, labels, X_test, Y_test, 
                                                                                            mol_indices, numbers_of_iteractions,    
                                                                                            mode = 'highest')
        training_ind = np.vstack([training_ind,testing_ind[pos[it]]])
        testing_ind = np.delete(testing_ind, pos[it])
	
    w = ridge_regression( labels, X_train, w)                                                                                    
    return indices, X_train, labels, X_test, Y_test, pos, w
"""

def cosine_similarity_n_space(m1, m2, batch_size=100):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break 
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) 
        ret[start: end] = sim
    return ret

def FPS_iterate2(  X_train, labels, X_test, Y_test, mol_indices,  w, FPS_indices, 
numbers_of_iteractions = 10, subgroup_size_ratio = 0.66666, models_in_the_committee = 3):
    #print( 'FPS : unlabelled test dimension ' + str(X_test.shape[0]) + '/' + str(X_test.shape[0] + X_train.shape[0]))
    #indices = np.zeros(numbers_of_iteractions, dtype = int)
    pos = np.zeros(numbers_of_iteractions, dtype = int) 
    X = np.vstack([X_train, X_test])
    all_labels = np.vstack([labels, Y_test])
    Nselect = X_train.shape[0] + numbers_of_iteractions
    X_train = X[FPS_indices[:Nselect]]
    labels = all_labels[FPS_indices[:Nselect]]
    Y_test = all_labels[FPS_indices[Nselect:]]
    X_test = X[FPS_indices[Nselect:]]
    w = ridge_regression( labels, X_train, w) 
    return np.ones_like(mol_indices), X_train, labels, X_test, Y_test, FPS_indices[:Nselect], w

def prepare_FPS (X_train, X_test):
    #TOASK which kernel???
    Nselect = 4000
    kernel = KernelPower(zeta = 1)
    #kernel = KernelSum(KernelPower(zeta = 1))
    X = np.vstack([X_train, X_test])
    compressor = FPSFilter(Nselect,kernel,act_on='sample',precompute_kernel=False,disable_pbar=False)
    compressor.fit(X,dry_run=True)
    indices = compressor.selected_ids
    
    return indices

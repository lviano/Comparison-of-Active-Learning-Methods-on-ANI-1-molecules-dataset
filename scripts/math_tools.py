import numpy as np
from ml_tools.descriptors import RawSoapInternal
from ml_tools.kernels import KernelPower,KernelSum

def compute_soap_matrix(frames, fast_avg = True):
    global_species = []
    for frame in frames:
        global_species.extend(frame.get_atomic_numbers())
    global_species = np.unique(global_species)
    soap_params = dict(rc=3.5, nmax=6, lmax=6, awidth=0.4,
                   global_species=global_species,nocenters=[], fast_avg = True)

    representation = RawSoapInternal(**soap_params)
    X = representation.transform(frames)
    print(X.shape)
    print('above size of soap matrix')
    return X
    

def compute_kernel(feature_matrix_1, fetaure_matrix_2, zeta=None, chunk_shape=None):
    # kernel = KernelSum(KernelPower(zeta = zeta),chunk_shape=[chunk_shape, chunk_shape])
    kernel = KernelPower(zeta = 1)
    Kmat = kernel.transform(feature_matrix_1, fetaure_matrix_2)
    return Kmat




import h5py,sys
from glob import glob
sys.path.insert(0,'../ANI-1_release/readers/lib/')
import numpy as np
from ase.io import read,write
from ase.visualize import view
import pyanitools as pya
import json

fns = glob('../ANI-1_release/ani*.h5')
frames_different_molecule = []
frames_different_molecule_test = []
frames100_conf_per_mol = []
total_molecules = 22057374
seed = 2020
    
Nstruct = 0

for it,fn in enumerate(fns):
	print(fn)

	adl = pya.anidataloader(fn)
	# Print the species of the data set one by one

	for in_data,data in enumerate(adl):
	    
		# Extract the data
		E = data['energies']
		
		mol_in_the_block = E.shape[0]
		shifts = np.random.RandomState(seed = seed).permutation(np.arange(mol_in_the_block)) 
		frames_different_molecule.append(Nstruct + shifts[0])
		frames_different_molecule_test.append(Nstruct + shifts[1])
		Nstruct += mol_in_the_block
	adl.cleanup()   
	print('Number of molecule in the dataset is ' + str(Nstruct))
frames = {'frames':frames_different_molecule,
		  'frames_test': frames_different_molecule_test,
			}
with open('../ANI-1_release/frames21.json', 'w') as fp:
                json.dump(frames, fp)

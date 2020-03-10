# Comparison-of-Active-Learning-Methods-on-ANI-1-molecules-dataset

The following Active Learning methods are implemented:

- Query by Commitee QBC
- Expected Model Change (ECNA and ECLA)
- Expected Variance Reduction
- Farthest Point Sampling
- Exploration Guided Active Learning

And are tested on the molecular dataset ANI-1 where the aim is to predict the ground state energy of the different structures.
For additional details look at the pdf report in the repository.

This work is part of my study plan at EPFL and it was carried out under the supervision of Michele Ceriotti and Felix Musil

## Prerequisities

QUIPPY (https://libatoms.github.io/QUIP/install.html)

Python 2 for the processing part, Python 3 for uploading the data

Dataset (Available Pool for Training Set): ANI-1 (https://github.com/isayev/ANI1_dataset)

Test Set: COMP-6 (https://github.com/isayev/COMP6)
## Usage 

Launch one of the script main to run the desired methods on different size of available pool. Insert in the list of methods the scripts corresponding to the active learning methods to be tested.

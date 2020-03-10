
all: 
	cython --cplus parallel_sort.pyx   
	g++  -g -march=native -Ofast -fpic -c    parallel_sort.cpp -o paraparallel_sortllelSort.o -fopenmp `python-config --includes`  -I`python -c "import numpy as np;print np.get_include()"`
	g++  -g -march=native -Ofast -shared  -o parallel_sort.so paraparallel_sortllelSort.o `python-config --libs` -lgomp  

clean:
	rm -f parallel_sort.cpp *.o *.so


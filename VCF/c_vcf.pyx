import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

cimport numpy as np
cimport libc.string
from numpy cimport ndarray
from libc.stdlib cimport malloc, free


# declare the interface to the C code
cdef extern void compute_c (double* data,  int samples, float T)

# static type annotations
cdef double *cdata
cdata = <double *>malloc(2048*cython.sizeof(double))

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_c_cython(np.ndarray[double, ndim=1, mode="c"] data not None, int FRAMES_PER_BUFFER, float T):

	cdef unsigned int i

	#move data from python np.arrray to C memory located 
	for i in xrange(len(data)):
		cdata[i] = data[i]
		


	#call the C function
	compute_c(cdata,FRAMES_PER_BUFFER, T)

	#read modified data into np.array
	for i in xrange(len(data)):
		data[i] = cdata[i]
	

	return data




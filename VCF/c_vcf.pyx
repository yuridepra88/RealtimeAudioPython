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



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_c1(np.ndarray[double, ndim=1, mode="c"] data not None, int FRAMES_PER_BUFFER, float T):

	cdef double *cdata
	cdef unsigned int i

	
	cdata = <double *>malloc(len(data)*cython.sizeof(double))
	if cdata is NULL:
		raise MemoryError()
	
	#move data from python np.arrray to C memory located 
	for i in xrange(len(data)):
		cdata[i] = data[i]
		


	#call the C function
	compute_c(cdata,FRAMES_PER_BUFFER, T)

	#read modified data into np.array
	for i in xrange(len(data)):
		data[i] = cdata[i]
	
	#low-pass filter applied to computed data
	#new_data, zi= signal.lfilter(b, a, data, zi=zi)
	
	# Analize results

	#plt.plot(np.arange(FRAMES_PER_BUFFER), data)
	#plt.show()

	#free the memory locked before 
	free(cdata)
	#free(mod_cdata)

	return data




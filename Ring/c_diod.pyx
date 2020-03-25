import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

cimport numpy as np
cimport libc.string
from numpy cimport ndarray
from libc.stdlib cimport malloc, free


# declare the interface to the C code
cdef extern void compute_c (double* data, double* modulator, int samples, float T)



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_c1(np.ndarray[double, ndim=1, mode="c"] data not None, np.ndarray[double, ndim=1, mode="c"] mod_data not None, int FRAMES_PER_BUFFER, float T, int over_f, np.ndarray[double, ndim=1, mode="c"] a, np.ndarray[double, ndim=1, mode="c"] b, np.ndarray[double, ndim=1, mode="c"] zi, np.ndarray[double, ndim=1, mode="c"] zi0, np.ndarray[double, ndim=1, mode="c"] zi1):

	cdef double *cdata
	cdef double *mod_cdata
	cdef unsigned int i

	#resample the two buffers to oversampling rate using 0-padding

	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	mod_data=np.insert(mod_data, indices, 0)

	data0, zi0= signal.lfilter(b, a, data, zi=zi0)
	data1, zi1= signal.lfilter(b, a, mod_data, zi=zi1)
	#transform np.arrays to pointer to C array and reserve memory
	cdata = <double *>malloc(len(data0)*cython.sizeof(double))
	if cdata is NULL:
		raise MemoryError()
	mod_cdata = <double *>malloc(len(data1)*cython.sizeof(double))
	if mod_cdata is NULL:
		raise MemoryError()
	
	#move data from python np.arrray to C memory located 
	for i in xrange(len(data)):
		cdata[i] = data[i]
		mod_cdata[i] = mod_data[i]


	#call the C function
	compute_c(cdata,mod_cdata,FRAMES_PER_BUFFER*over_f, T)

	#read modified data into np.array
	for i in xrange(len(data)):
		data[i] = cdata[i]
	
	#low-pass filter applied to computed data
	new_data, zi= signal.lfilter(b, a, data, zi=zi)
	

	#free the memory locked before 
	free(cdata)
	free(mod_cdata)

	return new_data[::over_f],zi, zi0,zi1 #take one sample each over_f samples




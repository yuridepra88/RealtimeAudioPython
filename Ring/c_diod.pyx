import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

cimport numpy as np
cimport libc.string
from numpy cimport ndarray
from libc.stdlib cimport malloc


# declare the interface to the external C code
cdef extern void compute_c (double* data, double* modulator, int samples, float T)

# static type annotations
cdef double *cdata
cdef double *mod_cdata


cdata = <double *>malloc(16384*cython.sizeof(double))
mod_cdata = <double *>malloc(16384*cython.sizeof(double))

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cython_c(np.ndarray[double, ndim=1, mode="c"] data not None, np.ndarray[double, ndim=1, mode="c"] mod_data not None, int FRAMES_PER_BUFFER, float T, int over_f, np.ndarray[double, ndim=1, mode="c"] a, np.ndarray[double, ndim=1, mode="c"] b, np.ndarray[double, ndim=1, mode="c"] zi):

	cdef unsigned int i
	
	#resample the two buffers to oversampling rate using 0-padding
	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	mod_data=np.insert(mod_data, indices, 0)


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
	
	
	return new_data[::over_f],zi #take one sample each over_f samples




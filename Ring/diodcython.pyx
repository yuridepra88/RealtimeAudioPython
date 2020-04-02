import cython
import numpy as np
cimport numpy as np
cimport libc.string
from numpy cimport ndarray
import line_profiler
from scipy import signal

#tecnical electical data
cdef double C   = 10e-9  #F
cdef double C_p = 10e-9  #F
cdef float L   = 0.8     #H
cdef int R_a = 600      #ohm
cdef int R_i = 50       #ohm
cdef int R_m = 80       #ohm

cdef double u1 = 0.
cdef double u2 = 0.
cdef double u3 = 0.
cdef double u4 = 0.
cdef double u5 = 0.
cdef double u6 = 0.
cdef double u7 = 0.

cdef double i1 = 0.
cdef double i2 = 0.
cdef double k  = 0.8  #amplitude reduction factor


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_c(np.ndarray[double, ndim=1, mode="c"] data not None, np.ndarray[double, ndim=1, mode="c"] mod_signal not None, int FRAMES_PER_BUFFER, float T, int over_f,  np.ndarray[double, ndim=1, mode="c"] a not None, np.ndarray[double, ndim=1, mode="c"] b not None, np.ndarray[double, ndim=1, mode="c"] zi not None):

	global u1,u2,u3,u4,u5,u6,u7,i1,i2

	#resample the two buffers to oversampling rate using 0-padding
	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	mod_signal=np.insert(mod_signal, indices, 0)

	samples = FRAMES_PER_BUFFER*over_f
	for i in range (0,samples):
		u1 += T/C*(i1 - (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. + (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - (u1-mod_signal[i]*k)/R_m )
		u2 += T/C*(i2 + (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. - (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - u2/R_a )
		u3 += T/C_p*((u4*u4*u4*u4*0.17) - (u7*u7*u7*u7*0.17) + (u5*u5*u5*u5*0.17) - (u6*u6*u6*u6*0.17)- (u3)/R_i )
	    

		u4 =  u1/2. - u3 - data[i]*k - u2/2.
		u5 = -u1/2. - u3 - data[i]*k+ u2/2.
		u6 =  u1/2. + u3 + data[i]*k + u2/2.
		u7 = -u1/2. + u3 + data[i]*k- u2/2.

		i1 = i1+ T/L*(-u1)
		i2 = i2+ T/L*(-u2)
		data[i] = u2

	#low-pass filter applied to computed data
	new_data, zi= signal.lfilter(b, a, data, zi=zi)

	return new_data[::over_f],zi #take one sample each over_f samples


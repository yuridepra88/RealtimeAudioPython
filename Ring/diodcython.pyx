import cython
import numpy as np
cimport numpy as np
cimport libc.string
from numpy cimport ndarray
import line_profiler


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



cdef np.ndarray[double, ndim=1, mode="c"] compute_c1(np.ndarray[double, ndim=1, mode="c"] data, int samples, float T, int over_f):

	global u1,u2,u3,u4,u5,u6,u7,i1,i2,C,C_p,L,R_a,R_m,R_i


	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	
	mod_signal = data.copy()
	
	for i in range (samples):
		u1 += T/C*(i1 - (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. + (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - (u1-mod_signal[i])/R_m )
		u2 += T/C*(i2 + (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. - (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - u2/R_a )
		u3 += T/C_p*((u4*u4*u4*u4*0.17) - (u7*u7*u7*u7*0.17) + (u5*u5*u5*u5*0.17) - (u6*u6*u6*u6*0.17)- (u3)/R_i )
	    

		u4 =  u1/2. - u3 - data[i]*0.1 - u2/2.
		u5 = -u1/2. - u3 - data[i]*0.1 + u2/2.
		u6 =  u1/2. + u3 + data[i]*0.1 + u2/2.
		u7 = -u1/2. + u3 + data[i]*0.1 - u2/2.

		i1 = i1+ T/L*(-u1)
		i2 = i2+ T/L*(-u2)
		data[i] = u2
	   
		    
	return data

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_c(np.ndarray[double, ndim=1, mode="c"] data not None, int FRAMES_PER_BUFFER, float T, int over_f):
	return compute_c1(data,FRAMES_PER_BUFFER*over_f, T, over_f)




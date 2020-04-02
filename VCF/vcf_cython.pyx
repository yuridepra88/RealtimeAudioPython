import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

cimport numpy as np
cimport libc.string
from numpy cimport ndarray
import math




cdef double gfdbk = 5.175 #1.5
cdef double I0 = 30e-6
cdef double eta = 1.836
cdef double Vt = 0.026
cdef double c = 0.01e-6
cdef double REL_ERR = 10e-4



cdef double s1 = 0
cdef double s2 = 0
cdef double s3 = 0
cdef double s4 = 0


cdef double vout = 0

cdef double vc2 = 0
cdef double vc3 = 0
cdef double vc4 = 0

cdef double vc11 = 0
cdef double vc21 = 0
cdef double vc31 = 0
cdef double vc41 = 0


cdef double xc1 = 0
cdef double xc2 = 0
cdef double xc3 = 0
cdef double xc4 = 0



def compute_cython(np.ndarray[double, ndim=1, mode="c"] data not None, int samples, float T):

	global s1,s2,s3,s4
	global vout, vc2,vc3, vc4, vc11, vc21,vc31,vc41
	
	cdef double vin1 = 0
	cdef double vc1  = 0
	cdef double gmma = eta*Vt
	
	for i in range(0,samples):

		vc4 = 0.
		vc4Past = 1.
		nIter = 0
   
		while( abs(vc4-vc4Past) > REL_ERR*abs(vc4Past) ):    

			vc4Past = vc4
			vin1 = math.tanh((data[i] - vout)/(2.*Vt))

			xc1 = (I0/2./c) * (vin1 + vc11)
			vc1 = T/2.*xc1 + s1
			vc11 = math.tanh((vc2-vc1)/(2.*gmma))
       
			xc2 = (I0/2./c) * (vc21 - vc11)
			vc2 = T/2.*xc2 + s2
			vc21 = math.tanh((vc3-vc2)/(2.*gmma))

			xc3 = (I0/2./c) * (vc31 - vc21)
			vc3 = T/2.*xc3 + s3
			vc31 = math.tanh((vc4-vc3)/(2.*gmma))

			xc4 = (I0/2./c) * (-vc41 - vc31)
			vc4 = T/2.*xc4 + s4
			vc41 = math.tanh(vc4/(6.*gmma))

			vout = vc4/2. + vc4*gfdbk
       
			nIter +=1
			if (nIter == 100):
				nIter=0          			
				break

		data[i] = vout

		s1 = T/2.*xc1 + vc1
		s2 = T/2.*xc2 + vc2
		s3 = T/2.*xc3 + vc3
		s4 = T/2.*xc4 + vc4
		
	return  data




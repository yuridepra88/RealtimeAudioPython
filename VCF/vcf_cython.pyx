import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

cimport numpy as np
cimport libc.string
from numpy cimport ndarray
from libc.stdlib cimport malloc, free
import math



cdef double gfdbk = 1.5
cdef float I0 = 30e-6
cdef float eta = 1.836
cdef float Vt = 0.026
cdef float c = 0.01e-6
cdef float REL_ERR = 10e-4


cdef float xc1 = 0
cdef float xc2 = 0
cdef float xc3 = 0
cdef float xc4 = 0
cdef float vc1 = 0
cdef float vc2 = 0
cdef float vc3 = 0
cdef float vc4 = 0
cdef float vc11 = 0
cdef float vc21 = 0
cdef float vc31 = 0
cdef float vc41 = 0


cdef float s1 = 0
cdef float s2 = 0
cdef float s3 = 0
cdef float s4 = 0

cdef float vin1Temp = 0
cdef float voutTemp = 0
cdef float vc1Temp = 0
cdef float vc2Temp = 0
cdef float vc3Temp = 0
cdef float vc4Temp = 0

cdef float vc11Temp = 0
cdef float vc21Temp = 0
cdef float vc31Temp = 0
cdef float vc41Temp = 0


cdef float xc1Temp = 0
cdef float xc2Temp = 0
cdef float xc3Temp = 0
cdef float xc4Temp = 0



def compute_cython(np.ndarray[double, ndim=1, mode="c"] data not None, int samples, float T):
	global I0, eta, Vt, c, REL_ERR, xc1, xc2, xc3, xc4, vc1,vc2,vc3,vc4, vc11,vc21,vc31,vc41, s1,s2,s3,s4
	global vin1Temp, voutTemp, vc1Temp, vc2Temp,vc3Temp, vc4Temp, vc11Temp, vc21Temp,vc31Temp,vc41Temp
	global xc1Temp, xc2Temp, xc3Temp, xc4Temp

	cdef float gmma = eta*Vt
	
	for i in range(0,samples):

		vc4Temp = 0
		vc4Past = 1
		nIter = 0
   
		while( abs(vc4Temp-vc4Past) > REL_ERR*abs(vc4Past) ):    

			vc4Past = vc4Temp
			vin1Temp = math.tanh((data[i] - voutTemp)/(2*Vt))

			xc1Temp = (I0/2/c) * (vin1Temp + vc11Temp)
			vc1Temp = T/2*xc1Temp + s1
			vc11Temp = math.tanh((vc2Temp-vc1Temp)/(2*gmma))
       
			xc2Temp = (I0/2/c) * (vc21Temp - vc11Temp)
			vc2Temp = T/2*xc2Temp + s2
			vc21Temp = math.tanh((vc3Temp-vc2Temp)/(2*gmma))

			xc3Temp = (I0/2/c) * (vc31Temp - vc21Temp)
			vc3Temp = T/2*xc3Temp + s3
			vc31Temp = math.tanh((vc4Temp-vc3Temp)/(2*gmma))

			xc4Temp = (I0/2/c) * (-vc41Temp - vc31Temp)
			vc4Temp = T/2*xc4Temp + s4
			vc41Temp = math.tanh(vc4Temp/(6*gmma))

			voutTemp = vc4Temp/2 + vc4Temp*gfdbk
       
			nIter +=1
			if (nIter == 100):
				nIter=0          			
				break

		xc1 = xc1Temp
		xc2 = xc2Temp
		xc3 = xc3Temp
		xc4 = xc4Temp
		vc1 = vc1Temp
		vc2 = vc2Temp
		vc3 = vc3Temp
		vc4 = vc4Temp
		vc11 = vc11Temp
		vc21 = vc21Temp
		vc31 = vc31Temp
		vc41 = vc41Temp  
		data[i] = voutTemp


   
		s1 = T/2*xc1 + vc1
		s2 = T/2*xc2 + vc2
		s3 = T/2*xc3 + vc3
		s4 = T/2*xc4 + vc4



	
	return data




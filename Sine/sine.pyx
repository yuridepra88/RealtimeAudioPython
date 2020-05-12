import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc
import math

cdef class OscSine:

	cdef int chunksize;
	cdef float T;
	cdef float freq;
	cdef float amp; 
	cdef double *out;

	def __init__(self, pfreq, pamp, FRAMES_PER_BUFFER, t):
		self.chunksize = FRAMES_PER_BUFFER
		self.T = t	
		self.freq = pfreq
		self.amp = pamp
		self.out = <double *>malloc(self.chunksize*cython.sizeof(double))
		self.out[0] = 0.
		self.out[FRAMES_PER_BUFFER-2] =  -self.amp*math.sin(2*math.pi*self.freq*self.T)

	
	def gen_sine(self):
		w= 2*math.pi*self.freq*self.T
		self.out[0]= 2 * math.cos(w) * self.out[self.chunksize-1] - self.out[self.chunksize-2]
		self.out[1]= 2 * math.cos(w) * self.out[0] - self.out[self.chunksize-1]
		for i in range (2, self.chunksize):
			self.out[i] = 2 * math.cos(w) * self.out[i-1] - self.out[i-2]

			
	def get_samples(self):
		self.gen_sine()
		y=np.zeros(self.chunksize)
		for i in range(0,self.chunksize):
			y[i] = self.out[i]
		return y

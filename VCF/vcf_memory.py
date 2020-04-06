########
#
# Test the memory use of the algorithm
# on the console, after memory_profiler is installed (pip install memory_profiler) run
# >> mprof run -T 0.1 test_memory.py
# >> mprof plot -f
#
###############

import pyaudio
import numpy as np
import wave
import time
import sys

from scipy import signal    
import line_profiler
import matplotlib.pyplot as plt
import scipy

import math

import vcf
import vcf_cython
from audio_tools import *
from numba import jit, float64, double, types


FRAMES_PER_BUFFER =512
sample_rate = 44100
T = 1. / (sample_rate)

# VCS3 parameters
gfdbk = 5.175 
I0 = 30e-6
eta = 1.836
Vt = 0.026
c = 0.01e-6
REL_ERR = 10e-4



s1 = 0
s2 = 0
s3 = 0
s4 = 0

vin1 = 0
vout = 0
vc1 = 0
vc2 = 0
vc3 = 0
vc4 = 0

vc11 = 0
vc21 = 0
vc31 = 0
vc41 = 0

xc1 = 0
xc2 = 0
xc3 = 0
xc4 = 0


@profile
def measure_wrapper(data):

	return compute(data)
	
def compute(data):
	
	global s1,s2,s3,s4, vout, vc2,vc3, vc4, vc11, vc21,vc31,vc41
	
	vin1 = 0
	gmma = eta*Vt
	
	for i in range(0,FRAMES_PER_BUFFER):

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
		
	return data

@jit(types.Tuple((float64[:],double,double,double,double,double,double,double,double,double,double,double,double)) (float64[:],double,double,double,double,double,double,double,double,double,double,double,double),nopython=True)
def compute_numba_low(data,s1,s2,s3,s4,vout,vc2,vc3, vc4, vc11, vc21,vc31,vc41):
	
	vin1 = 0
	gmma = eta*Vt
	
	for i in range(0,FRAMES_PER_BUFFER):

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
		
	return data,s1,s2,s3,s4,vout,vc2,vc3, vc4, vc11, vc21,vc31,vc41

@profile
def compute_numba(data):
	global s1,s2,s3,s4,vout, vc2,vc3, vc4, vc11, vc21,vc31,vc41
	
	data,s1,s2,s3,s4,vout,vc2,vc3, vc4, vc11, vc21,vc31,vc41 = compute_numba_low(data,s1,s2,s3,s4,vout, vc2,vc3, vc4, vc11, vc21,vc31,vc41)
	
	return data


# cython with C bindings approach
@profile
def compute_cython_c(data):

	data =  vcf.compute_c_cython(data, FRAMES_PER_BUFFER ,T)
	return data

# cython annotation approach
@profile
def compute_cython(data):

	data =  vcf_cython.compute_cython(data, FRAMES_PER_BUFFER ,T)
	return data

wf = wave.open('./rock_1.wav', 'rb')


p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)




#iterate the implementations over 5 chuncks with 0.2 seconds of pause 
for i in range (0,5): # iterate over the chunks
	
	data = wf.readframes(FRAMES_PER_BUFFER)
	sample = pcm_to_double(data,FRAMES_PER_BUFFER)
	
	compute_numba(sample)
	time.sleep(0.2)
	compute_cython(sample)
	time.sleep(0.2)
	compute_cython_c(sample)
	time.sleep(0.2)
	measure_wrapper(sample)
	time.sleep(0.4)

	

stream.stop_stream()
stream.close()

p.terminate()


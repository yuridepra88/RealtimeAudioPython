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

FRAMES_PER_BUFFER =2048
format_unpack = "%dh"%(FRAMES_PER_BUFFER) 
SHORT_NORMALIZE = (1.0/32768.0)

sample_rate = 44100
T = 1. / (sample_rate)


gfdbk = 0.4
I0 = 30e-6
eta = 1.836
Vt = 0.026
c = 0.01e-6
REL_ERR = 10e-4


xc1 = 0
xc2 = 0
xc3 = 0
xc4 = 0
vc1 = 0
vc2 = 0
vc3 = 0
vc4 = 0
vc11 = 0
vc21 = 0
vc31 = 0
vc41 = 0


s1 = 0
s2 = 0
s3 = 0
s4 = 0

vin1Temp = 0
voutTemp = 0
vc1Temp = 0
vc2Temp = 0
vc3Temp = 0
vc4Temp = 0

vc11Temp = 0
vc21Temp = 0
vc31Temp = 0
vc41Temp = 0

xc1Temp = 0
xc2Temp = 0
xc3Temp = 0
xc4Temp = 0




def pcm_to_double(block):
    global SHORT_NORMALIZE, format_unpack    
    shorts = np.frombuffer(block, format_unpack) 
    doubles = np.squeeze(SHORT_NORMALIZE*np.array(shorts, dtype=np.float64)) 
    
    return doubles

def double_to_pcm(d):
    s = 32768*np.array(d)
    sample = s.astype(np.short).tostring()
    return sample		

# interpreted python approach
def compute(data):
	global I0, eta, Vt, c, REL_ERR, xc1, xc2, xc3, xc4, vc1,vc2,vc3,vc4, vc11,vc21,vc31,vc41, s1,s2,s3,s4
	global vin1Temp, voutTemp, vc1Temp, vc2Temp,vc3Temp, vc4Temp, vc11Temp, vc21Temp,vc31Temp,vc41Temp
	global xc1Temp, xc2Temp, xc3Temp, xc4Temp
	gmma = eta*Vt
	
	for i in range(0,FRAMES_PER_BUFFER):

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

# cython with C bindings approach
def compute_cython_c(data):

	data =  vcf.compute_c1(data, FRAMES_PER_BUFFER ,T)
	return data

# cython annotation approach
def compute_cython(data):

	data =  vcf_cython.compute_cython(data, FRAMES_PER_BUFFER ,T)
	return data

wf = wave.open('./test_saudio.wav', 'rb')


p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)


l = line_profiler.LineProfiler()
l.add_function(compute_cython_c)
l.add_function(compute_cython)
l.add_function(compute)


data = wf.readframes(FRAMES_PER_BUFFER)
i=0
# run over 1.02 seconds of audio (22 chunks of 2048 samples)
while i <23:
	i +=1
	sample = pcm_to_double(data)

	
	#y = compute(sample)
	l.run('compute_cython_c(sample)')
	l.run('compute_cython(sample)')
	l.run('compute(sample)')	
	data = wf.readframes(FRAMES_PER_BUFFER)


l.print_stats()

stream.stop_stream()
stream.close()

p.terminate()


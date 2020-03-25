########
#
# Test the performance of the 3 implentations: interpreted approach, cython annotation, cython with C bindings
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

import c_diod
import diod_cython as diod



FRAMES_PER_BUFFER =256
format_unpack = "%dh"%(FRAMES_PER_BUFFER) 
SHORT_NORMALIZE = (1.0/32768.0)

#define sample rate, ovesampling factor and sample interval
sample_rate = 44100
over_f =25 #oversampling factor
T = 1. / (sample_rate*over_f)



   
#tecnical electical data
C   = 10e-9  #F
C_p = 10e-9  #F
L   = 0.8    #H
R_a = 600.   #ohm
R_i = 50.    #ohm
R_m = 80.    #ohm

u1 = 0.
u2 = 0.
u3 = 0.
u4 = 0.
u5 = 0.
u6 = 0.
u7 = 0.

i1 = 0.
i2 = 0.


def g(n):
   return  0.17*(n**4)

def compute(data):
	global u1,u2,u3,u4,u5,u6,u7,i1,i2,C,C_p,L,R_a,R_m,R_i, T

	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	

	y = np.arange(FRAMES_PER_BUFFER)
	mod_signal = data.copy()

	for i in range (FRAMES_PER_BUFFER*over_f): 
		    

	    u1 += T/C*(i1 - (u4**2**2*0.17)/2. + (u7**2**2*0.17)/2. + (u5**2**2*0.17)/2. - (u6**2**2*0.17)/2. - (u1-mod_signal[i]*0.1)/R_m )  
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
	

def pcm_to_double(block):
    global SHORT_NORMALIZE, format_unpack    
    shorts = np.frombuffer(block, format_unpack) 
    doubles = np.squeeze(SHORT_NORMALIZE*np.array(shorts, dtype=np.float64)) 
    
    return doubles

def double_to_pcm(d):
    s = 32768*np.array(d)
    sample = s.astype(np.short).tostring()
    return sample		


def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def compute_cython(data):
	return diod.compute_c(data, FRAMES_PER_BUFFER ,T, over_f)


def compute_cython_c(data,mod_data):
	global zi,zi0,zi1
	data, zi,zi0,zi1 =  c_diod.compute_c1(data, mod_data, FRAMES_PER_BUFFER ,T, over_f,a,b,zi,zi0,zi1)
	return data

wf = wave.open('./test_sine.wav', 'rb')
wf_mod = wave.open('./test_sine_mod.wav', 'rb')


p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)


stream_mod = p.open(format=p.get_format_from_width(wf_mod.getsampwidth()),
                channels=wf_mod.getnchannels(),
                rate=wf_mod.getframerate(),
                output=True)

b, a = butter_lowpass(sample_rate/over_f, sample_rate, 3)
zi = signal.lfilter_zi(b,a)
zi0=zi
zi1=zi

# read modulation and carrier data
data = wf.readframes(FRAMES_PER_BUFFER)
mod_data = wf_mod.readframes(FRAMES_PER_BUFFER)



sample = pcm_to_double(data)
mod_sample = pcm_to_double(mod_data)


l = line_profiler.LineProfiler()
l.add_function(compute_cython)
l.add_function(compute_cython_c)
l.add_function(compute)
l.run('compute(sample)')
l.run('compute_cython(sample)')
l.run('compute_cython_c(sample, mod_sample)')
l.print_stats()


stream.stop_stream()
stream.close()

p.terminate()


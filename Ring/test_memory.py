########
#
# Test the memory use of the algorithm
# on the console, after memory_profiler is installed (pip install memory_profiler) run
# >> mprof run -T 0.1 test_memory.py
# >> mprof plot -f
#
###############

from numba import jit, float64, double, types
import pyaudio
import numpy as np
import wave

from scipy import signal    
import line_profiler

import c_diod
import diod_cython as diod
from audio_tools import *
import time


#define buffer size, sample rate, ovesampling factor and sample interval
FRAMES_PER_BUFFER =512
sample_rate = 44100
over_f =30 #oversampling factor
T = 1. / (sample_rate*over_f)

   
#tecnical electical data
C   = 10e-9  #F
C_p = 10e-9  #F
L   = 0.8    #H
R_a = 600.   #ohm
R_i = 50.    #ohm
R_m = 80.    #ohm

#init algorithm variables
u1 = 0.
u2 = 0.
u3 = 0.
u4 = 0.
u5 = 0.
u6 = 0.
u7 = 0.

i1 = 0.
i2 = 0.

#signal reduction factor
k = 0.4


#define the 4 different implementations

# Python interpreter implemtantion
@profile
def compute(data,mod_signal):
	global u1,u2,u3,u4,u5,u6,u7,i1,i2,zi

	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	mod_signal=np.insert(mod_signal, indices, 0)
	

	for i in range (FRAMES_PER_BUFFER*over_f): 
	    u1 += T/C*(i1 - (u4**2**2*0.17)/2. + (u7**2**2*0.17)/2. + (u5**2**2*0.17)/2. - (u6**2**2*0.17)/2. - (u1-mod_signal[i]*k)/R_m )  
	    u2 += T/C*(i2 + (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. - (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - u2/R_a )
	    u3 += T/C_p*((u4*u4*u4*u4*0.17) - (u7*u7*u7*u7*0.17) + (u5*u5*u5*u5*0.17) - (u6*u6*u6*u6*0.17)- (u3)/R_i )
	    
	    u4 =  u1/2. - u3 - data[i]*k - u2/2.
	    u5 = -u1/2. - u3 - data[i]*k + u2/2.
	    u6 =  u1/2. + u3 + data[i]*k + u2/2.
	    u7 = -u1/2. + u3 + data[i]*k - u2/2.

	    i1 = i1+ T/L*(-u1)
	    i2 = i2+ T/L*(-u2)
	    data[i] = u2
		
	new_data, zi= signal.lfilter(b, a, data, zi=zi)
	return new_data[::over_f]
	


#numba implementation
@jit(types.Tuple((float64[:],double,double,double,double,double,double,double,double,double)) (float64[:],float64[:],double,double,double,double,double,double,double,double,double),nopython=True)

def compute_numba_low(data, mod_signal, u1,u2,u3,u4,u5,u6,u7,i1,i2):
	
	for i in range (FRAMES_PER_BUFFER*over_f): 
	    u1 += T/C*(i1 - (u4**2**2*0.17)/2. + (u7**2**2*0.17)/2. + (u5**2**2*0.17)/2. - (u6**2**2*0.17)/2. - (u1-mod_signal[i]*k)/R_m )  
	    u2 += T/C*(i2 + (u4*u4*u4*u4*0.17)/2. + (u7*u7*u7*u7*0.17)/2. - (u5*u5*u5*u5*0.17)/2. - (u6*u6*u6*u6*0.17)/2. - u2/R_a )
	    u3 += T/C_p*((u4*u4*u4*u4*0.17) - (u7*u7*u7*u7*0.17) + (u5*u5*u5*u5*0.17) - (u6*u6*u6*u6*0.17)- (u3)/R_i )
	    
	    u4 =  u1/2. - u3 - data[i]*k - u2/2.
	    u5 = -u1/2. - u3 - data[i]*k + u2/2.
	    u6 =  u1/2. + u3 + data[i]*k + u2/2.
	    u7 = -u1/2. + u3 + data[i]*k - u2/2.

	    i1 = i1+ T/L*(-u1)
	    i2 = i2+ T/L*(-u2)
	    data[i] = u2

	return data,u1,u2,u3,u4,u5,u6,u7,i1,i2

#numba wrapper
@profile
def compute_numba(data, mod_data):
	global u1,u2,u3,u4,u5,u6,u7,i1,i2,zi

	indices = np.repeat(np.arange(data.shape[0]+2), (over_f-1))[1*(over_f-1):-1*(over_f-1)]      
	data=np.insert(data, indices, 0)
	mod_signal=np.insert(mod_data, indices, 0)
	data,u1,u2,u3,u4,u5,u6,u7,i1,i2 = compute_numba_low(data,mod_signal, u1,u2,u3,u4,u5,u6,u7,i1,i2)
	new_data, zi= signal.lfilter(b, a, data, zi=zi)
	return new_data[::over_f]
	
# call annotated cython implementation from diod module
@profile
def compute_cython(data, mod_data):
	global zi
	data, zi= diod.compute_c(data, mod_data, FRAMES_PER_BUFFER ,T, over_f,a,b,zi)
	return data

# call cython + C implementation from c_diod module
@profile
def compute_cython_c(data,mod_data):
	global zi
	data, zi =  c_diod.compute_cython_c(data, mod_data, FRAMES_PER_BUFFER ,T, over_f,a,b,zi)
	return data


#open two wave file to test
wf = wave.open('./test_sine.wav', 'rb')
wf_mod = wave.open('./test_sine_mod.wav', 'rb')

#init pyAudio
p = pyaudio.PyAudio()

#open the two audio streams
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)


stream_mod = p.open(format=p.get_format_from_width(wf_mod.getsampwidth()),
                channels=wf_mod.getnchannels(),
                rate=wf_mod.getframerate(),
                output=True)


# define output signal LPF 
b, a = butter_lowpass(sample_rate/over_f, sample_rate, 3)
zi = signal.lfilter_zi(b,a)

# read modulation and carrier data
data = wf.readframes(FRAMES_PER_BUFFER)
mod_data = wf_mod.readframes(FRAMES_PER_BUFFER)

sample = pcm_to_double(data,FRAMES_PER_BUFFER)
mod_sample = pcm_to_double(mod_data,FRAMES_PER_BUFFER)



#iterate the implementations over 5 chuncks with 0.2 seconds of pause 
for i in range (0,5): # iterate over the chunks


	compute_numba(sample, mod_sample)
	time.sleep(0.2)
	compute_cython(sample, mod_sample)
	time.sleep(0.2)
	compute_cython_c(sample, mod_sample)
	time.sleep(0.2)
	compute(sample, mod_sample)
	time.sleep(0.4)
		
	# read modulation and carrier data
	data = wf.readframes(FRAMES_PER_BUFFER)
	mod_data = wf_mod.readframes(FRAMES_PER_BUFFER)

	sample = pcm_to_double(data,FRAMES_PER_BUFFER)
	mod_sample = pcm_to_double(mod_data,FRAMES_PER_BUFFER)
	

stream.stop_stream()
stream.close()

p.terminate()


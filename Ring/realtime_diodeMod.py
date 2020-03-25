################
#
# Real-time implementation of the Diode Ring Modulator
#
########################
import pyaudio
import numpy as np
import wave
import time
import sys
from scipy import signal    
import line_profiler
import matplotlib.pyplot as plt

import c_diod


FRAMES_PER_BUFFER =128
format_unpack = "%dh"%(FRAMES_PER_BUFFER) 
SHORT_NORMALIZE = (1.0/32768.0)

#define sample rate, ovesampling factor and sample interval
sample_rate = 44100
over_f =30 
T = 1. / (sample_rate*over_f)



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


def compute_cython_c(data,mod_data):
	global zi, zi0, zi1
	data, zi, zi0,zi1 =  c_diod.compute_c1(data, mod_data, FRAMES_PER_BUFFER ,T, over_f,a,b,zi,zi0,zi1)
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

b, a = butter_lowpass(sample_rate/over_f, sample_rate, 5)
zi = signal.lfilter_zi(b,a)
zi0 = zi
zi1=zi

# read modulation and carrier data
data = wf.readframes(FRAMES_PER_BUFFER)
mod_data = wf_mod.readframes(FRAMES_PER_BUFFER)


while len(data) > 0:
	sample = pcm_to_double(data)
	mod_sample = pcm_to_double(mod_data)
	y = compute_cython_c(sample, mod_sample)
	y*=4
	enc = double_to_pcm(y)
	stream.write(enc)
	data = wf.readframes(FRAMES_PER_BUFFER)
	mod_data = wf_mod.readframes(FRAMES_PER_BUFFER)


stream.stop_stream()
stream.close()

p.terminate()


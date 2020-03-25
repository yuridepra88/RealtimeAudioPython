"""PyAudio realtime low-pass filter

	y[i]= coeff*x[i] + (1-coeff)*y[i-1]

	python low_filter_raw.py --input=./rock.wav

"""

import pyaudio
import wave
import time
import sys 
import argparse
from scipy import signal
import scipy
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


FRAMES_PER_BUFFER = 512
format_unpack = "%dh"%(FRAMES_PER_BUFFER*2)
SHORT_NORMALIZE = (1.0/32768.0)

INIT_FREQ = 0.5
last_sample= 0.



def update(val):
    global  coeff
    freq = sfreq.val

    coeff = freq
    fig.canvas.draw_idle()



def process(data):
	global last_sample, coeff
	
	_coeff= 1-coeff 
	n = FRAMES_PER_BUFFER*2

	y = np.arange(n, dtype=float)
	
	y[0]= coeff*data[0] + _coeff*last_sample
	for i in range(1,n):
		y[i]= coeff*data[i] + _coeff*y[i-1]
	last_sample=y[i]
	return  y


def pcm_to_double(block):
    global SHORT_NORMALIZE, format_unpack    
    shorts = np.frombuffer(block, format_unpack) 
    doubles = np.squeeze(SHORT_NORMALIZE*np.array(shorts))    
    return doubles

def double_to_pcm(d):
    s = 32768*np.array(d)
    sample = s.astype(np.short).tostring()
    return sample			


def callback(in_data, frame_count, time_info, flag):

    global l
    data = wf.readframes(frame_count)
    
    if flag:
        print("Error: %i" % flag)

    sample = pcm_to_double(data)
    y = process(sample)
    y_fft = scipy.fft(y) 
    l.set_ydata(2.0/frame_count * np.abs(y_fft[0:int(frame_count/2)]))
    pcm = double_to_pcm(y)
   
    return (pcm, pyaudio.paContinue)




if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help = "input file")
args = vars(ap.parse_args())	

wf = wave.open(args["input"], 'rb')
p = pyaudio.PyAudio()
fs = wf.getframerate()  # sample rate, Hz

coeff = INIT_FREQ

# change parameters while playing
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

T = 1.0 / fs
N = FRAMES_PER_BUFFER
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))	

y_fft = np.zeros(FRAMES_PER_BUFFER)	
l, = plt.semilogx(xf, 2.0/N * np.abs(y_fft[0:int(N/2)]))
plt.axis([0, fs/2, 0, 0.4])

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
sfreq = Slider(axfreq, 'Cutoff param', 0.001, 0.999, valinit=INIT_FREQ)

sfreq.on_changed(update)
plt.ion()
plt.show()

# open stream using callback 
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels= wf.getnchannels(),
                rate= wf.getframerate(),
		frames_per_buffer= FRAMES_PER_BUFFER,
                output=True,
                stream_callback=callback)

stream.start_stream()


while stream.is_active():
	
	plt.pause(0.001)
	time.sleep(0.001)



# stop stream 
stream.stop_stream()
stream.close()
wf.close()

p.terminate()

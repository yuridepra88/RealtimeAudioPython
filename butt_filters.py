"""PyAudio controllable filters"""

"""Usage: python butt_filters.py --input=./rock.wav --type=high|low|noparam=bandpass   """

import pyaudio
import wave
import time
import sys 
import argparse
from scipy import signal
import scipy
import struct
import numpy as np
import psutil, os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


FRAMES_PER_BUFFER = 512
format_unpack = "%dh"%(FRAMES_PER_BUFFER*2)
SHORT_NORMALIZE = (1.0/32768.0)

INIT_FREQ = 1000

def update(val):
    global cutoff, old_cutoff
    freq = sfreq.val/2
    old_cutoff = cutoff
    cutoff = freq
    fig.canvas.draw_idle()

def design_filter(lowcut, highcut, fs, order=3):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order, [low,high], btype='band')
    return b,a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def process(data):
	global b ,a, zi

	if 'zi' not in globals():
		bandpass_samples= signal.lfilter(b, a, data)
	else:
		bandpass_samples, zi= signal.lfilter(b, a, data, zi=zi)
	return bandpass_samples

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
ap.add_argument("-t", "--type", help = "type of filter")
ap.add_argument("-i", "--input", help = "input file")
args = vars(ap.parse_args())	

wf = wave.open(args["input"], 'rb')
p = pyaudio.PyAudio()
fs = wf.getframerate()  # sample rate, Hz

if (args["type"] == 'low'):
	# Low pass
	order = 3
	cutoff = INIT_FREQ
	old_cutoff = INIT_FREQ
	b, a = butter_lowpass(cutoff, fs, order)
	zi = signal.lfilter_zi(b,a)

elif (args["type"] == 'high'):

	# High pass
	order = 3
	cutoff = INIT_FREQ
	old_cutoff = INIT_FREQ
	b, a = butter_highpass(cutoff, fs, order)
	zi = signal.lfilter_zi(b,a)
else:

	#  Band pass
	order = 3
	cutoff = INIT_FREQ
	old_cutoff = INIT_FREQ
	b,a = design_filter(cutoff,cutoff + 50,fs, order)
	
	zi = signal.lfilter_zi(b,a)



# change parameters of filter while playing

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

T = 1.0 / fs
N = FRAMES_PER_BUFFER
print (T)
print(N)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))	
y_fft = np.zeros(FRAMES_PER_BUFFER)	
l, = plt.semilogx(xf, 2.0/N * np.abs(y_fft[0:int(N/2)]))
plt.axis([0, fs/2, 0, 0.4])



axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
sfreq = Slider(axfreq, 'Freq', 20, 2000.0, valinit=INIT_FREQ)

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

	if (args["type"] == 'low'):

		if cutoff != old_cutoff:
			diff = (old_cutoff -cutoff)/(fs/cutoff)
			n_iter= int(fs/cutoff)
			for i in range(0,n_iter):
				b, a = butter_lowpass(old_cutoff-diff*i, fs, order)
				
			old_cutoff = cutoff
			

	elif (args["type"] == 'high'):
		if cutoff != old_cutoff:
			diff = (old_cutoff -cutoff)/(fs/cutoff)
			n_iter= int(fs/cutoff)
			for i in range(0,n_iter):
				b, a = butter_highpass(old_cutoff-diff*i, fs, order)
				
			old_cutoff = cutoff

	else:
		if cutoff != old_cutoff:
			diff = (old_cutoff -cutoff)/500
			n_iter= 499
			
			for i in range(0,n_iter):
				b,a = design_filter(old_cutoff-diff*i, (old_cutoff-diff*i)+ 150,fs, order)
			old_cutoff = cutoff
	time.sleep(0.001)



# stop stream 
stream.stop_stream()
stream.close()
wf.close()

p.terminate()

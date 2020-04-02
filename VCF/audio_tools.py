from scipy import signal
import numpy as np

 
SHORT_NORMALIZE = (1.0/32768.0)

def pcm_to_double(block, FRAMES_PER_BUFFER):
    format_unpack = "%dh"%(FRAMES_PER_BUFFER)   
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

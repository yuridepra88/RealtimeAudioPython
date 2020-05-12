import numpy as np
import matplotlib.pyplot as plt
import line_profiler
import math
import time

from numba import jitclass # decorator for the class optimization
from numba import int32, float32, float64 

import cython_sine


FRAMES_PER_BUFFER = 2048
sample_rate = 48000
T = 1. / sample_rate



#Class Sine called by Python interpreter
class OscSine():
	def __init__(self, freq, amp):
		self.freq = freq
		self.amp = amp
		self.out = np.zeros(FRAMES_PER_BUFFER)
		self.out[FRAMES_PER_BUFFER-2] =  -amp*math.sin(2*math.pi*freq*T)
			
	def gen_sine(self):
		w= 2*math.pi*self.freq*T
		self.out[0]= 2 * math.cos(w) * self.out[FRAMES_PER_BUFFER-1] - self.out[FRAMES_PER_BUFFER-2]
		self.out[1]= 2 * math.cos(w) * self.out[0] - self.out[FRAMES_PER_BUFFER-1]
		for i in range (2, FRAMES_PER_BUFFER):
			self.out[i] = 2 * math.cos(w) * self.out[i-1] - self.out[i-2]
	
	def get_samples(self):
		self.gen_sine()
		return self.out


#Numba refactoring of the Sine class
#typing of the object attriutes
sineobj = [

    ('freq', float32),              
    ('amp', float32),
	('chunksize', int32),
	('T', float64),
	('out', float64[:]),
]

@jitclass(sineobj)  
class OscSineNumba():
	def __init__(self, freq, amp,FRAMES_PER_BUFFER, T):
		self.freq = freq
		self.amp = amp
		self.chunksize = FRAMES_PER_BUFFER
		self.T = T
		self.out = np.zeros(FRAMES_PER_BUFFER, dtype=np.float64)
		self.out[FRAMES_PER_BUFFER-2] =  -amp*math.sin(2*math.pi*freq*T)
	
	def gen_sine(self):
		w= 2*math.pi*self.freq*self.T
		self.out[0]= 2 * math.cos(w) * self.out[self.chunksize-1] - self.out[self.chunksize-2]
		self.out[1]= 2 * math.cos(w) * self.out[0] - self.out[self.chunksize-1]
		for i in range (2, self.chunksize):
			self.out[i] = 2 * math.cos(w) * self.out[i-1] - self.out[i-2]
			
	def get_samples(self):
		self.gen_sine()
		return self.out




		
# instantiare Sine Objects: 2x in Python, 1x in Cython, 1x in Numba
sine_1 = OscSine(440, 0.8)
sine_2 = OscSine(300, 0.8)
sine_3 = cython_sine.OscSine(200, 0.5, FRAMES_PER_BUFFER, T)
sine_4 = OscSineNumba(440, 0.8, FRAMES_PER_BUFFER, T)


# wrapper methods for profiling
def gen_sine_python():
	return sine_1.get_samples()

def gen_sine_cython():
	return sine_3.get_samples()

def gen_sine_numba():
	return sine_4.get_samples()



#a method to show the generated signals and the mult of two signals
def visualize():
	fig, axs = plt.subplots(3)
	fig.suptitle('Sine mult.')	
	
	a = gen_sine_python()
	b = gen_sine_cython()

	c = a * b
	
	axs[0].plot(a)
	axs[1].plot(b)
	axs[2].plot(c)
	
	plt.show()
	
#compute first chuck
#Numba optimize the method in thi call (slow performance)
t1 = time.clock()
gen_sine_python()
t2 = time.clock()
gen_sine_cython()
t3 = time.clock()
gen_sine_numba()	
t4 = time.clock()

print("Python", t2-t1)
print("Cython", t3-t2)
print("Numba", t4-t3)


profile = line_profiler.LineProfiler()
profile.add_function(gen_sine_python)
profile.add_function(gen_sine_cython)  
profile.add_function(gen_sine_numba)  

# profile the generation of 1000 chuncks
for i in range (0, 1000):                                                                                                   
	profile.run('gen_sine_python()')
	profile.run('gen_sine_cython()')
	profile.run('gen_sine_numba()')

#print the results	
profile.print_stats() 




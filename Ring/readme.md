This folder contains the Python implementation of a Diode Ring Modulator.

First of all run "python setup.py build_ext --inplace" to compile the .pyx files.

- test_speed.py compares offline the performance of 3 different approaches: interpreted approach, cython variable annotations, cython with C bindings. At the end, it print the statistics of the execution time.

- realtime_diodeMod.py plays the real-time implementation with C bindings on 2 signals (test_sine.wav and test_sine_mod.wav)

- diodcython.pyx is an Cython file cointaning annoted code. The result can't be executed in real-time.

- diod_cdef.c is a C file implementig the equations of the model

- c_diod.pyx is a Cython file use to bind the C functions of diod_cdef.c to the corrisponded functions exposed to Python 

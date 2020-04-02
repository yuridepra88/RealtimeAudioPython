This folder contains the Python implementation of a Diode Ring Modulator.

First of all run "python setup.py build_ext --inplace" to compile the .pyx Cython files.

- test_speed.py compares offline the performance of 4 different approaches: Interpreted approach, Numba refactoring, Cython refactoring, Cython with C bindings. At the end, it print the statistics of the execution time.

- realtime_diodeMod.py plays the real-time implementation on 2 test signals with different implementation (test_sine.wav and test_sine_mod.wav)

- diodcython.pyx is an Cython file cointaning annoted code (simple version).

- diod_cdef.c is a C file implementig the equations of the model (for code incapulation)

- c_diod.pyx is a Cython file use to bind the C functions of diod_cdef.c to the corrisponded functions exposed to Python

- audio_tools.py contains support functions

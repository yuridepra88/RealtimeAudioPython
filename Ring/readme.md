This folder contains the Python implementation of a Diode Ring Modulator.

First of all run "python setup.py build_ext --inplace" to compile the .pyx files.

- test_speed.py compares offline the performance of 3 different approaches: interpreted approach, cython variable annotations, cython with C bindings. At the end, it print the statistics of the execution time.

- realtime_diodeMod.py plays the real-time implementation with C bindings on 2 signals (test_sine.wav and test_sine_mod.wav)

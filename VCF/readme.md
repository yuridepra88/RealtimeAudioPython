This folder contains the Python implementation of the VCS3 algorithm.

First of all run "python setup.py build_ext --inplace" to compile the .pyx files.

- vcf_speed.py compares offline the performance of 4 different approaches: interpreted approach, Cython refactoring, Cython with C bindings and Numba. At the end, it print the statistics of the execution time.

- vcf_memory.py measures the memory used along time

- vcf_realtime.py execute the algorithms on .wav file example

- audio_tools.py is a support library

- vcf_cdef.c contains the algorithm in written in C
- c_vcf.pyx is the binding file for C code embedding

- vcf_cython.pyx is the Cython refactoring (simple version)


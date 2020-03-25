#!/usr/bin/env python

''' Usage: python setup.py build_ext --inplace '''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("diod_cython",
                             sources=["diodcython.pyx"],
                             include_dirs=[numpy.get_include()])],
)


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("c_diod",
                             sources=["c_diod.pyx", "diod_cdef.c"],
                             include_dirs=[numpy.get_include()])],
)

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
    ext_modules = [Extension("vcf",
                             sources=["c_vcf.pyx", "vcf_cdef.c"],
                             include_dirs=[numpy.get_include()])],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("vcf_cython",
                             sources=["vcf_cython.pyx"],
                             include_dirs=[numpy.get_include()])],
)


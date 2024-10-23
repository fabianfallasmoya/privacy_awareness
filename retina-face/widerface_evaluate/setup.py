"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', ['box_overlaps.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))

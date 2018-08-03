# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import cython_gsl
import numpy

setup(
    include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("fastMSM",
                             ["fastMSM.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()])]
    )
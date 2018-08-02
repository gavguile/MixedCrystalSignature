#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:29:54 2018

@author: dietz
"""

from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

#extensions = [
#    Extension("sphharm", ["sphharm.pyx"],
#        include_dirs = ['~/anaconda3/include'],
#        library_dirs = ['~/anaconda3/lib']),
#]
#
#setup(
#    name = "sphharm",
#    ext_modules = cythonize(extensions)
#)
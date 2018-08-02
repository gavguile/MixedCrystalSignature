#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:28:03 2018

@author: dietz
"""

from scipy.special.cython_special cimport sph_harm

#cdef extern:
#    double complex sph_harm(long, long, double, double)

cpdef sph_harm_vectorized(double theta, double phi):
    return sph_harm(6, 6, theta, phi)
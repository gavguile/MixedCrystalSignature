#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:22:23 2018

@author: dietz
"""

from sphharm import sph_harm_vectorized
from scipy.special import lpmv
from scipy.special import sph_harm
import numpy as np

test=sph_harm_vectorized(0.3,0.3)


xargs=np.zeros((2,),dtype=np.float64)
xargs[0]=np.pi/5
xargs[1]=np.pi/7
yargs=np.zeros((2,),dtype=np.float64)
yargs[0]=np.pi/8
yargs[1]=np.pi/3
margs=np.zeros((3,2),dtype=np.int32)+6
margs[0]=1
margs[1]=2
largs=np.zeros((2),dtype=np.int32)+6
test2=lpmv(margs,largs,xargs)

test3=sph_harm(margs,largs,xargs,yargs)

print(sph_harm(1,6,np.pi/5,np.pi/8))
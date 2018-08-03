#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:22:23 2018

@author: dietz
"""

from scipy.special import sph_harm
import numpy as np
import pandas as pd

def generate_m_l_arr(l,num_datapoints):
    m_arr=np.arange(-l,l+1,dtype=np.int32)
    m_arr=np.tile(m_arr,(num_datapoints,1))
    return m_arr.transpose()


n_data=10**2*12
np.random.seed(0)
theta_arr=np.random.rand(n_data)*2*np.pi
phi_arr=np.random.rand(n_data)*np.pi

l=6
m_arr=generate_m_l_arr(l,n_data)
l_arr=np.zeros((n_data,),dtype=np.int32)+l

test3=sph_harm(m_arr,l_arr,theta_arr,phi_arr)

df=pd.DataFrame()

df['test1']=[0.1,0.2]
df['test2']=[1,2]
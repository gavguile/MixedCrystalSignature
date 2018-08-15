# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:52:49 2016

@author: dietz
"""
import numpy as np

def add_gaussian_noise(datapoints, sigma, rnd_seed):
    "apply gaussian distributed noise to 3d pointcloud data"
    if sigma == 0:
        return datapoints
    x_len = np.size(datapoints, 0)
    y_len = np.size(datapoints, 1)
    np.random.seed(seed=rnd_seed)

    rand_gaussian = np.random.normal(loc=0.0, scale=sigma, size=(x_len, y_len))
    return datapoints+rand_gaussian

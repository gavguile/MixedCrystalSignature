# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:52:49 2016

@author: dietz
"""
import numpy as np
#import GenerateCrystalNeighborhood as gc
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


def add_gaussian_noise(datapoints, sigma, rnd_seed):
    if sigma == 0:
        return datapoints
    x_len = np.size(datapoints, 0)
    y_len = np.size(datapoints, 1)
    np.random.seed(seed=rnd_seed)
    rand_gaussian = np.random.normal(loc=0.0, scale=sigma, size=(x_len, y_len))
    
    return datapoints+rand_gaussian
    
#datapoints=gc.fill_volume_hcp(10,10,10)
#gauss=add_gaussian_noise(datapoints,20*10**(-2))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(gauss[:,0],gauss[:,1],gauss[:,2],c='b')
#plt.show()

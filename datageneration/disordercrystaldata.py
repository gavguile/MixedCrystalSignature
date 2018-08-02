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


if __name__ == '__main__':
    from generatecrystaldata import fill_volume_fcc
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fcctest = fill_volume_fcc(5, 5, 5)
    fcctest = add_gaussian_noise(fcctest, 0.05, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fcctest[:, 0], fcctest[:, 1], fcctest[:, 2])
    ax.set_title('fcc with noise (5 percent)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 15:00:00 2018

@author: Christopher Dietz
"""

import numpy as np
from scipy.spatial import Voronoi

class MixedCrystalSignature:
    """ This is the docstring """
    solid_thresh = 0.5

    def __init__(self, solid_thresh):
        self.solid_thresh = solid_thresh


if __name__ == '__main__':
    MCS = MixedCrystalSignature(0.4)
    print('test1')

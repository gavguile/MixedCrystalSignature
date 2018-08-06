#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:10:20 2018

@author: dietz
"""

from datageneration.generatecrystaldata import fill_volume_fcc
from calculations import get_inner_volume_bool_vec
import numpy as np
from scipy.spatial import Voronoi

from sympy import Ynm, Symbol,simplify,N


#def calc_convex_hulls(indices,regions,point_region,vertices):
#    conv_hulls=[]
#    for i in indices:
#        voro_points=vertices[regions[point_region[i]]]
#        conv_hulls.append(ConvexHull(voro_points,qhull_options="QJ"))
#    return conv_hulls

datapoints=fill_volume_fcc(5,5,5)
volume=[[2,3],[2,3],[2,3]]
bool_vec= get_inner_volume_bool_vec(datapoints,volume)
idx=np.arange(datapoints.shape[0])[bool_vec][0]

voro=Voronoi(datapoints)
regions=voro.regions
point_region=voro.point_region
vertices=voro.vertices
ridge_points=voro.ridge_points
ridge_vertices=voro.ridge_vertices



region=regions[point_region[idx]]


voro_points=vertices[regions[point_region[idx]]]

theta=Symbol("theta")
phi=Symbol("phi")

generated_code=''

first_iter=True
for n in range(0,7):
    for m in range(0,n+1):
        if n==6 and m==n:
            generated_code += 'else:\n'
        else:
            if first_iter:
                generated_code += 'if '
            else:
                generated_code += 'elif '
            generated_code +='l=={:d} and m=={:d}: \n'.format(n,m)
            
        
        
        sph_harm=N(Ynm(n,m,theta,phi).expand(func=True))
        sph_harm=str(sph_harm).replace('I','1j')
        generated_code+='    return '+sph_harm+'\n'
        first_iter=False

    
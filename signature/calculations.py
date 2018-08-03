#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:10:35 2018

@author: dietz
"""

import numpy as np
from scipy.spatial import ConvexHull
import numba
from math import sqrt,atan2,acos,pi

def get_inner_volume_bool_vec(datapoints,volume):
    x_min, x_max = volume[0]
    y_min, y_max = volume[1]
    z_min, z_max = volume[2]
    
    bool_matrix_min = datapoints >= [x_min,y_min,z_min]
    bool_matrix_max = datapoints <= [x_max,y_max,z_max]
    
    bool_matrix=np.logical_and(bool_matrix_min,bool_matrix_max)
    
    return np.all(bool_matrix,axis=1)

@numba.njit(numba.float64(numba.float64[:],numba.float64[:]))
def calc_area(u,v):
    #cdef double cx,cy,cz
    cx=u[1]*v[2]-u[2]*v[1]
    cy=u[2]*v[0]-u[0]*v[2]
    cz=u[0]*v[1]-u[1]*v[0]
    
    return 0.5*sqrt(cx**2+cy**2+cz**2)

@numba.njit(numba.float64[:,:](numba.int64,numba.float64[:,:],numba.int32[:,:],numba.float64[:,:]))
def calc_voro_area_angle(n_faces, norm_vecs,simpls,points):
    u_vec=np.zeros((3,),dtype=np.float64)
    v_vec=np.zeros((3,),dtype=np.float64)
    result_array = np.zeros((n_faces, 3),dtype=np.float64)
    for i in range(n_faces):
        for j in range(3):
            u_vec[j] = points[simpls[i,1],j]-points[simpls[i,0],j]
            v_vec[j] = points[simpls[i,2],j]-points[simpls[i,0],j]
        result_array[i,0]=calc_area(u_vec,v_vec)
        #phi - azimuthal angle
        result_array[i,1]=atan2(norm_vecs[i, 1], norm_vecs[i, 0])%(2*pi)
        #theta - polar angle
        result_array[i,2]=acos(norm_vecs[i,2])
        
    return result_array

def calc_voro_area_angles(conv_hulls):
    voro_area_angles=[]
    for hull in conv_hulls:
        voro_area_angle=calc_voro_area_angle(hull.simplices.shape[0],hull.equations[:,0:3],
                                             hull.simplices,hull.points)
        voro_area_angles.append(voro_area_angle)
    return voro_area_angles

def calc_convex_hulls(indices,regions,point_region,vertices):
    conv_hulls=[]
    for i in indices:
        voro_points=vertices[regions[point_region[i]]]
        conv_hulls.append(ConvexHull(voro_points,qhull_options="QJ"))
    return conv_hulls

if __name__=='__main__':
    from generatecrystaldata import fill_volume_fcc
    from scipy.spatial import Voronoi
    import time
    
    datapoints=fill_volume_fcc(20, 20, 20)
    volume=[[2,18],[2,18],[2,18]]
    
    inner_bool = get_inner_volume_bool_vec(datapoints,volume)
    
    voro=Voronoi(datapoints)
    regions=voro.regions
    point_region=voro.point_region
    vertices=voro.vertices
    indices = np.arange(0, len(datapoints), dtype=np.int32)
    
    t=time.process_time()
    conv_hulls=calc_convex_hulls(indices,regions,point_region,vertices)
    print('calc_convex_hulls',time.process_time()-t)
    t=time.process_time()
    voro_area_angles=calc_voro_area_angles(conv_hulls)
    print('calc_voro_area_angles',time.process_time()-t)
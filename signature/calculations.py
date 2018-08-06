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
#
# ein "polygon" ist hier einfach nur eine liste von punkten
# das heißt np.array(np.array([[0,0,0],[base,0,0], [base / 2.,height,0]]))
#
    
#cdef int vertex_index_strider(int index, int num_vertices):
#    cdef int forward_index
#    forward_index = index + 1
#    if forward_index > (num_vertices - 1):
#        forward_index = 0
#    return forward_index
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef planar_polygon_area(double[:,:] vertices):
#    cdef int N = vertices.shape[0]
#    cdef int i, forward_index, backward_index
#    cdef double area = 0
#    cdef double delta_x
#
#    for i in range(N):
#        forward_index = vertex_index_strider(i, N)
#        backward_index = i - 1
#        if backward_index < 0:
#            backward_index = N - 1
#        delta_x = (vertices[forward_index, 0] -
#                   vertices[backward_index, 0])
#        area += delta_x * vertices[i, 1]
#    area *= 0.5
#    return area


@njit(numba.complex128[:](numba.int32,numba.int32,numba.int32[:],numba.int32,numba.float64[:],numba.float64[:],numba.float64,numba.float64[:]))
def calc_msm_qlm(lmax, len_l, l_vec,len_angles, theta_vec,phi_vec,total_area,areas):
#    cdef size_t len_plm = 0
#    cdef int len_result = 0
#    cdef size_t index = 0
#    cdef int m = 0
#    cdef int j = 0
#    cdef int index_l =0
#    cdef int l = 0
#    cdef double complex Ylm
    len_plm = gsl_sf_legendre_array_n(lmax)
    
    for i in range(len_l):
        len_result += (2*l_vec[i]+1)

    cdef np.ndarray[double,ndim=1] plm_array = np.zeros(len_plm,dtype=np.float64)
    
    cdef np.ndarray[double complex,ndim=1] qlm_result = np.zeros(len_result,dtype=np.complex128)
    for i in range(len_angles):
        gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM,lmax, cos(theta_vec[i]),
                              <double*> plm_array.data)
        index_l=0
        for j in range(len_l):
            l=l_vec[j]
            for m in range(-l,l+1):
                if m < 0:
                    index = gsl_sf_legendre_array_index(l,-m)
                    Ylm = cexp(1j*m*phi_vec[i])*plm_array[index]
                else:
                    index= gsl_sf_legendre_array_index(l,m)
                    Ylm = ((-1)**(m))*cexp(1j*m*phi_vec[i])*plm_array[index]
                qlm_result[index_l+m+l]+=Ylm*areas[i]
            index_l+=2*l+1
            
    for i in range(len_result):
        qlm_result[i]/=total_area
        
    return qlm_result

#def calc_qlm_array(conv_hulls,voro_area_angles):
#    
#    args=[]
#    for voro_area_angle,hull in zip(voro_area_angles,conv_hulls):
#        args.append([voro_area_angle.shape[0],voro_area_angle[:,2],
#                     voro_area_angle[:,1],hull.area,voro_area_angle[:,0]])
#    
#    
#    calc_qlm_from_voro_part=partial(calc_qlm_from_voro,
#                                    max_l=self.max_l,l_vec=self.l_vec)
#    if self.n_proc > 1:    
#        self.qlm_arrays=np.array(self.p.map(calc_qlm_from_voro_part,args))
#    else:
#        self.qlm_arrays=np.array(list(map(calc_qlm_from_voro_part,args)))
#        
#    self.voro_vols=[hull.volume for hull in self.conv_hulls]

if __name__=='__main__':
    from datageneration.generatecrystaldata import fill_volume_fcc
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
    
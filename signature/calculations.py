#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:10:35 2018

@author: dietz
"""

from math import sqrt, atan2, acos, pi
import numpy as np
import numba
from sphericalharmonics.sphharmhard import sph_harm_hard
from sympy.physics.wigner import wigner_3j
from sympy import N

@numba.njit(numba.float64(numba.float64[:], numba.float64[:]))
def calc_area(u, v):
    """calculates the area between two vectors u and v"""
    cx = u[1]*v[2]-u[2]*v[1]
    cy = u[2]*v[0]-u[0]*v[2]
    cz = u[0]*v[1]-u[1]*v[0]

    return 0.5*sqrt(cx**2+cy**2+cz**2)

@numba.njit(numba.float64[:, :](numba.int64, numba.float64[:, :], numba.int32[:, :], numba.float64[:, :]))
def calc_voro_area_angle(n_faces, norm_vecs, simpls, points):
    """calculates areas of voronoi facets and azumimuthal and polar angles of the facet normals"""
    u_vec = np.zeros((3, ), dtype=np.float64)
    v_vec = np.zeros((3, ), dtype=np.float64)
    result_array = np.zeros((n_faces, 3), dtype=np.float64)
    for i in range(n_faces):
        for j in range(3):
            u_vec[j] = points[simpls[i, 1], j]-points[simpls[i, 0], j]
            v_vec[j] = points[simpls[i, 2], j]-points[simpls[i, 0], j]
        result_array[i, 0] = calc_area(u_vec, v_vec)
        #phi - azimuthal angle
        result_array[i, 1] = atan2(norm_vecs[i, 1], norm_vecs[i, 0])%(2*pi)
        #theta - polar angle
        result_array[i, 2] = acos(norm_vecs[i, 2])

    return result_array

@numba.njit(numba.complex128[:](numba.int64, numba.int32[:], numba.float64[:], numba.float64[:], numba.float64, numba.float64[:]))
def calc_msm_qlm(len_array, l_vec, theta_vec, phi_vec, total_area, areas):
    """calculates the minkowski structure metric (MSM) from voronoi areas and angles

    Explanation is in https://doi.org/10.1063/1.4774084
    """
    len_l = l_vec.shape[0]
    len_angles = theta_vec.shape[0]

    qlm_result = np.zeros(len_array, dtype=np.complex128)
    for i in range(len_angles):
        index_l = 0
        for j in range(len_l):
            l = l_vec[j]
            for m in range(-l, l+1):
                ylm = sph_harm_hard(l, m, theta_vec[i], phi_vec[i])
                qlm_result[index_l+m+l] += ylm*areas[i]
            index_l += 2*l+1

    for i in range(len_array):
        qlm_result[i] /= total_area

    return qlm_result

@numba.njit(numba.float64(numba.int64, numba.complex128[:], numba.int64, numba.complex128[:, :]))
def calc_si(l, qlms, len_neigh, qlms_neigh):
    """calculates the structural order parameter si from bond order parameters qlm

    More on this: https://doi.org/10.1103/PhysRevE.96.011301
    """
    qlm_sum = 0.
    for m in range(2*l+1):
        qlm_sum += abs(qlms[m])**2
    qlm_sum = sqrt(qlm_sum)

    si = 0.
    for i in range(len_neigh):
        qlm_sum_neigh = 0.
        for m in range(2*l+1):
            qlm_sum_neigh += abs(qlms_neigh[i, m])**2
        qlm_sum_neigh = sqrt(qlm_sum_neigh)
        si_inner = 0.
        for m in range(2*l+1):
            si_inner += (qlms[m]*qlms_neigh[i, m].conjugate()).real
        si += si_inner/(qlm_sum*qlm_sum_neigh)

    return si/len_neigh

@numba.njit(numba.float64[:, :](numba.int32[:], numba.complex128[:, :]))
def calc_qls_from_qlm_arrays(l_vec, qlm_arrays):
    "calculates the final ql (over all m) from qlm data"
    len_l = l_vec.shape[0]
    
    result = np.zeros((qlm_arrays.shape[0], len_l), dtype=np.float64)
    
    for i in range(qlm_arrays.shape[0]):
        j = 0
        m = 0
        l = 0
        index_l = 0
        qlm_sum = 0.
        for j in range(len_l):
            l = l_vec[j]
            qlm_sum = 0.
            
            for m in range(-l, l+1):
                qlm_sum += abs(qlm_arrays[i, index_l+m+l])**2
            result[i, j] = sqrt(4.*pi/(2.*l+1)*qlm_sum)
            index_l += 2*l+1

    return result

@numba.njit(numba.float64[:, :](numba.int32[:], numba.complex128[:, :],numba.float64[:],numba.int32[:,:],numba.int32[:]))
def calc_wls_from_qlm_arrays(l_vec, qlm_arrays,wigner_arr,m_arr,count_arr):
    "calculates the final ql (over all m) from qlm data"
    len_l = l_vec.shape[0]
    
    result = np.zeros((qlm_arrays.shape[0], len_l), dtype=np.float64)

    for i in range(qlm_arrays.shape[0]):
        prevcount=0
        index_l=0
        for j in range(len_l):
            l = l_vec[j]
            w=0.+0*1j
            for k in range(prevcount,count_arr[j]):
                w+=wigner_arr[k]*qlm_arrays[i][m_arr[k,0]]*qlm_arrays[i][m_arr[k,1]]*qlm_arrays[i][m_arr[k,2]]
            
            qlm_sum=0
            for m in range(-l, l+1):
                qlm_sum += abs(qlm_arrays[i, index_l+m+l])**2
            qlm_sum=qlm_sum**(3/2)
            w=w/qlm_sum
            result[i,j]=w.real
            prevcount=count_arr[j]
            index_l += 2*l+1

    return result

def calc_wigner3j_general(l_vec):
    wignerlist=[]
    mlist=[]
    index_l=0
    count=0
    countlist=[]
    for j in range(l_vec.shape[0]):
        l = l_vec[j]
        for m1 in range(-l,l+1):
            for m2 in range(-l,l+1):
                for m3 in range(-l,l+1):
                    if m1+m2+m3 == 0:
                        m1_new=index_l+m1+l
                        m2_new=index_l+m2+l
                        m3_new=index_l+m3+l
                        wigner=float(N(wigner_3j(l,l,l,m1,m2,m3)))
                        wignerlist.append(wigner)
                        mlist.append([m1_new,m2_new,m3_new])
                        count+=1
        index_l += 2*l+1
        countlist.append(count)
    return np.array(wignerlist,dtype=np.float64),np.array(mlist,dtype=np.int32),np.array(countlist,dtype=np.int32)

@numba.njit(numba.float64[:](numba.int32[:],numba.float64[:,:],numba.float64[:]))
def calc_angles(neighbors,datapoints,centerpoint):
    n_neighbors=len(neighbors)
    angles=np.zeros((n_neighbors*(n_neighbors-1)//2),dtype=np.float64)
    count=0
    for j in range(n_neighbors):
        for k in range(j+1,n_neighbors):
            idx1=neighbors[j]
            idx2=neighbors[k]
            numerator=0.
            sum_u=0.
            sum_v=0.
            for l in range(3):
                u=datapoints[idx1,l]-centerpoint[l]
                v=datapoints[idx2,l]-centerpoint[l]
                numerator+=u*v
                sum_u+=u**2
                sum_v+=v**2
            angles[count]=numerator/(sqrt(sum_u*sum_v))
            count+=1
    return angles

def calc_bond_angles(indices,neighborlist,datapoints):
    angle_edges=np.array([-1.0, -0.945, -0.915, -0.755, -0.195, 0.195, 0.245, 0.795, 1.0])
    bond_angles=np.zeros((datapoints.shape[0],angle_edges.shape[0]-1),dtype=np.int32)
    
    for i in indices:
        angles=calc_angles(np.array(neighborlist[i],dtype=np.int32),datapoints,datapoints[i])
        bond_angles[i]=fast_hist(angles,angle_edges)
    return bond_angles

@numba.njit(numba.float64[:](numba.int32[:],numba.float64[:,:]))
def calc_distances(neighbors,datapoints):
    n_neighbors=len(neighbors)
    distances=np.zeros((n_neighbors*(n_neighbors-1)//2),dtype=np.float64)
    count=0
    for j in range(n_neighbors):
        for k in range(j+1,n_neighbors):
            idx1=neighbors[j]
            idx2=neighbors[k]
            normsq=0.
            for l in range(3):
                normsq+=(datapoints[idx1,l]-datapoints[idx2,l])**2
            distances[count]=sqrt(normsq)
            count+=1
    return distances

def calc_hist_distances(indices,neighborlist,datapoints,volumes):
    nbins_distances=12
    hist_distances=np.zeros((datapoints.shape[0],nbins_distances),dtype=np.int32)
    
    for i in indices:
        d0=volumes[i]**(1/3)
        distance_edges=fast_edges(d0,nbins_distances)
        distances=calc_distances(np.array(neighborlist[i],dtype=np.int32),datapoints)/d0
        hist_distances[i]=fast_hist(distances,distance_edges)
    return hist_distances

@numba.njit(numba.int32[:](numba.float64[:],numba.float64[:]))
def fast_hist(data, bin_edges):
    hist,_=np.histogram(data,bin_edges)
    return hist.astype(np.int32)

@numba.njit(numba.float64[:](numba.float64,numba.int64))
def fast_edges(d0,nbins_distances):
    return np.linspace(0,d0*1.5,nbins_distances+1)

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

@numba.njit(numba.float64[:](numba.complex128[:], numba.float64[:], numba.complex128[:], numba.float64[:]))
def calc_w4_w6_from_qlm_array(q4m_arr, wignerw4, q6m_arr, wignerw6):
    result = np.zeros(2,dtype=np.float64)
    
    i=0
    
    w4_msum=0.
    w4=0.
    
    for i in range(1,5):
        w4_msum+=wignerw4[i]*(q4m_arr[i+4]*q4m_arr[i+4].conjugate()).real
    w4_msum*=6.
    
    w4=q4m_arr[4].real
    w4*=wignerw4[0]*(q4m_arr[4]*q4m_arr[4].conjugate()).real + w4_msum
    w4+=12.*wignerw4[5]*(q4m_arr[8].conjugate()*q4m_arr[5]*q4m_arr[7]).real
    w4-=12.*wignerw4[6]*(q4m_arr[7].conjugate()*q4m_arr[5]*q4m_arr[6]).real
    w4+= 6.*wignerw4[7]*(q4m_arr[8].conjugate()*q4m_arr[6]*q4m_arr[6]).real
    w4+= 6.*wignerw4[8]*(q4m_arr[6].conjugate()*q4m_arr[5]*q4m_arr[5]).real
    
    q4=0.
    for i in range(9):
        q4+=(q4m_arr[i]*q4m_arr[i].conjugate()).real
    q4=sqrt(q4)
    
    result[0]=w4/q4**3
    
    #w6
    w6_msum=0.
    w6=0.
    
    for i in range(1,7):
        w6_msum+=wignerw6[i]*(q6m_arr[i+6]*q6m_arr[i+6].conjugate()).real
    w6_msum*=6.
    
    w6=q6m_arr[6].real
    w6*=wignerw6[0]*(q6m_arr[6]*q6m_arr[6].conjugate()).real + w6_msum
    w6+= 12.*wignerw6[7]*(q6m_arr[12].conjugate()*q6m_arr[7]*q6m_arr[11]).real
    w6+= 12.*wignerw6[8]*(q6m_arr[12].conjugate()*q6m_arr[8]*q6m_arr[10]).real
    w6+= 12.*wignerw6[9]*(q6m_arr[10].conjugate()*q6m_arr[7]*q6m_arr[9]).real
    w6-=12.*wignerw6[10]*(q6m_arr[11].conjugate()*q6m_arr[7]*q6m_arr[10]).real
    w6-=12.*wignerw6[11]*(q6m_arr[11].conjugate()*q6m_arr[8]*q6m_arr[9]).real
    w6-=12.*wignerw6[12]*(q6m_arr[9].conjugate()*q6m_arr[7]*q6m_arr[8]).real
    w6+= 6.*wignerw6[13]*(q6m_arr[12].conjugate()*q6m_arr[9]*q6m_arr[9]).real
    w6+= 6.*wignerw6[14]*(q6m_arr[10].conjugate()*q6m_arr[8]*q6m_arr[8]).real
    w6+= 6.*wignerw6[15]*(q6m_arr[8].conjugate()*q6m_arr[7]*q6m_arr[7]).real
    
    q6=0.
    for i in range(13):
        q6+=(q6m_arr[i]*q6m_arr[i].conjugate()).real
    q6=sqrt(q6)

    result[1]=w6/q6**3
    
    return result
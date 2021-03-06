#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 15:00:00 2018

@author: Christopher Dietz
"""

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, ConvexHull
import signature.calculations as calc
from functools import partial

class MixedCrystalSignature:
    """Class for calculation of the Mixed Crystal Signature 
    Description in https://doi.org/10.1103/PhysRevE.96.011301"""

    L_VEC = np.array([4, 5, 6],dtype=np.int32) #Choose which l to use for calculation of qlm 
    MAX_L = np.max(L_VEC)

    def __init__(self, solid_thresh=0.55, pool=None):
        """solid_thresh is a threshold between 0 (very disordered) and 1 (very crystalline)
        pool is a pool from the multiprocessing module.
        If no pool is provided, the calculation will be single-core""" 
        self.solid_thresh = solid_thresh
        self.inner_bool = None
        self.indices = None
        self.outsider_indices = None
        self.insider_indices = None
        self.voro = None
        self.neighborlist = None
        self.conv_hulls = None
        self.voro_vols = None
        self.qlm_arrays = None
        self.signature = pd.DataFrame()
        self.datapoints = None
        
        self.p = None
        if pool is not None:
            self.p = pool
        
        self.len_qlm=0
        self.idx_qlm=dict()
        for i,l in enumerate(self.L_VEC):
            self.idx_qlm[l]=np.arange(self.len_qlm,self.len_qlm+2*l+1,dtype=np.int32)
            self.len_qlm += (2*self.L_VEC[i]+1)   
    
    def set_datapoints(self,data):
        """provide datapoints for signature calculation"""
        self.datapoints=data
        self.inner_bool=np.ones(self.datapoints.shape[0],dtype=np.bool)
        self.calc_inner_outer_indices()
        
    def calc_voro(self):
        """calculate voronoi diagram of the datapoints"""
        self.voro=Voronoi(self.datapoints)
        
    def calc_neighborlist(self):
        """retrieve neighborlists from voronoi diagram"""
        ridge_points=self.voro.ridge_points
        self.neighborlist=[[] for _ in range(self.datapoints.shape[0])]
        for j in range(len(ridge_points)):
            if ridge_points[j,1] not in self.neighborlist[ridge_points[j,0]]:
                self.neighborlist[ridge_points[j,0]].append(ridge_points[j,1])
            if ridge_points[j,0] not in self.neighborlist[ridge_points[j,1]]:
                self.neighborlist[ridge_points[j,1]].append(ridge_points[j,0])
        
    def calc_inner_outer_indices(self):
        """calculate indices to choose inner volume datapoints or outer volume datapoints"""
        self.indices = np.arange(0, self.datapoints.shape[0], dtype=np.int32)
        self.outsider_indices = self.indices[np.invert(self.inner_bool)]
        self.insider_indices = self.indices[self.inner_bool]
        
    def set_inner_volume(self,volume):
        """define the inner volume
        format of the volume:
        volume=[[x_min,xmax],[y_min,y_max],[z_min,z_max]]"""
        x_min, x_max = volume[0]
        y_min, y_max = volume[1]
        z_min, z_max = volume[2]
        
        bool_matrix_min = self.datapoints >= [x_min,y_min,z_min]
        bool_matrix_max = self.datapoints <= [x_max,y_max,z_max]
        
        bool_matrix=np.logical_and(bool_matrix_min,bool_matrix_max)
    
        self.inner_bool=np.all(bool_matrix,axis=1)
        self.calc_inner_outer_indices()
        
    def set_inner_bool_vec(self,bool_vec):
        """define inner volume with a customized array of booleans
        length of bool_vec needs to be the number of rows in datapoints"""
        self.inner_bool=bool_vec
        self.calc_inner_outer_indices()
    
    def calc_convex_hulls(self):
        """calculate the convex hulls for all datapoints"""
        regions=self.voro.regions
        point_region=self.voro.point_region
        vertices=self.voro.vertices
        voro_points_list=[vertices[regions[point_region[i]]] for i in self.indices]
        
        if self.p is not None:
            self.conv_hulls= self.p.map(partial(ConvexHull,qhull_options="QJ"),voro_points_list,chunksize=400)
        else:
            self.conv_hulls=[ConvexHull(voro_points_list[i],qhull_options="QJ") for i in self.indices]
    
    def calc_voro_area_angles(self):
        """calculate voronoi facet areas and normal vectors"""
        voro_area_angles=[]
        for hull in self.conv_hulls:
            voro_area_angle=calc.calc_voro_area_angle(hull.simplices.shape[0],
                                                      hull.equations[:,0:3],
                                                      hull.simplices,hull.points)
            voro_area_angles.append(voro_area_angle)
        return voro_area_angles
        
    def calc_qlm_array(self):
        """calculate qlm from minkowski structure metric
        Description in https://doi.org/10.1103/PhysRevE.96.011301"""
        self.calc_voro()
        self.calc_neighborlist()
        self.calc_convex_hulls()
        self.voro_area_angles=self.calc_voro_area_angles()
        
        self.total_areas=[hull.area for hull in self.conv_hulls]
        self.voro_vols=[hull.volume for hull in self.conv_hulls]
        
        len_array=0 
        for i in range(self.L_VEC.shape[0]): 
            len_array += (2*self.L_VEC[i]+1) 
            
        self.qlm_arrays=np.zeros((len(self.total_areas),len_array),dtype=np.complex128) 
        
        for i in range(len(self.total_areas)):
            self.qlm_arrays[i,:]=calc.calc_msm_qlm(len_array,
                                                   self.L_VEC,
                                                   self.voro_area_angles[i][:,2],
                                                   self.voro_area_angles[i][:,1],
                                                   self.total_areas[i],
                                                   self.voro_area_angles[i][:,0])
    
    def calc_struct_order(self):
        """calculate the structural order for every particle
        Description in https://doi.org/10.1103/PhysRevE.96.011301"""
        si_l=6 #this should only make sense with l=6, so its hardcoded
        self.solid_bool=np.zeros(self.datapoints.shape[0],dtype=np.bool)
        self.struct_order=np.zeros(self.datapoints.shape[0],dtype=np.float64)
        for i in self.insider_indices:
            voro_neighbors = np.array(self.neighborlist[i],dtype=np.int64)
            qlm_array_neighbors = self.qlm_arrays[voro_neighbors][:,self.idx_qlm[si_l]]
            num_neighbors=len(self.neighborlist[i])
            si=calc.calc_si(6,self.qlm_arrays[i,self.idx_qlm[si_l]],num_neighbors,qlm_array_neighbors)
            self.solid_bool[i]=(si>=self.solid_thresh)
            self.struct_order[i]=si
        self.solid_indices=self.indices[np.logical_and(self.inner_bool,self.solid_bool)]
    
    def calc_num_neigh(self):
        """calculate the number of neighbors for all solid particles"""
        self.signature['N']=[len(self.neighborlist[i]) for i in self.solid_indices]
    
    def calc_msm(self):
        """calculate ql from minkowski structure metric for all solid particles
        Description in https://doi.org/10.1103/PhysRevE.96.011301"""
        ql_array=calc.calc_qls_from_qlm_arrays(self.L_VEC,self.qlm_arrays[self.solid_indices]).transpose()
        for l in self.L_VEC:
            self.signature['q{:d}'.format(l)]=ql_array[self.L_VEC==l][0]
        
        wigner_arr,m_arr,count_arr=calc.calc_wigner3j_general(self.L_VEC)
        wl_array=calc.calc_wls_from_qlm_arrays(self.L_VEC,self.qlm_arrays[self.solid_indices],wigner_arr,m_arr,count_arr).transpose()
        for l in self.L_VEC:
            if l%2==0: #odd number w_l are useless
                self.signature['w{:d}'.format(l)]=wl_array[self.L_VEC==l][0]
    
    def calc_bond_angles(self):
        """calculate bond angles for all solid particles
        definition in: https://doi.org/10.1103/PhysRevB.73.054104"""
        bond_angles=calc.calc_bond_angles(self.solid_indices,self.neighborlist,self.datapoints)
        for dim in range(bond_angles.shape[1]):
            self.signature['ba{:d}'.format(dim)]=bond_angles[:,dim]
        
    def calc_hist_distances(self):
        """calculate histogram of normalized distances
        Modified from https://doi.org/10.1103/PhysRevE.96.011301"""
        hist_distances=calc.calc_hist_distances(self.solid_indices,self.neighborlist,self.datapoints,self.voro_vols)
        for dim in range(hist_distances.shape[1]):
            self.signature['dist{:d}'.format(dim)]=hist_distances[:,dim]

    def calc_minkowski_eigvals(self):
        """calculate eigenvalues of rank 4 minkowski tensor for all solid particles
        Description in https://doi.org/10.1103/PhysRevE.85.030301"""
        eigenvals_arr=np.zeros((self.solid_indices.shape[0],6),dtype=np.float64)
        for idx in range(self.solid_indices.shape[0]):
            i=self.solid_indices[idx]
            eigenvals_arr[idx]=calc.calc_minkowski_eigenvalues(self.total_areas[i],
                                                               self.voro_area_angles[i][:,0],
                                                               self.conv_hulls[i].equations[:,0:3])
        for dim in range(eigenvals_arr.shape[1]):
            self.signature['zeta{:d}'.format(dim)]=eigenvals_arr[:,dim]

    def calc_signature(self):
        """Function to calculate the mixed crystal signature on the dataset
        Description in https://doi.org/10.1103/PhysRevE.96.011301 (with minor modifications)"""
        self.signature=pd.DataFrame()
        
        self.calc_qlm_array()
        self.calc_struct_order()
        self.calc_num_neigh()
        self.calc_bond_angles()
        self.calc_msm()
        self.calc_minkowski_eigvals()
        self.calc_hist_distances()

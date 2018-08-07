#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 15:00:00 2018

@author: Christopher Dietz
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import signature.calculations as calc
from functools import partial

try:
    import multiprocessing as mp
    MP_EXISTS=False
except ImportError:
    MP_EXISTS=False

class MixedCrystalSignature:
    """ This is the docstring """
    
    NBINS_DISTANCES=12
    L_VEC=np.array([2,4,6],dtype=np.int32)
    MAX_L=np.max(L_VEC)

    def __init__(self,data,solid_thresh=0.652):
        self.solid_thresh=solid_thresh
        self.inner_bool=None
        self.indices=None
        self.outsider_indices=None
        self.insider_indices=None
        self.voro=None
        self.neighborlist=None
        self.conv_hulls=None
        self.voro_vols=None
        self.qlm_arrays=None
        
        self.set_datapoints(data)

        if MP_EXISTS:
            self.p = mp.Pool()
        
        self.len_qlm=0
        self.idx_qlm=[]
        for i,l in enumerate(self.L_VEC):
            self.idx_qlm.append(np.arange(self.len_qlm,self.len_qlm+2*l+1,dtype=np.int32))
            self.len_qlm += (2*self.L_VEC[i]+1)
    
    def __del__(self):
        if MP_EXISTS:
            self.p.close()
            self.p.join()
            
    def set_datapoints(self,data):
        self.datapoints=data
        self.inner_bool=np.ones(self.datapoints.shape[0],dtype=np.bool)
        self.calc_inner_outer_indices()
        
    def calc_voro(self):
        self.voro=Voronoi(self.datapoints)
        
    def calc_neighborlist(self):
        ridge_points=self.voro.ridge_points
        self.neighborlist=[[] for _ in range(self.datapoints.shape[0])]
        for j in range(len(ridge_points)):
            if ridge_points[j,1] not in self.neighborlist[ridge_points[j,0]]:
                self.neighborlist[ridge_points[j,0]].append(ridge_points[j,1])
            if ridge_points[j,0] not in self.neighborlist[ridge_points[j,1]]:
                self.neighborlist[ridge_points[j,1]].append(ridge_points[j,0])
        
    def calc_inner_outer_indices(self):
        self.indices = np.arange(0, self.datapoints.shape[0], dtype=np.int32)
        self.outsider_indices = self.indices[np.invert(self.inner_bool)]
        self.insider_indices = self.indices[self.inner_bool]
        
    def set_inner_volume(self,volume):
        x_min, x_max = volume[0]
        y_min, y_max = volume[1]
        z_min, z_max = volume[2]
        
        bool_matrix_min = self.datapoints >= [x_min,y_min,z_min]
        bool_matrix_max = self.datapoints <= [x_max,y_max,z_max]
        
        bool_matrix=np.logical_and(bool_matrix_min,bool_matrix_max)
    
        self.inner_bool=np.all(bool_matrix,axis=1)
        self.calc_inner_outer_indices()
        
    def set_inner_bool_vec(self,bool_vec):
        self.inner_bool=bool_vec
        self.calc_inner_outer_indices()
    
    def calc_convex_hulls(self):
        regions=self.voro.regions
        point_region=self.voro.point_region
        vertices=self.voro.vertices
        voro_points_list=[vertices[regions[point_region[i]]] for i in self.indices]
        
        if MP_EXISTS:
            self.conv_hulls= self.p.map(partial(ConvexHull,qhull_options="QJ"),voro_points_list,chunksize=400)
        else:
            self.conv_hulls=[ConvexHull(voro_points_list[i],qhull_options="QJ") for i in self.indices]
    
    def calc_voro_area_angles(self):
        voro_area_angles=[]
        for hull in self.conv_hulls:
            voro_area_angle=calc.calc_voro_area_angle(hull.simplices.shape[0],
                                                      hull.equations[:,0:3],
                                                      hull.simplices,hull.points)
            voro_area_angles.append(voro_area_angle)
        return voro_area_angles
        
    def calc_qlm_array(self):
        self.calc_voro()
        self.calc_neighborlist()
        self.calc_convex_hulls()
        voro_area_angles=self.calc_voro_area_angles()
        
        total_areas=[hull.volume for hull in self.conv_hulls]
        self.voro_vols=[hull.volume for hull in self.conv_hulls]
        
        len_array=0 
        for i in range(self.L_VEC.shape[0]): 
            len_array += (2*self.L_VEC[i]+1) 
            
        self.qlm_arrays=np.zeros((len(total_areas),len_array),dtype=np.complex128) 
        
        for i in range(len(total_areas)):
            self.qlm_arrays[i,:]=calc.calc_msm_qlm(len_array,
                                                   self.L_VEC,
                                                   voro_area_angles[i][:,2],
                                                   voro_area_angles[i][:,1],
                                                   total_areas[i],
                                                   voro_area_angles[i][:,0])
    
    def calc_struct_order(self):
        self.solid_bool=np.zeros(self.datapoints.shape[0],dtype=np.bool)
        self.struct_order=np.zeros(self.datapoints.shape[0],dtype=np.float64)
        for i in self.insider_indices:
            voro_neighbors = np.array(self.neighborlist[i],dtype=np.int64)
            qlm_array_neighbors = self.qlm_arrays[voro_neighbors][:,self.idx_qlm[2]]
            num_neighbors=len(self.neighborlist[i])
            si=calc.calc_si(6,self.qlm_arrays[i,self.idx_qlm[2]],num_neighbors,qlm_array_neighbors)
            self.solid_bool[i]=(si>=self.solid_thresh)
            self.struct_order[i]=si

if __name__ == '__main__':
    from datageneration.generatecrystaldata import fill_volume_fcc
    import time
    
    t_tot=time.process_time()
    
    size=[10,10,10]
    datapoints=fill_volume_fcc(size[0], size[1], size[2])
    volume=[[2,size[i]-2] for i in range(3)]
    
    mcs=MixedCrystalSignature(datapoints)
    mcs.set_inner_volume(volume)
    
    t=time.process_time()
    mcs.calc_qlm_array()
    print('calc_qlm_array',time.process_time()-t)
    
    t=time.process_time()
    mcs.calc_struct_order()
    print('calc_struct_order',time.process_time()-t)
    
    print('total time:',time.process_time()-t_tot)
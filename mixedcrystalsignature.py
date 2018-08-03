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
        
    def calc_voro_area_angles(self):
        args=[]
                
        for hull in self.conv_hulls:
            args.append([len(hull.simplices),hull.equations[:,0:3],hull.simplices,hull.points])
        
#        if self.n_proc > 1:
#            self.voro_area_angles=self.p.map(calc_voro_area_angle,args)
#        else:
#            self.voro_area_angles=list(map(calc_voro_area_angle,args))
        
                
    def calc_convex_hulls(self):
        voro_points_list=[]
        regions=self.voro.regions
        point_region=self.voro.point_region
        vertices=self.voro.vertices
        for i in self.indices:
            voro_points_list.append(vertices[regions[point_region[i]]])
        
 #       calc_convex_hull=partial(ConvexHull,qhull_options="QJ")
        
#        if self.n_proc > 1:    
#            self.conv_hulls=self.p.map(calc_convex_hull,voro_points_list)
#        else:
#            self.conv_hulls=list(map(calc_convex_hull,voro_points_list))
        
    def calc_qlm_array(self):
        self.calc_convex_hulls()
        self.calc_voro_area_angles()
        
        args=[]
        for voro_area_angle,hull in zip(self.voro_area_angles,self.conv_hulls):
            args.append([voro_area_angle.shape[0],voro_area_angle[:,2],
                         voro_area_angle[:,1],hull.area,voro_area_angle[:,0]])

#        calc_qlm_from_voro_part=partial(calc_qlm_from_voro,
#                                        max_l=self.max_l,l_vec=self.l_vec)
#        if self.n_proc > 1:    
#            self.qlm_arrays=np.array(self.p.map(calc_qlm_from_voro_part,args))
#        else:
#            self.qlm_arrays=np.array(list(map(calc_qlm_from_voro_part,args)))
            
            
        self.voro_vols=[hull.volume for hull in self.conv_hulls]

if __name__ == '__main__':
    MCS = MixedCrystalSignature(0.4)
    print('test1')

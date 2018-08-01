#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:16:50 2017

@author: dietz
"""

import numpy as np
from scipy.spatial import Voronoi
import GenerateCrystalNeighborhood as gc
import time
from Wigner3JTools import calc_w4_wigner3j, calc_w6_wigner3j, calc_wigner3j_general

from DisorderCrystalStructure import add_gaussian_noise
from functools import partial
from psutil import cpu_count
from multiprocessing import Pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import AnalyzeMDSim as asim
#from AnalyzeMDSim import get_dumpfile_datapoints

import itertools as it
from scipy.spatial import ConvexHull
import fastMSM

class MCS:
    """Mixed Crystal Signature Class"""
    si_threshold = 0.652
    nbins_distances = 12
#    wignerw4=calc_w4_wigner3j()
#    wignerw6=calc_w6_wigner3j()
    
    wignerw4=calc_wigner3j_general(4)
    wignerw6=calc_wigner3j_general(6)
    sign_dimension=1+8+4+6+nbins_distances
    
    def __init__(self, l_vec=np.array([2,4,6],dtype=np.int32), si_threshold=0.652, nbins_distances=12, n_proc=1):

        self.l_vec=l_vec
        self.max_l=np.max(self.l_vec)
        self.si_threshold=si_threshold
        self.nbins_distances=nbins_distances
        self.n_proc=n_proc
        if self.n_proc > 1:
            self.p = Pool(processes=n_proc)
        
        self.len_qlm=0
        self.idx_qlm=[]
        for i,l in enumerate(l_vec):
            self.idx_qlm.append(np.arange(self.len_qlm,self.len_qlm+2*l+1,dtype=np.int32))
            self.len_qlm += (2*l_vec[i]+1)
            
    def __del__(self):
        if self.n_proc > 1:
            self.p.close()
    
    def close_proc(self):
        if self.n_proc > 1:
            self.p.close()
            
    def set_datapoints(self,datapoints):
        self.datapoints=datapoints
        self.inner_bool=np.ones(len(datapoints),dtype=np.bool)
        self.calc_voro()
        self.calc_neighborlist()
        self.calc_inner_outer_indices()
    
    def calc_voro(self):
        self.voro=Voronoi(self.datapoints)
        
    def calc_neighborlist(self):
        ridge_points=self.voro.ridge_points
        self.neighborlist=[[] for _ in range(len(self.datapoints))]
        for j in range(len(ridge_points)):
            if ridge_points[j,1] not in self.neighborlist[ridge_points[j,0]]:
                self.neighborlist[ridge_points[j,0]].append(ridge_points[j,1])
            if ridge_points[j,0] not in self.neighborlist[ridge_points[j,1]]:
                self.neighborlist[ridge_points[j,1]].append(ridge_points[j,0])
                
    def calc_voro_area_angles(self):
        args=[]
                
        for hull in self.conv_hulls:
            args.append([len(hull.simplices),hull.equations[:,0:3],hull.simplices,hull.points])
        
        if self.n_proc > 1:
            self.voro_area_angles=self.p.map(calc_voro_area_angle,args)
        else:
            self.voro_area_angles=list(map(calc_voro_area_angle,args))
        
                
    def calc_convex_hulls(self):
        voro_points_list=[]
        regions=self.voro.regions
        point_region=self.voro.point_region
        vertices=self.voro.vertices
        for i in self.indices:
            voro_points_list.append(vertices[regions[point_region[i]]])
        
        calc_convex_hull=partial(ConvexHull,qhull_options="QJ")
        
        if self.n_proc > 1:    
            self.conv_hulls=self.p.map(calc_convex_hull,voro_points_list)
        else:
            self.conv_hulls=list(map(calc_convex_hull,voro_points_list))
        
                
    def calc_qlm_array(self):
#        self.qlm_arrays=np.zeros((len(self.datapoints),self.len_qlm),dtype=np.complex128)
#        for i in self.indices:
#            self.qlm_arrays[i,:]=self.calc_qlm_from_voro(i)

#        voro_points_list=[]
#        regions=self.voro.regions
#        point_region=self.voro.point_region
#        vertices=self.voro.vertices
#        for i in self.indices:
#            voro_points_list.append(vertices[regions[point_region[i]]])
            
        self.calc_convex_hulls()
        self.calc_voro_area_angles()
        
        args=[]
        for voro_area_angle,hull in zip(self.voro_area_angles,self.conv_hulls):
            args.append([voro_area_angle.shape[0],voro_area_angle[:,2],
                         voro_area_angle[:,1],hull.area,voro_area_angle[:,0]])
        
            #    return fastMSM.calc_msm_qlm(max_l,len(l_vec),l_vec,
#                                voro_area_angle.shape[0],
#                                voro_area_angle[:,2],
#                                voro_area_angle[:,1],
#                                voro_hull.area, voro_area_angle[:,0])
            
        calc_qlm_from_voro_part=partial(calc_qlm_from_voro,
                                        max_l=self.max_l,l_vec=self.l_vec)
        if self.n_proc > 1:    
            self.qlm_arrays=np.array(self.p.map(calc_qlm_from_voro_part,args))
        else:
            self.qlm_arrays=np.array(list(map(calc_qlm_from_voro_part,args)))
            
            
        self.voro_vols=[hull.volume for hull in self.conv_hulls]
        
#        self.voro_vols=self.qlm_arrays[:,-7]
#        self.qlm_arrays=self.qlm_arrays[:,:-6]
#        self.minkowksi_eigenvalues=self.qlm_arrays[:,-6:]
    
    def calc_si_bool(self):
        self.solid_bool=np.zeros(len(self.datapoints),dtype=np.bool)
        self.softness=np.zeros(len(self.datapoints),dtype=np.float64)
        for i in self.insider_indices:
            voro_neighbors = np.array(self.neighborlist[i],dtype=np.int64)
            qlm_array_neighbors = self.qlm_arrays[voro_neighbors][:,self.idx_qlm[2]]
            num_neighbors=len(self.neighborlist[i])
            si=fastMSM.calc_si(6,self.qlm_arrays[i,self.idx_qlm[2]],num_neighbors,qlm_array_neighbors)
            self.solid_bool[i]=(si>=self.si_threshold)
            self.softness[i]=si
    
    def calc_sign_array(self):
#        self.sign_array=np.zeros((len(self.insider_indices),self.sign_dimension),dtype=np.double)-1
#        for k,idx in enumerate(self.insider_indices):
#            self.sign_array[k]=self.calc_particle_signature(idx)

#        norm_vecs=[]
#        for hull,datapoint in zip(self.conv_hulls,self.datapoints):    
#            norm_vecs.append(calc_voro_norm_vec(hull.simplices, hull.points, datapoint))
        datalist=[]
        self.solid_indices=self.indices[np.logical_and(self.inner_bool,self.solid_bool)]
        for i in self.solid_indices:
            voro_neighbors = np.array(self.neighborlist[i],dtype=np.int64)
            if len(voro_neighbors >= 6):
                qlm_array=self.qlm_arrays[i]
                datapoint=self.datapoints[i]
                neighborpoints=self.datapoints[voro_neighbors]
                datalist.append([voro_neighbors,qlm_array,datapoint,neighborpoints,
                                 self.conv_hulls[i].area,self.voro_area_angles[i][:,0],
                                 self.conv_hulls[i].equations[:,0:3]])
            else:
                print('Warning: low number of neighbors')
    
#        voro_area=datalist[5]
#        voro_areas=datalist[6]
#        normvecs=datalist[7]
    
        calc_particle_signature_part=partial(calc_particle_signature,sign_dimension=self.sign_dimension,
                                             idx_qlm=self.idx_qlm, si_threshold=self.si_threshold,
                                             nbins_distances=self.nbins_distances, l_vec=self.l_vec,
                                             wignerw4=self.wignerw4, wignerw6=self.wignerw6)
        if self.n_proc > 1:
            self.sign_array=np.array(self.p.map(calc_particle_signature_part,datalist))
        else:
            self.sign_array=np.array(list(map(calc_particle_signature_part,datalist)))
        return self.sign_array
        
        
    def calc_w4_w6_from_qlm_array(self,qlm_array):
        result = np.zeros(2, dtype=np.double)
        
        #W4
        q4m_arr=qlm_array[self.idx_qlm[1]]
        w4_msum=6*np.sum(self.wignerw4[1:5]*np.abs(q4m_arr[5:])**2)
        
        w4=np.real(q4m_arr[4])
        w4*=self.wignerw4[0]*np.abs(q4m_arr[4])**2 + w4_msum
        w4+=12*self.wignerw4[5]*np.real(np.conj(q4m_arr[8])*q4m_arr[5]*q4m_arr[7])
        w4-=12*self.wignerw4[6]*np.real(np.conj(q4m_arr[7])*q4m_arr[5]*q4m_arr[6])
        w4+= 6*self.wignerw4[7]*np.real(np.conj(q4m_arr[8])*q4m_arr[6]*q4m_arr[6])
        w4+= 6*self.wignerw4[8]*np.real(np.conj(q4m_arr[6])*q4m_arr[5]*q4m_arr[5])
        
        q4=np.sqrt(np.sum(np.abs(q4m_arr**2)))
        result[0]=w4/q4**3
        
        #w6
        q6m_arr=qlm_array[self.idx_qlm[2]]
        w6_msum=6*np.sum(self.wignerw6[1:7]*np.abs(q6m_arr[7:])**2)
        
        w6=np.real(q6m_arr[6])
        w6*=self.wignerw6[0]*np.abs(q6m_arr[6])**2 + w6_msum
        w6+= 12*self.wignerw6[7]*np.real(np.conj(q6m_arr[12])*q6m_arr[7]*q6m_arr[11])
        w6+= 12*self.wignerw6[8]*np.real(np.conj(q6m_arr[12])*q6m_arr[8]*q6m_arr[10])
        w6+= 12*self.wignerw6[9]*np.real(np.conj(q6m_arr[10])*q6m_arr[7]*q6m_arr[9])
        w6-=12*self.wignerw6[10]*np.real(np.conj(q6m_arr[11])*q6m_arr[7]*q6m_arr[10])
        w6-=12*self.wignerw6[11]*np.real(np.conj(q6m_arr[11])*q6m_arr[8]*q6m_arr[9])
        w6-=12*self.wignerw6[12]*np.real(np.conj(q6m_arr[9])*q6m_arr[7]*q6m_arr[8])
        w6+= 6*self.wignerw6[13]*np.real(np.conj(q6m_arr[12])*q6m_arr[9]*q6m_arr[9])
        w6+= 6*self.wignerw6[14]*np.real(np.conj(q6m_arr[10])*q6m_arr[8]*q6m_arr[8])
        w6+= 6*self.wignerw6[15]*np.real(np.conj(q6m_arr[8])*q6m_arr[7]*q6m_arr[7])
        
        q6=np.sqrt(np.sum(np.abs(q6m_arr**2)))
        result[1]=w6/q6**3
        
        return result
        
    def calc_bond_angles(self, i, neighbor_combs):
        "calculates bond angles between neighbors and a center point"
        r_1 = self.datapoints[i] - self.datapoints[neighbor_combs[:, 0]]
        norm_r1 = np.linalg.norm(r_1, axis=1)
    
        r_2 = self.datapoints[i] - self.datapoints[neighbor_combs[:, 1]]
        norm_r2 = np.linalg.norm(r_2, axis=1)
    
        #returns cos(theta)
        return np.einsum('ij,ij->i', r_1, r_2)/(norm_r1*norm_r2)
    
    def calc_ql_from_qlm_array(self,qlm_array):
        "calculates the final ql (over all m) from qlm data"
        result = np.zeros(len(self.l_vec), dtype=np.double)
        for i,idx in enumerate(self.idx_qlm):
            result[i] = np.sqrt(4*np.pi/len(qlm_array[idx])*np.sum(np.abs(qlm_array[idx]**2)))
        return result
        
        
    def calc_inner_outer_indices(self):
        self.indices = np.arange(0, len(self.datapoints), dtype=np.int32)
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
        
    def get_cleaned_signature(self):
        "remove rows with index == -1"
        return self.sign_array[np.invert(self.sign_array[:, 0] == -1.)]
    
    def set_solid_bool_vec(self,bool_vec):
        self.solid_bool=bool_vec


def calc_voro_norm_vec(voro_simplices, voro_points, datapoint):
    "calculates the normal vectors on voronoi facets facing outside of the vol"
    coord_a = voro_points[voro_simplices[:, 0], :]
    coord_b = voro_points[voro_simplices[:, 1], :]
    coord_c = voro_points[voro_simplices[:, 2], :]
    vec_1 = coord_b - coord_a
    vec_2 = coord_c - coord_a
    r_center = 1./3. * (coord_a + coord_b + coord_c)
    norm_vec = np.cross(vec_1, vec_2)

    flip_norm_vec = np.linalg.norm(r_center - datapoint + norm_vec/10**4, axis=1) < \
                    np.linalg.norm(r_center- datapoint, axis=1)
    norm_vec[flip_norm_vec] *= -1

    return norm_vec

def calc_minkowski_eigenvalues(voro_area,voro_areas,normvecs):
    tensor=np.zeros((6,6),dtype=np.float64)
#    normvecs[0]*=2.
#    normvecs[1]*=3.
#    normvecs[2]*=4.
    sqrt2=1.41421356237
    for (a,n) in zip(voro_areas,normvecs):
        tensor[0,0]+=a*n[0]*n[0]*n[0]*n[0]
        tensor[0,1]+=a*n[0]*n[0]*n[1]*n[1]
        tensor[0,2]+=a*n[0]*n[0]*n[2]*n[2]
        tensor[0,3]+=sqrt2*a*n[0]*n[0]*n[1]*n[2]
        tensor[0,4]+=sqrt2*a*n[0]*n[0]*n[2]*n[0]
        tensor[0,5]+=sqrt2*a*n[0]*n[0]*n[0]*n[1]
        
        tensor[1,1]+=a*n[1]*n[1]*n[1]*n[1]
        tensor[1,2]+=a*n[1]*n[1]*n[2]*n[2]
        tensor[1,3]+=sqrt2*a*n[1]*n[1]*n[1]*n[2]
        tensor[1,4]+=sqrt2*a*n[1]*n[1]*n[2]*n[0]
        tensor[1,5]+=sqrt2*a*n[1]*n[1]*n[0]*n[1]
        
        tensor[2,2]+=a*n[2]*n[2]*n[2]*n[2]
        tensor[2,3]+=sqrt2*a*n[2]*n[2]*n[1]*n[2]
        tensor[2,4]+=sqrt2*a*n[2]*n[2]*n[2]*n[0]
        tensor[2,5]+=sqrt2*a*n[2]*n[2]*n[0]*n[1]
        
        tensor[3,3]+=2*a*n[1]*n[2]*n[1]*n[2]
        tensor[3,4]+=2*a*n[1]*n[2]*n[2]*n[0]
        tensor[3,5]+=2*a*n[1]*n[2]*n[0]*n[1]
        
        tensor[4,4]+=2*a*n[2]*n[0]*n[2]*n[0]
        tensor[4,4]+=2*a*n[2]*n[0]*n[0]*n[1]
        
        tensor[5,5]+=2*a*n[0]*n[1]*n[0]*n[1]
        
#        tensor += np.array(
#                    [[n[0]*n[0]*n[0]*n[0],n[0]*n[0]*n[1]*n[1],n[0]*n[0]*n[2]*n[2],n[0]*n[0]*n[1]*n[2],n[0]*n[0]*n[0]*n[2],n[0]*n[0]*n[0]*n[1]],
#                     [0.00000000000000000,n[1]*n[1]*n[1]*n[1],n[1]*n[1]*n[2]*n[2],n[1]*n[1]*n[1]*n[2],n[1]*n[1]*n[0]*n[2],n[1]*n[1]*n[0]*n[1]],
#                     [0.00000000000000000,0.00000000000000000,n[2]*n[2]*n[2]*n[2],n[2]*n[2]*n[1]*n[2],n[2]*n[2]*n[0]*n[2],n[2]*n[2]*n[0]*n[1]],
#                     [0.00000000000000000,0.00000000000000000,0.00000000000000000,n[1]*n[2]*n[1]*n[2],n[1]*n[2]*n[0]*n[2],n[1]*n[2]*n[0]*n[1]],
#                     [0.00000000000000000,0.00000000000000000,0.00000000000000000,0.00000000000000000,n[0]*n[2]*n[0]*n[2],n[0]*n[2]*n[0]*n[1]],
#                     [0.00000000000000000,0.00000000000000000,0.00000000000000000,0.00000000000000000,0.00000000000000000,n[0]*n[1]*n[0]*n[1]]],
#                     dtype=np.float64)*facet_area
    tensor/=voro_area
    eigenvalues=np.linalg.eigvalsh(tensor,UPLO='U')
    
#    if len(voro_areas)==24:
#        print(eigenvalues)
    
    return eigenvalues


def calc_voro_area_angle(args):
#    return fastMSM.calc_voro_area_angle(len(voro_hull.simplices),
#                                       voro_hull.equations[:,0:3],
#                                       voro_hull.simplices,
#                                       voro_hull.points)
    return fastMSM.calc_voro_area_angle(args[0],args[1],args[2],args[3])


def calc_qlm_from_voro(args, max_l, l_vec):
    "directly calculate qlm data for a voronoi cell"
    
    #norm_vecs=calc_voro_norm_vec(voro_hull.simplices,voro_hull.points)

#    return fastMSM.calc_msm_qlm(max_l,len(l_vec),l_vec,
#                                voro_area_angle.shape[0],
#                                voro_area_angle[:,2],
#                                voro_area_angle[:,1],
#                                voro_hull.area, voro_area_angle[:,0])
    return fastMSM.calc_msm_qlm(max_l,len(l_vec),l_vec,args[0],
                                args[1],args[2],args[3],args[4])

def calc_distances(neighborpoints):
    "calculates distances between all neighbors"
    r21 = neighborpoints[:, 1] - neighborpoints[:, 0]
    return np.linalg.norm(r21, axis=1)
    
def calc_w_from_qlm_array(qlm_arr,l,wignerlist):
    w=0+0*1j
    for [w3j,m1,m2,m3] in wignerlist:
        w+=w3j*(qlm_arr[m1+l]*qlm_arr[m2+l]*qlm_arr[m3+l])
    norm=np.sqrt(np.sum(np.abs(qlm_arr)**2))**3
    w=w/norm
    return np.real_if_close(w)

def calc_w4_w6_from_qlm_array_new(qlm_array,wignerw4,wignerw6):
    #W4
    w4=calc_w_from_qlm_array(qlm_array[1],4,wignerw4)
    w6=calc_w_from_qlm_array(qlm_array[2],6,wignerw6)
    return np.array([w4,w6])

def calc_particle_signature(datalist,sign_dimension, idx_qlm, si_threshold,
                                nbins_distances, l_vec, wignerw4, wignerw6):
    signature=np.zeros(sign_dimension,dtype=np.double)-1
    #i=datalist[0]
    voro_neighbors=datalist[0]
    qlm_array=datalist[1]
    datapoint=datalist[2]
    neighborpoints=datalist[3]
    voro_area=datalist[4]
    voro_areas=datalist[5]
    normvecs=datalist[6]
    
    

    num_neighbors = len(voro_neighbors)
    voro_distances = np.sqrt(np.sum((datapoint - neighborpoints)**2, axis=1))
    sorting_indices = np.argsort(voro_distances)
    sorted_distances = voro_distances[sorting_indices]
    r0=np.mean(sorted_distances[0:6])
    
    comb_array=np.array(list(it.combinations(range(num_neighbors),2)),dtype=np.int32)
    neighbor_comb = neighborpoints[comb_array]

    cos_theta = fastMSM.calc_bond_angles(datapoint,
                                         neighbor_comb[:, 0],
                                         neighbor_comb[:, 1])
    
    distances = calc_distances(neighbor_comb)/r0
    
    mind=min(distances)
    maxd=max(distances)
    #mind=0.5
    #maxd=3.0
    dist_edges=np.linspace(mind,maxd,nbins_distances+1,endpoint=True)
    #print('dists:',dist_edges)
    dist_hist = fast_hist(distances, dist_edges)
    bond_hist = fast_hist(cos_theta, [-1.0, -0.945, -0.915, -0.755,
                                         -0.195, 0.195, 0.245, 0.795, 1.0])

    ql_vec = fastMSM.calc_ql_from_qlm_array(l_vec,qlm_array)
    
    q4m_arr=qlm_array[idx_qlm[1]]
    q6m_arr=qlm_array[idx_qlm[2]]
    #wl_vec = fastMSM.calc_w4_w6_from_qlm_array(q4m_arr,wignerw4,
    #                                           q6m_arr,wignerw6)
    q2m_arr=qlm_array[idx_qlm[0]]
    testqlm=[q2m_arr,q4m_arr,q6m_arr]
    wl_vec=calc_w4_w6_from_qlm_array_new(testqlm,wignerw4,wignerw6)
    
    eigenvals=calc_minkowski_eigenvalues(voro_area,voro_areas,normvecs)
    
    #signature[0] = i
    signature[0] = num_neighbors
    signature[1:9] = bond_hist
    signature[9:11] = ql_vec[1:3]
    signature[11:13] = wl_vec
    signature[13:19] = eigenvals
    signature[19:19+nbins_distances] = dist_hist

    return signature

def fast_hist(data, bin_edges):
    return np.bincount(np.digitize(data, bin_edges[1:-1]), minlength=len(bin_edges) - 1)
        
if __name__ == '__main__':
    
    lim=15
    ranges=[]
    
    datapoints=gc.fill_volume_fcc(lim/3, lim, lim)
    ranges.append(range(len(datapoints)))
    
    hcp=gc.fill_volume_hcp(lim/3, lim, lim)
    hcp[:,0]+=lim/3+0.5
    datapoints=np.vstack((datapoints,hcp))
    ranges.append(range(ranges[0][-1]+1,len(datapoints)))
    
    bcc=gc.fill_volume_bcc(lim/3, lim, lim)
    bcc[:,0]+=2*(lim/3+0.5)
    datapoints=np.vstack((datapoints,bcc))
    ranges.append(range(ranges[1][-1]+1,len(datapoints)))
    
    volume=[[2,datapoints.max(axis=0)[0]-2],[2,datapoints.max(axis=0)[1]-2],[2,datapoints.max(axis=0)[2]-2]]
    
    datapoints=add_gaussian_noise(datapoints,0.03,1)
    
    
    
#    num_atoms=13500
#    timesteps=np.array([10000000])
#    datapoints=asim.get_dumpfile_datapoints('../crystalanalysis/pc_17112016_9h31m_10Pa.dump'
#                                            ,num_atoms,timesteps)[0]
#    
#    distance=2*53.
#    x_min, x_max = np.min(datapoints[:,0]), np.max(datapoints[:,0])
#    y_min, y_max = np.min(datapoints[:,1]), np.max(datapoints[:,1])
#    z_min, z_max = np.min(datapoints[:,2]), np.max(datapoints[:,2])
#    volume = [[x_min+distance,x_max-distance],[y_min+distance,y_max-distance],[z_min+distance,z_max-distance]]
    

    t=time.time()
    Signature=MCS(datapoints,n_proc=1)
    Signature.set_inner_volume(volume)
    Signature.calc_qlm_array()
    Signature.calc_si_bool()
    sign=Signature.calc_sign_array()
    
    Signature.close_proc()
    
    fcc_bool=np.array([sign[i,0] in ranges[0] for i in range(len(sign))],dtype=np.bool)
    hcp_bool=np.array([sign[i,0] in ranges[1] for i in range(len(sign))],dtype=np.bool)
    bcc_bool=np.array([sign[i,0] in ranges[2] for i in range(len(sign))],dtype=np.bool)
    
    print(np.mean(sign[fcc_bool,10:14],axis=0))
    print(np.mean(sign[hcp_bool,10:14],axis=0))
    print(np.mean(sign[bcc_bool,10:14],axis=0))
    
    plt.figure()
    y_idx=13
    x_idx=12
    plt.scatter(sign[fcc_bool,x_idx],sign[fcc_bool,y_idx],color='r')
    plt.scatter(sign[hcp_bool,x_idx],sign[hcp_bool,y_idx],color='g')
    plt.scatter(sign[bcc_bool,x_idx],sign[bcc_bool,y_idx],color='b')
    
    plt.figure()
    plt.hist2d(sign[:,x_idx],sign[:,y_idx],bins=40)

#    stds=np.std(sign[:,1:],axis=0)
#    plt.figure()
#    plt.plot(np.arange(sign.shape[1]-1),stds)
    fcc_idx=sign[fcc_bool,0].astype(np.int32)
    hcp_idx=sign[hcp_bool,0].astype(np.int32)
    bcc_idx=sign[bcc_bool,0].astype(np.int32)
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(datapoints[fcc_idx,0],datapoints[fcc_idx,1],datapoints[fcc_idx,2])
    ax.scatter(datapoints[hcp_idx,0],datapoints[hcp_idx,1],datapoints[hcp_idx,2])
    ax.scatter(datapoints[bcc_idx,0],datapoints[bcc_idx,1],datapoints[bcc_idx,2])
    #
    #sign=Signature.calc_particle_signature(10427)
    #clean=st.cleanup_signature(sign)
    #print(clean[:,10:12])
    #test=Signature.qlm_arrays
    print(time.time()-t)
    

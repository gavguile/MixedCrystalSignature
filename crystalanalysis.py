#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:51:08 2017

@author: dietz
"""

import numpy as np
import time

import datageneration.generatecrystaldata as gcn
import datageneration.disordercrystaldata as dcs

from sklearn.metrics import accuracy_score
import pickle

class CrystalAnalyzer:
    """small helper class to train and test the classifiers using artifical crystal data"""
    
    STRUCTURES=['fcc','hcp','bcc']
    LABELS2STRUCT={'fcc':1,'hcp':2,'bcc':3} 
    
    def __init__(self, classifier, scaler, sign_calculator, train_seed=0, test_seed=0,
                 train_noiselist=list(range(4,12,1)), noiselist=list(range(0,21)),
                 volume=[15,15,15], inner_distance=2, loglevel=1):
        self.classifier=classifier
        self.scaler=scaler
        self.sign_calculator=sign_calculator
        
        self.train_seed=train_seed
        self.test_seed=test_seed
        self.train_noiselist=train_noiselist
        self.noiselist=noiselist
        self.volume=volume
        self.inner_distance=inner_distance
        self.loglevel=loglevel
        
    def create_artificial_datasets(self, noise_arr, structure_arr, volume, rnd_seed):
        datasets=dict()
        for structure in structure_arr:
            datasets[structure]={'noise':[],'datalist':[]}
            basedata=[]
            if 'fcc' == structure:
                basedata=gcn.fill_volume_fcc(volume[0], volume[1], volume[2])
            if 'hcp' == structure:
                basedata=gcn.fill_volume_hcp(volume[0], volume[1], volume[2])
            if 'bcc' == structure:
                basedata=gcn.fill_volume_bcc(volume[0], volume[1], volume[2])
            
            for noise in noise_arr:
                datasets[structure]['datalist'].append(dcs.add_gaussian_noise(basedata, noise/100, rnd_seed))
                datasets[structure]['noise'].append(noise)
        return datasets
    
    def calculate_artificial_signatures(self, datasets):
        signatures=dict()
        for structure in datasets:
            signatures[structure]={'sign_arr':[],'voro_vols':[], 'softness':[], 'data_idx':[]}
            initial_datapoints=datasets[structure]['datalist'][0]
            inner_bool_vec=self.get_inner_volume_bool_vec(initial_datapoints)
            for i,datapoints in enumerate(datasets[structure]['datalist']):
                self.sign_calculator.set_datapoints(datapoints)
                self.sign_calculator.set_inner_bool_vec(inner_bool_vec)
                self.sign_calculator.calc_signature()
                signatures[structure]['sign_arr'].append(self.sign_calculator.signature)
                signatures[structure]['voro_vols'].append(self.sign_calculator.voro_vols)
                signatures[structure]['softness'].append(self.sign_calculator.struct_order)
                signatures[structure]['data_idx'].append(self.sign_calculator.solid_indices)
                if self.loglevel >= 3:
                    print('struc:',structure,
                          'noise:',datasets[structure]['noise'][i],
                          'num:', len(self.sign_calculator.signature))
        return signatures
    
    def convert_artificial_signatures_to_matrix(self,signatures):
        index_dict=dict()
        startindex=0
        signlist=[]
        labels=[]
        
        for structure in signatures:
            index_dict[structure]=[]
            for i,sign_arr in enumerate(signatures[structure]['sign_arr']):
                if i in self.train_noiselist:
                    length=sign_arr.shape[0]
                    if length>0:
                        signlist.append(sign_arr)
                        data_idx=signatures[structure]['data_idx'][0]
                        index_dict[structure].append([i,range(startindex,startindex+length),data_idx])
                        labels.append(np.zeros(length,dtype=np.int32)+self.LABELS2STRUCT[structure])
                        startindex+=length

        return np.concatenate(signlist,axis=0), np.concatenate(labels), index_dict
    
    def generate_train_signatures(self):
        if self.loglevel >= 1:
            print('generating training signatures')
        self.train_datasets=self.create_artificial_datasets(self.noiselist,
                                                            self.STRUCTURES,
                                                            self.volume,
                                                            self.train_seed)
        self.train_signatures=self.calculate_artificial_signatures(self.train_datasets)
        if self.loglevel >= 1:
            print('finished')
        
    def generate_test_signatures(self):
        if self.loglevel >= 1:
            print('generating test signatures')
        self.test_datasets=self.create_artificial_datasets(self.noiselist,
                                                            self.STRUCTURES,
                                                            self.volume,
                                                            self.test_seed)
        self.test_signatures=self.calculate_artificial_signatures(self.test_datasets)
        if self.loglevel >= 1:
            print('finished')
    
    def train_classifier(self):
        if self.loglevel >=1:
            print('started training')
            t = time.time()
        
        self.trainmatrix, self.trainlabels, self.trainindex_dict=self.convert_artificial_signatures_to_matrix(self.train_signatures)
        
        self.classifier.fit(self.scaler.fit_transform(self.trainmatrix),self.trainlabels)
        
        if self.loglevel >=1:
            print('finished training, time:',time.time()-t)
        
        if self.loglevel >=1:
            trainprediction=self.classifier.predict(self.scaler.transform(self.trainmatrix))
            print('Accuracy on Train set:',
                  accuracy_score(trainprediction,self.trainlabels))
        
    def predict_test(self):
        for structure in self.test_signatures:
            self.test_signatures[structure]['prediction']=[]
            for i,sign_arr in enumerate(self.test_signatures[structure]['sign_arr']):
                if len(sign_arr)>0:
                    pred=self.classifier.predict(self.scaler.transform(sign_arr))
                else:
                    pred=np.array([],dtype=np.int32)
                self.test_signatures[structure]['prediction'].append(pred)
                if self.loglevel >=1:
                    orig_labels=np.zeros(len(pred),dtype=np.int32)+self.LABELS2STRUCT[structure]
                    print('predicted','struc',structure,'idx', i,
                          accuracy_score(pred,orig_labels))

    def get_inner_volume_bool_vec(self,datapoints):            
        x_min = 0+self.inner_distance
        y_min = 0+self.inner_distance
        z_min = 0+self.inner_distance
        
        x_max = self.volume[0]-self.inner_distance
        y_max = self.volume[1]-self.inner_distance
        z_max = self.volume[2]-self.inner_distance
        
        bool_matrix_min = datapoints >= [x_min,y_min,z_min]
        bool_matrix_max = datapoints <= [x_max,y_max,z_max]
        
        bool_matrix=np.logical_and(bool_matrix_min,bool_matrix_max)
        
        return np.all(bool_matrix,axis=1)
    
    def load_object(self, filepath):
        objects=[]
        with open(filepath,'rb') as file:
            objects=pickle.load(file)
        return objects
        
    def save_object(self,objects, filepath):
        with open(filepath,'wb') as file:
            pickle.dump(objects,file)
            
    def save_training_signatures(self,path):
        self.save_object(self.train_signatures,path)
        
    def load_training_signatures(self,path):
        self.train_signatures=self.load_object(path)
        
    def save_test_signatures(self,path):
        self.save_object(self.test_signatures,path)
        
    def load_test_signatures(self,path):
        self.test_signatures=self.load_object(path)
    
    def save_classifier(self,path):
        self.save_object(self.classifier,path)
    
    def load_classifier(self,path):
        self.classifier=self.load_object(path)
        
    def save_scaler(self,path):
        self.save_object(self.scaler,path)
    
    def load_scaler(self,path):
        self.scaler=self.load_object(path)
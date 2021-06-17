#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:45:13 2021

Script to contain classification decision trees

@author: mattard
"""

#imports
from abstract.base_tree import DecisionTree
import numpy as np
from scipy import stats

#Decision Tree Classifier
class DecisionTreeClassifier(DecisionTree):
    #initializer
    def __init__(self,max_depth=None,min_samples_split=2,loss='gini',balance_class_weights=False):
        super().__init__(max_depth,min_samples_split)
        self.loss                  = loss   
        self.balance_class_weights = balance_class_weights
        self.class_weights         = None
    
    #private function to define the gini impurity
    def __gini(self,D):
        #initialize the output
        G = 0
        #iterrate through the unique classes
        for c,w in zip(np.unique(D[:,-1]),self.class_weights):
            #compute p for the current c
            p = w*D[D[:,-1]==c].shape[0]/D.shape[0]
            #compute term for the current c
            G += p*(1-p)
        #return gini impurity
        return(G)
    
    #private function to define the shannon entropy
    def __entropy(self,D):
        #initialize the output
        H = 0
        #iterrate through the unique classes
        for c,w in zip(np.unique(D[:,-1]),self.class_weights):
            #compute p for the current c
            p = w*D[D[:,-1]==c].shape[0]/D.shape[0]
            #compute term for the current c
            H -= p*np.log2(p)
        #return entropy
        return(H)
    
    #protected function to define the impurity
    def _impurity(self,D):
        #use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'gini':
            ip = self.__gini(D)
        elif self.loss == 'entropy':
            ip = self.__entropy(D)
        #return results
        return(ip)
    
    #protected function to compute the value at a leaf node
    def _leaf_value(self,D):
         return(stats.mode(D[:,-1])[0])
     
    #public function to return model parameters
    def get_params(self,deep=False):
        return{'max_depth':self.max_depth,
               'min_samples_split':self.min_samples_split,
               'loss':self.loss,
               'balance_class_weights':self.balance_class_weights}
    
    #train the tree model
    def fit(self,Xin,Yin):
        #check if class weights need to be computed?
        if self.balance_class_weights:
            self.class_weights = Yin.shape[0]/(np.unique(Yin).shape[0]*np.bincount(Yin.flatten().astype(int)))
        else:
            self.class_weights = np.ones(np.unique(Yin).shape[0])
        #call the base fit function
        super().fit(Xin, Yin)
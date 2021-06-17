#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:01:58 2021

Initialization file for decisiontrees package

@author: mattard
"""
## imports ##
import os
import sys

## set path to package ##
sys.path.append(os.getcwd()+'/decisiontrees')

## import relevant functionality ##
from treeclassifier import *
from treeregressor import *
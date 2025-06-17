#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:59:15 2025

@author: ugurburak
"""

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
model = LogisticRegression(max_iter=1000)
rfe = RFE(model,n_features_to_select=1)
rfe = rfe.fit(iris.data, iris.target)

print(rfe.support_)
print(rfe.ranking_)
# %%


import numpy as np
from sklearn.feature_selection import chi2

X = np.array([[1,1,3],
             [0,1,5],
             [5,4,1],
             [6,6,2],
             [1,4,0],
             [0,0,0]])

y = np.array([1,1,0,0,2,2])
chi2_stats,p_values = chi2(X,y)
print(chi2_stats)
print(p_values)
# %%

from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
X,y = load_iris(return_X_y= True)

f_statistics, p_values = f_classif(X, y)
print(f_statistics)
print(p_values)
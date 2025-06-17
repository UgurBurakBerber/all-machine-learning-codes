#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:14:58 2025

@author: ugurburak

Polinom Regresyon Ornek 2
"""

import numpy as np 
import matplotlib.pyplot as plt 

x = np.array([6,9,12,16,22,28,33,40,47,51,55,60])
y = np.array([14,28,50,64,67,57,55,57,68,74,88,60])

plt.figure(1)
plt.scatter(x, y, color = 'red',marker='o')

from sklearn.preprocessing import PolynomialFeatures
model_Polinom_Regresyon = PolynomialFeatures(degree=3)
X_polinom = model_Polinom_Regresyon.fit_transform(x.reshape(-1, 1))

from sklearn.linear_model import LinearRegression
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_polinom, y)

print(model_Regresyon.coef_)
print(model_Regresyon.intercept_)
# %%

import statsmodels.api as sm
model_Regresyon_OLS = sm.OLS(y, X_polinom).fit()
print(model_Regresyon_OLS.summary())
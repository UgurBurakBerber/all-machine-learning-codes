#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 00:30:04 2025

@author: ugurburak

Çoklu Doğrusal Regresyon Örnek1
"""

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/Pv.xlsx')

X = veriSeti.iloc[:,:1].values
y = veriSeti.iloc[:,-1].values

uzunluk = len(X)

X = np.append(arr=np.ones((uzunluk,1)).astype(int),values=X,axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=3)

from sklearn.linear_model import LinearRegression
model_Regresyon =LinearRegression()
model_Regresyon.fit(X_train, y_train)

print(model_Regresyon.coef_,model_Regresyon.intercept_)
# %%

import statsmodels.api as sm
model_Regresyon_OLS = sm.OLS(y_train,X_train).fit()
print(model_Regresyon_OLS.summary())

# %%

xyeni = X_train[:,[0,1,2,4,5,6,7]]
model_Regresyon_OLS = sm.OLS(y_train,xyeni).fit()
print(model_Regresyon_OLS.summary())

# %%

xopt = X_train[:,[0,1,2,4,5,6,7]]
model_Regresyon_OLS = sm.OLS(y_train, xopt).fit()
print(model_Regresyon_OLS.summary())























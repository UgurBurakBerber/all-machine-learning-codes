#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:26:18 2025

@author: ugurburak
"""

#Kategorik Verilerin Dönüşümü

import pandas as pd
from sklearn.preprocessing import LabelEncoder

veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/VeriOnIsleme_2.xlsx')

X = veriSeti.iloc[:,:-1].values
y = veriSeti.iloc[:,-1].values

LabelEncoder_X = LabelEncoder()
X[:,0] = LabelEncoder_X.fit_transform(X[:,0])
print(X)
# %%

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/VeriOnIsleme_2.xlsx')

X = veriSeti.iloc[:,:-1].values
y = veriSeti.iloc[:,-1].values

ColumnTransformer_X = ColumnTransformer(([('endercoder',OneHotEncoder(),[0])]),remainder='passthrough')

X = ColumnTransformer_X.fit_transform(X)

print(X)

# %%

# Beyin bağımlı değişken tahmin etme 

import numpy as np 
import pandas as pd 

veriSeti = pd.read_csv('/Users/ugurburak/Desktop/Machine Learning/veri setleri/DogrusalRegresyon.csv')

X = veriSeti['Bas_cevresi(cm^3)'].values
y = veriSeti['Beyin_agirligi(gr)'].values

uzunluk = len(X)
X = X.reshape((uzunluk,1))
X = np.append(arr = np.ones((uzunluk,1)).astype(int),values = X,axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
modelRegresyon = LinearRegression()
modelRegresyon.fit(X_train,y_train)

print(modelRegresyon.coef_)
print(modelRegresyon.intercept_)
# %%

import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter(X_train[:,1], y_train,color= "red",marker="o")
pred_X_train = (modelRegresyon.intercept_) + (modelRegresyon.coef_[1])*X_train[:,1]
plt.plot(X_train[:,1].pred_X_train,color = "blue")


plt.figure(2)
plt.scatter(X_test[:,1], y_test,color= "red",marker="o")
pred_X_train = (modelRegresyon.intercept_) + (modelRegresyon.coef_[1])*X_test[:,1]
plt.plot(X_test[:,1],pred_X_test,color = "blue")

# %%

from sklearn.metrics import mean_absolute_error,r2_score

print(mean_absolute_error(y_test, pred_X_test))
print(r2_score(y_test,pred_X_test))

pred_X_train = modelRegresyon.predict(X_train)
print(r2_score(y_train, pred_X_train))

# %%

import statsmodels.api as sm
modelRegresyon_OLS = sm.OLS(y_train,X_train).fit()
modelRegresyon_OLS.summary()

# %%

"""
X = [3200,4500,3879] için y_ tahmini=?
"""
Xyeni = np.array([[1,3200],[1,4500],[1,3879]])
y_tahmin = modelRegresyon.predict(Xyeni)
print(y_tahmin)
y_tahmin1 = 354.8403+0.2553*3200
print(y_tahmin1)

# %%

import numpy as np 

X =np.array([8,10,12,14,16])
y= np.array([20,24,25,26,30])

uzunluk = len(X)
X=X.reshape((uzunluk, 1))

X = np.append(arr = np.ones((uzunluk,1)).astype(int),values=X,axis=1)

from sklearn.linear_model import LinearRegression
modelRegresyon = LinearRegression()
modelRegresyon.fit(X, y)

print(modelRegresyon.coef_)
print(modelRegresyon.intercept_)

from sklearn.metrics import r2_score
y_tahmin = modelRegresyon.predict(X)

R2 = r2_score(y, y_tahmin)
print(R2)

"""
X = (13,15,20) için y_tahmin=?
 """

Xyeni = np.array([[1,13],[1,15],[1,20]])

y_tahmin = modelRegresyon.predict(Xyeni)
print(y_tahmin)













#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 23:47:36 2025

@author: ugurburak
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)

def fonksiyon(x):
    return 10*np.sinc(x/2)+x/7+6

X= np.random.rand(100)*10
y=fonksiyon(X)+2*np.random.randn(*X.shape)

uzunluk = len(X)
X=X.reshape(uzunluk,1)

plt.figaspect(1)
plt.scatter(X, y, color = "red")


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
model_Polinom_Regresyon = PolynomialFeatures(degree=3)
X_polinom = model_Polinom_Regresyon.fit_transform(X_train)
from sklearn.linear_model import LinearRegression
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_polinom,y_train)

print(model_Regresyon.coef_)
print(model_Regresyon.intercept_)
# %%

import statsmodels.api as sm
model_Regresyon_OLS =sm.OLS(y_train, X_polinom).fit()
print(model_Regresyon_OLS.summary())
# %%

Xgrid = np.arange(min(X_train),max(X_train),0.1)
Xgrid = Xgrid.reshape(-1, 1)
ypred= model_Regresyon.predict(model_Polinom_Regresyon.fit_transform(Xgrid))
plt.plot(Xgrid, ypred, color = 'blue')
# %%

from sklearn.metrics import r2_score
ytahmin = model_Regresyon.predict(X_polinom)
print(r2_score(y_train, ytahmin))
# %%

from sklearn.metrics import r2_score
X_test = model_Polinom_Regresyon.fit_transform(X_test)
y_test_tahmin = model_Regresyon.predict(X_test)
print(r2_score(y_test, y_test_tahmin))
# %%

"""
X = [1,6,5,4,5]
"""
X_yeni = np.array([1,6,5,4,5])
X_yeni = X_yeni.reshape(-1, 1)
X_yeni = model_Polinom_Regresyon.fit_transform(X_yeni)
print(model_Regresyon.predict(X_yeni))
































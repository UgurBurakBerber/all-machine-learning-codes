#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:36:40 2025

@author: ugurburak
"""

# Polinomal Regresyon

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#2. veri_ onisleme
#2.1. veri yükleme
veriler = pd.read_csv('/Users/ugurburak/Desktop/python/btk machine learning/maaslar.csv')



# data frame dilimleme (slice)
x = veriler.iloc[:,1:2] # eğitim seviyesini x olarak aldık
y = veriler.iloc[:,2:] # maas seviyesini y olarak aldık


#numpy array -> dizi dönüşümü
X = x.values
Y = y.values



#Linear Regression -> doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



# plynomial regression -> doğrusal olmayan (nonlinear model) model oluşturma
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) # 2. dereceden bir polinom features oluştur.
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)



# 4. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree=4) # 2. dereceden bir polinom features oluştur.
x_poly = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly, y)



# Görselleştirme
plt.scatter(X,Y, color = 'red')
plt.plot(x, lin_reg.predict(X), color= 'blue')
plt.show()
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.show()
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color= 'blue')
plt.show()



# Tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

#verilerin ölçeklenmesi
# SVR Destek Vektörleri Regresyon ile Tahmin Vektörleri
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1, 1)))    

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli,color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color='blue')

plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y,color='red')
plt.plot(X, r_dt.predict(X),color='blue')
plt.plot(X,r_dt.predict(Z),color='green')
plt.plot(X,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
plt.show()


# Rassal Ağaçlar (Random Forest)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators= 10, random_state=0)
rf_reg.fit(X, Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(X,r_dt.predict(K),color='yellow')























































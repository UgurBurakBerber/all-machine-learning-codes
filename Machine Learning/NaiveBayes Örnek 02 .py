#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:13:13 2025

@author: ugurburak
"""

# Naive Bayes Örnek 02 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X=iris.data[:,2:]
y=iris.target


from sklearn.naive_bayes import GaussianNB

siniflandirici = GaussianNB()
siniflandirici.fit(X,y)
y_tahmin=siniflandirici.predict(X)
# %%

from sklearn.metrics import confusion_matrix, classification_report
hm = confusion_matrix(y, y_tahmin)
sr=classification_report(y, y_tahmin)
print(hm,sr)
# %%

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.linspace(x_min, x_max,300),np.linspace(y_min, y_max,300))

Z = siniflandirici.predict(np.c_[xx.ravel(),yy.ravel()])

#Karar Sınırları
Z = Z.reshape(xx.shape)
plt.contour(xx,yy,Z,alpha=0.3,cmap=plt.cm.Set1)
#plt.scatter(X[:,0], X[:,1],c=y,edgecolors='k',cmap=plt.cm.Set1)
plt.scatter(X[y==0,0],X[y==0,1], color = 'g',edgecolors='k',label='Setosa',s=100)
plt.scatter(X[y==1,0],X[y==1,1], color = 'g',edgecolors='k',label='Versicolor',s=100)
plt.scatter(X[y==2,0],X[y==2,1], color = 'g',edgecolors='k',label='Virginica',s=100)
plt.legend()

plt.title("İris Veri Seti - Gaussian Naive Bayes- Karar Sınırları")
plt.xlabel("Sepal Length",fontsize=14)
plt.ylabel("Sepal Width",fontsize=14)
plt.show()
# %%

farkli_indeksler= np.where(y != y_tahmin)[0]
print('Hatalı tahmin edilen indeksler: ',farkli_indeksler)

for i in farkli_indeksler:
    plt.text(X[i,0]+0.05,X[i,1],str(i),fontsize=9,color='black')
indeksler= np.where(np.all(X==X[70],axis=1))[0]


















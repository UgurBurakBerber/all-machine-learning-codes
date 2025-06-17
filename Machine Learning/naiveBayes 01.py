#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:48:03 2025

@author: ugurburak
"""
# Naive Bayes Örnek 01

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X,y = make_classification(n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,n_classes=2,random_state=42)  


from sklearn.naive_bayes import GaussianNB

siniflandirici = GaussianNB()
siniflandirici.fit(X,y)
y_tahmin=siniflandirici.predict(X)
# %%

from sklearn.metrics import confusion_matrix, classification_report
hm = confusion_matrix(y, y_tahmin)
sr=classification_report(y, y_tahmin)
print(hm,sr)

###
from sklearn.metrics import r2_score
print('r2: ',r2_score(y,y_tahmin))
###
# %%

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.linspace(x_min, x_max,300),np.linspace(y_min, y_max,300))

Z = siniflandirici.predict(np.c_[xx.ravel(),yy.ravel()])

#Karar Sınırları
Z = Z.reshape(xx.shape)
plt.contour(xx,yy,Z,alpha=0.4,cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1],c=y,edgecolors='k',cmap=plt.cm.coolwarm)



plt.title("Gaussian Naive Bayes- Karar Sınırları")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.show()



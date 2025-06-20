#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:30:56 2025

@author: ugurburak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.txt')

X = veriler.iloc[:,3:].values # maaş ve hacim

#kmeans

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++') # 3 tane merkez noktası orta noktalar  
kmeans.fit(X)
print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11), sonuclar)    
plt.show()



kmeans = KMeans(n_clusters=4,init='k-means++',random_state=123)
y_tahmin= kmeans.fit(X)
plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='black')
plt.title("kmeans")
plt.show()

# HC
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters =3,metric="euclidean" ,linkage='ward')
y_tahmin = ac.fit_predict(X)
print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='black')
plt.title("HC")
plt.show()


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()












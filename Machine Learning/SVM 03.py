#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:17:38 2025

@author: ugurburak
"""

# SVM Örnek 03

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

np.random.seed(0)
X = np.r_[np.random.randn(10,2)-[2,2],np.random.randn(10,2)+[2,2]]
y = [0]* 10+ [1]*10

model = SVC(kernel='linear', C= 0.01)
model.fit(X,y)

plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c = y,cmap= plt.cm.bwr, edgecolors='k', label='Veri Noktaları')
# %% Hiper Düzlem ve marjinal sınırlar

ax= plt.gca()
xlim=ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
XX,YY = np.meshgrid(yy,xx)
xy= np.vstack([XX.ravel(),YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors= 'black', levels = [-1,0,1],
           alpha = 1, linestyles=['dashed','solid','dashed'])

ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],
          s=200,linewidths=2,facecolors='none',edgecolors='black',
          label= 'Destek Vektörleri')































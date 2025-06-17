#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:06:46 2025

@author: ugurburak
"""

#SVM örnek 02

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10, noise=0.2,random_state=42) #10 yerine 100 yaz diğerlerinini grafikleri için

plt.scatter(X[:,0], X[:,1], c = y, cmap= plt.cm.Paired)

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=42)
# %%
#Linear Kernel

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_tahmin= svm_linear.predict(X_test)
print(f" Linear Kernel Doğruluk : {accuracy_score(y_test, y_tahmin):.4f}")
# %%
#Polinomial Kernel

svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)
y_tahmin= svm_poly.predict(X_test)
print(f" Polynomial Kernel Doğruluk : {accuracy_score(y_test, y_tahmin):.4f}")
# %%

#RBF Kernel (Radial Basis Function)

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_tahmin= svm_rbf.predict(X_test)
print(f" RBF Kernel Doğruluk : {accuracy_score(y_test, y_tahmin):.4f}")
# %%

#Karar Sınırları (sınavda bu kod çıkmaz grafik çıkabilir...)

h = 0.02
x_min, x_max = X[:,0].min()-1,X[:,0].max()+1
y_min, y_max = X[:,0].min()-1,X[:,0].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z= svm_rbf.predict(np.c_[xx.ravel(),yy.ravel()]) #svm_ploy,svm_linear da yazılabilir..
Z = Z.reshape(xx.shape)

ax = plt.gca()
ax.contour(xx, yy, Z, alpha=0.8, cmap= plt.cm.Paired )
ax.scatter(X[:,0], X[:,1], c= y, edgecolors = 'k', s = 30, cmap= plt.cm.Paired)
ax.set_title("RBF KERNEL")
plt.show()
# %%

ax = plt.gca()
ax.scatter(X[:,0], X[:,1], c= y, edgecolors = 'k', s = 30, cmap= plt.cm.Paired)
plt.show()
# maviler 0 etiketler, kırmızılar 1 etiketleridir. 
# karar çizgisinin dışına mavi nokta çıkarsa 0 -> 1 olur !!! 
# telefonda gerçek tahmin fotoğrafı var..
# 1 soru var tablodan.. 
# %%

from sklearn.metrics import confusion_matrix, classification_report
y_tahmin = svm_linear.predict(X)
print(confusion_matrix(y,y_tahmin))
print(classification_report(y,y_tahmin))

# %%




























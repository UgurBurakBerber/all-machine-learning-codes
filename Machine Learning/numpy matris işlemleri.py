#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:09:51 2025

@author: ugurburak
"""

import numpy as np
iris = np.array([5,7,2,6,3,5,1,0])
print(iris)

from sklearn import datasets
iris2 = datasets.load_iris()

print(np.zeros(3))
print(np.linspace(0,1,num = 5, endpoint = False)) #parçalama start stop

print(np.random.randn(2,2))
print(np.random.randn(10,10))

dizi = np.array([[2,5,7],[3,6,8]])
print(dizi.ndim) # kaç boyutlu dizi olduğumu gösterir.
print(dizi.ravel()) # tek boyutlu hale dönüştürür.
print(dizi.min())
print(dizi.max())
print(dizi.std()) # standart sapması  

#abs = mutlak değerdir.
# %%
import numpy as np

dizi = np. arange(12)
dizi = dizi.reshape(3,4)
print(dizi)
print(dizi[2,3]) # dizi satır ve sütunundaki eleman


print(dizi[2:3])
print(dizi[2,:])
print(dizi[:,3])
# %%

dizi1 = np.full((2,4), 0)
dizi2 = np.full((3,4), 1)
dizi3 = np.full((4,4), 2) # bu full ile 4*4 lük bir matrisi 2 ile doldurdu.
print(dizi3)

dizi4 = np.concatenate((dizi1, dizi2, dizi3),axis = 0)

print(dizi4)



























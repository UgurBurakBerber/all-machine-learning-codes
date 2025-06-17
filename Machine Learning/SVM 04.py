#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:46:34 2025

@author: ugurburak
"""

#SVM Örnek 04

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

veriSeti = pd.read_csv('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/ObesityDataSet_raw_and_data_sinthetic.csv')


# %%

X = veriSeti.iloc[:,:-1].values
y = veriSeti.iloc[:,1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%

from sklearn.preprocessing import LabelEncoder # LabelEncoder sınıfını sklearn'den içeri aktar
le = LabelEncoder() # LabelEncoder sınıfından bir nesne oluştur

# Kodda hangi sütunların etiketlerini (label) dönüştüreceğimiz belirlenmiş
columns_to_encode = [0,4,5,8,9,11,14,15]
for i in columns_to_encode: # Her sütun için etiket dönüşümünü uygula
    X_train[:,i] = le.fit_transform(X_train[:,i]) # X_train'deki belirtilen sütunu dönüştür
    X_test[:,i] = le.fit_transform(X_test[:,i]) # X_test'deki belirtilen sütunu dönüştür
# %%

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%

from sklearn.svm import SVC
model = SVC(kernel='poly', C= 0.01)
model.fit(X_train,y_train)

from sklearn.metrics import classification_report
y_tahmin = model.predict(X_test)
sonuc = classification_report(y_test, y_tahmin)
print(sonuc)
# %%


from sklearn.preprocessing import LabelEncoder # LabelEncoder sınıfını sklearn'den içeri aktar
le = LabelEncoder() # LabelEncoder sınıfından bir nesne oluştur

# Kodda hangi sütunların etiketlerini (label) dönüştüreceğimiz belirlenmiş
columns_to_encode = [0,4,5,8,9,11,14,15]
for i in columns_to_encode: # Her sütun için etiket dönüşümünü uygula
    X[:,i] = le.fit_transform(X[:,i]) # X_train'deki belirtilen sütunu dönüştür

# %%

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
  
    
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=10)
print(scores, scores.mean())









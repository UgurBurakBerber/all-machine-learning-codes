#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 00:13:55 2025

@author: ugurburak
"""

# Pratikleşme Tekrar


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

boy = veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)



#eksik veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
yas = veriler.iloc[:,1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.fit_transform(yas[:,1:4])
print(yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing # veri Dönüştürme 

le = preprocessing.LabelEncoder()

ulke[:,0]= le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)



















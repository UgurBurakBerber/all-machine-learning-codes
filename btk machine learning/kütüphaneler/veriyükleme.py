#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 20:40:51 2025

@author: ugurburak
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#2. veri_ onisleme
#2.1. veri yükleme
veriler = pd.read_csv('/Users/ugurburak/Desktop/python/btk machine learning/eksikveriler.csv')
#print(veriler)

#veri on işleme

boy = veriler[['boy']]
print(boy)

boykilo= veriler[['boy','kilo']]
print(boykilo)

## deneme amaçlı
class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10
    # y = f(x) 
    # f(x) = x + 10
ali = insan()
veli =insan()
print(ali.boy)
print(ali.kosmak(90))
## 

#eksik veriler
import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

yas = veriler.iloc[:, 1:4].values
print(yas)
imputer = imputer.fit(yas[:, 1:4])
yas[:, 1:4] = imputer.transform(yas[:, 1:4])
print(yas)

#Encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() # sayısal olarak her birdeğere 0 dan başlayarak değer verir.

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)


ohe = preprocessing.OneHotEncoder() # colum başlıklarını etiketlere taşımak ve her etiketin altına 1 veya 0 diyerek oraya ai veya değil kanıtlamak
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


#numpy dizileri dataframe donusumu

sonuc =pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)


#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

# verilerin eğitim ve test için bölmesi işlemi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=3)

# verilerin Ölçeklemesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





























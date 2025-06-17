#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 20:23:53 2025

@author: ugurburak
"""
# k- çağraz değişim sınavda çıkar

# KNN - ÖRNEK 2

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

veriSeti = pd.read_csv('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/ObesityDataSet_raw_and_data_sinthetic.csv')
print(veriSeti.head(15))

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
# %%
from sklearn.neighbors import KNeighborsClassifier

siniflandirici = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
siniflandirici.fit(X_train, np.ravel(y_train))
y_tahmin = siniflandirici.predict(X_test)
y_olasilik = siniflandirici.predict_proba(X_test)

# %%

from sklearn.metrics import confusion_matrix, classification_report

hm=confusion_matrix(y_test, y_tahmin)
print(hm)
print(classification_report(y_test, y_tahmin))
# %%

hatalarlistesi = []

for k in range(1,31):
    siniflandirici = KNeighborsClassifier(n_neighbors=k,metric="minkowski",p = 2)
    siniflandirici.fit(X_train, np.ravel(y_train))
    tahmin_k = siniflandirici.predict(X_test)
    hatalarlistesi.append(np.mean(tahmin_k!=y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,31),hatalarlistesi,'b.',linestyle ='--',markersize=6)
plt.title('1.....30 aralığındaki K değerlerine karşılık hata oranları')
plt.xlabel('K değeri', fontsize=15)
plt.ylabel('Hata oranları', fontsize=15)
plt.xticks(fontsize= 15)
plt.yticks(fontsize=15)
plt.grid()
# %%

from sklearn.metrics import roc_curve,auc
ypo, dpo, esikdeger=roc_curve(y_test, y_tahmin,)
AucDegeri = auc(ypo, dpo)

plt.figure()
plt.plot(ypo, dpo,label= 'AUC %0.2f' % AucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('Yanlış Pozitif Oranı (YPO)', fontsize= 15)
plt.ylabel('Doğru Pozitif Oranı (DPO)', fontsize= 15)
plt.title('ROC Eğrisi')
plt.legend(Loc ="best")
plt.grid()
plt.show()
# %%

yeni = np.array([2,26,11,6,1,50,9])
yeni= scaler.transform(yeni.reshape(1, -1))
tahmin = siniflandirici.predict(yeni)
olasilik=siniflandirici.predict_proba(yeni)
print(tahmin,olasilik)



























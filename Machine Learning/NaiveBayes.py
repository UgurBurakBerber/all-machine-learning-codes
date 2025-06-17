#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:32:01 2025

@author: ugurburak
"""
# NaiveBayes Örnek GaussianNB 03

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/Immunotherapy1.xlsx')

X=veriSeti.iloc[:,:-1].values
y=veriSeti.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.25,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB
siniflandirici = GaussianNB()
siniflandirici.fit(X_train,y_train)

from sklearn.metrics import classification_report , confusion_matrix
y_tahmin=siniflandirici.predict(X_test)
y_olasilik = siniflandirici.predict_proba(X_test)
print(classification_report(y_test,y_tahmin),confusion_matrix(y_test, y_tahmin))


import seaborn as sns
hm = confusion_matrix(y_test, y_tahmin)
cm_df= pd.DataFrame(hm,index=['Başarılı','Başarısız'],columns=['Başarılı','Başarısız'])
plt.figure(figsize=(12,8))
sns.heatmap(hm,annot=True,fmt='g',cmap='crest')
plt.title('Hata Matrisi',fontsize=16)
plt.xlabel('Tahmin Edilen Değerler',fontsize=16)
plt.ylabel('Gerçek değerler',fontsize=16)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(siniflandirici, X, y, cv=5)
print(scores,scores.mean())














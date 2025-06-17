#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:36:09 2025

@author: ugurburak
"""
# çoklu doğrusal regresyon..

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#2. veri_ onisleme
#2.1. veri yükleme
veriler = pd.read_csv('/Users/ugurburak/Desktop/python/btk machine learning/veriler.csv')
#print(veriler)

#veri on işleme


#Encoder: Kategorik -> Numeric ülke
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() # sayısal olarak her birdeğere 0 dan başlayarak değer verir.

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder() # colum başlıklarını etiketlere taşımak ve her etiketin altına 1 veya 0 diyerek oraya ai veya değil kanıtlamak
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#Encoder: Kategorik -> Numeric cinsiyet
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() # sayısal olarak her birdeğere 0 dan başlayarak değer verir.

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


ohe = preprocessing.OneHotEncoder() # colum başlıklarını etiketlere taşımak ve her etiketin altına 1 veya 0 diyerek oraya ai veya değil kanıtlamak
c = ohe.fit_transform(c).toarray()
print(c)




#numpy dizileri dataframe donusumu

sonuc =pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data= yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)


#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

# verilerin eğitim ve test için bölmesi işlemi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=3)


# #verilerin Ölçeklemesi
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test = train_test_split(veri, boy ,test_size=0.33,random_state=0)


r2 = LinearRegression()
r2.fit(x_train, y_train)

y_pred = r2.predict(x_test)



import statsmodels.api as sm # modelin başarısını ölçmek R**2 vs.

X = np.append(arr = np.ones((22,1)).astype(int),values=veri,axis=1) # 0. indekse 1 lerden oluşan matris sütunu ekler. 

X_l = veri.iloc[:,[0,1,2,3,4,5]].values #daha sonra eleme yaparken kolonları daha sonra çıkarır
X_l = np.array(X_l, dtype= float)
model = sm.OLS(boy, X_l).fit() #  OLS istatistiksel değerleri çıkartmaya yarıyor..

print(model.summary())


X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())
















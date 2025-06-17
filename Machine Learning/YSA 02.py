#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:14:50 2025

@author: ugurburak

YSA Örnek 2 Çalışmıyor
"""

from tensorflow.keras import layers, models
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)
# %%

model = models.Sequential([layers.Input(shape=(X_train.shape[1],)), # Giriş Katmanı
                           layers.Dense(128,activation ='relu'), # Gizli Katman
                           layers.Dense(3,activation='softmax'),]) #Çıkış Katmanı (3 Sınıf)

model.summary()
# %%


model.compile(optimizer ='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# %%

history = model.fit(X_train,y_train,epochs=50,batch_size= 10,validation_data=(X_test,y_test))

test_loss, test_acc = model.evaluate(X_test,y_test)

print(f"Test Kaybı: {test_loss}")
print(f"Test Doğruluğu: {test_acc}")

# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))

plt.subplot(1, 2,1)
plt.plot(history.history['accuracy'],label ='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'],label ='Test Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label ='Eğitim Kaybı')
plt.plot(history.history['val_loss'],label ='Test Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()


plt.show()





















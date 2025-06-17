#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:30:48 2025

@author: ugurburak

YSA 05 ----- YSA 04 devamı
"""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

class_names = ["tişört","pantolon","kazaka","elbise","ceket","sandalet","gömlek","spor ayakkabı","çanta","bot"]
# %%

plt.imshow(X_train[0], cmap= plt.cm.binary)
plt.title(class_names[y_train[0]])

# %%

X_train = X_train/255.0
X_test = X_test/255.0

# %%

y_train_cat = to_categorical(y_train,10)
y_test_cat = to_categorical(y_test,10)
# %%

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
    ])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# %%


history = model.fit(X_train,y_train_cat, epochs = 10,
                    batch_size = 32,
                    validation_split =0.05,
                    verbose=1)

# %%

test_loss, test_acc = model.evaluate(X_test,y_test_cat)
print(f"Test Doğruluğu: {test_acc:.4f}")


# %%

plt.figure(figsize=(12,5))

plt.subplot(1, 2,1)
plt.plot(history.history['accuracy'],label ='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'],label ='Test Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label ='Eğitim Kaybı')
plt.plot(history.history['val_loss'],label ='Test Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# %%

tahmin = model.predict(X_test)
plt.figure(figsize=(6,3))
plt.imshow(X_test[0],cmap='gray')
plt.title(f"Gerçek : {y_test[100]}- Tahmin: {np.argmax(tahmin[100])}")
plt.axis('off')
plt.show()


























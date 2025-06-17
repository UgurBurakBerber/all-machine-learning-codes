#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:14:12 2025

@author: ugurburak

YAPAY SİNİR AĞLARI (YSA) ÖRNEK 1

"""

import matplotlib.pyplot as plt 

#parametreler
w = 0.3
b = 0
x = 2
y = 1 
eta = 0.1
epochs = 20

weights = []
losses = []

for epoch in range(epochs):
    # ileri yayılım
    z = w*x+b
    y_hat = z
    loss = 0.5*(y-y_hat)**2
    weights.append(w)
    losses.append(loss)
    
    
    # geri yayılım (gradyan)
    grad = (y_hat-y)*x
    
    # ağırlık güncelleme
    w = w-eta*grad
    
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(range(epochs),losses, marker='o',color='red')
plt.title("Epach'a Göre Kayıp")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)


plt.subplot(1,2,2)
plt.plot(range(epochs),weights, marker='s',color='green')
plt.title("Epach'a Göre Ağırlık (w)")
plt.xlabel("Epoch")
plt.ylabel("w")
plt.grid(True)



plt.tight_layout()
plt.show()








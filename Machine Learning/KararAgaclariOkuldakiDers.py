#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:16:42 2025

@author: ugurburak
"""

# Karar ağaçları

import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=42)

siniflandirici = DecisionTreeClassifier(criterion='gini',max_depth=3) #entropy
siniflandirici.fit(X_train, y_train)

plt.figure(figsize=(15,10))
plot_tree(siniflandirici, filled=True,feature_names = iris.feature_names,class_names=iris.target_names)
plt.title("Karar Ağacı Görselleştirme (Entropy ile)")
plt.show()

# Sınavda kesin var yorumlaması
# etiketi?, Frekansı?

# %%

from sklearn.metrics import classification_report, confusion_matrix

y_tahmin=siniflandirici.predict(X_test)
SR = classification_report(y_test, y_tahmin)
print(SR) 

hm = confusion_matrix(y_test, y_tahmin) 
print(hm)





















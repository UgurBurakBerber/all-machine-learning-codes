#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:11:08 2025

@author: ugurburak
"""

# Karar ağaçları 2

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Veri Seti

data = {
        'Hava': ['Güneşli','Güneşli','Bulutlu','Yağmurlu','Yağmurlu','Bulutlu','Güneşli','Yağmurlu'],
        'Nem': ['Yüksek','Yüksek','Yüksek','Normal','Normal','Normal','Normal','Yüksek'],
        'Rüzgar': ['Zayıf','Güçlü','Zayıf','Zayıf','Güçlü','Güçlü','Zayıf','Zayıf'],
        'Piknik': ['Hayır','Hayır','Evet','Evet','Hayır','Evet','Evet','Evet'],
            }
df = pd.DataFrame(data)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df[['Hava','Nem','Rüzgar']]
y = df[['Piknik']]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Düğüm sayısını yazdır
print("Toplam düğüm sayısı:", clf.tree_.node_count)
print("Yaprak sayısı:", clf.get_n_leaves())


plt.figure(figsize=(10,6))
tree.plot_tree(clf, feature_names=['Hava','Nem','Rüzgar'],
               class_names=['Hayır','Evet'],filled=True)
plt.show()


















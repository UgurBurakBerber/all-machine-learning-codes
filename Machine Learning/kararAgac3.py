#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:58:10 2025

@author: ugurburak

Karar Ağaçları 3
"""

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score,f1_score,classification_report

data = pd. read_csv('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/ObesityDataSet_raw_and_data_sinthetic.csv')
#data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print(data)

le = LabelEncoder()

columns_to_encode = ['Gender',
                     'family_history_with_overweight',
                     'FAVC',
                     'CAEC',
                     'SMOKE',
                     'SCC',
                     'CALC',
                     'MTRANS',
                     'NObeyesdad',]

for column in columns_to_encode:
    data[column] = le.fit_transform(data[column])

y = data.NObeyesdad.values
x = data.drop(['NObeyesdad'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.3,random_state=42)

clf = DecisionTreeClassifier(criterion='entropy',random_state=1)

clf.fit(X_train, y_train)

y_tahmin = clf.predict(X_test)
y_olasilik = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_tahmin),
      precision_score(y_test, y_tahmin,average='weighted'),
      f1_score(y_test, y_tahmin, average='weighted'),
      classification_report(y_test, y_tahmin))

import numpy as np
plt.figure(figsize=(200,5))
plot_tree(clf,max_depth=2,fontsize=10,filled=True,
          feature_names=list(X_train.columns),class_names=list(map(str,np.unique(y_train))))




















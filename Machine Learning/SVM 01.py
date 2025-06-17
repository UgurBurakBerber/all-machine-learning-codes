#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:40:30 2025

@author: ugurburak
"""

#SVM -> Örnek-1

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# y_olasilik = model.predict_proba(X_test)  # kullanmak istersen

print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(y_test, y_pred)
print(hm)
print(classification_report(y_test, y_pred))
# %%

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring='recall_macro')
print(scores)
print(scores.mean())
# %%

from sklearn.metrics import get_scorer_names
print(get_scorer_names())











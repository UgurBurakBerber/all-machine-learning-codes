#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:00:09 2025

@author: ugurburak

"""
# feature_selection 

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
model = LogisticRegression(max_iter= 1000)
rfe = RFE(model,n_features_to_select=1)
rfe = rfe.fit(iris.data,iris.target)

print(rfe.support_)
print(rfe.ranking_)


# %%

from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
X,y = load_iris(return_X_Y = True)

chi2_stats ,p_values = chi2(X,y)
print(chi2_stats)
print(p_values)

# %%

from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
X,y = load_iris(return_X_Y = True)

f_statistics,p_values = f_classif(X, y)
print(f_statistics)
print(p_values)
# %%

from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
X,y = make_classification(n_samples=100, n_features=10,n_informative=2,n_clusters_per_class=1,shuffle=False,random_state=42)

f_statistics,p_values = f_classif(X, y)
print(f_statistics)
print(p_values)

# %%

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

Immunotherapy = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/Immunotherapy.xlsx')

model = LogisticRegression()
rfe = RFE(model,n_features_to_select=3)
rfe = rfe.fit(Immunotherapy.iloc[:,0:6],Immunotherapy.iloc[:,7])

print(rfe.support_)
print(rfe.ranking_)

# %%

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

Immunotherapy = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/Immunotherapy.xlsx')

model = SVR(kernel='Linear')
rfe = RFE(model,n_features_to_select=4)
rfe = rfe.fit(Immunotherapy.iloc[:,0:6],Immunotherapy.iloc[:,7])

print(rfe.support_)
print(rfe.ranking_)

# %%

import numpy as np 
import sklearn.datasets

iris = datasets.load_iris()
canakyaprakUzunluk=iris.data[:,0]
ortalama = np.mean(canakyaprakUzunluk)
ss=np.std(canakyaprakUzunluk)

X = np.arange(150, dtype = float)

for i in range (150):
    X[i] = (canakyaprakUzunluk[i]-ortalama)/ss

print(X)    


# %%

import numpy as np 
import sklearn.datasets

iris = datasets.load_iris()
canakyaprakUzunluk=iris.data[:,0]
ortalama = np.mean(canakyaprakUzunluk)

mn = np.min(canakyaprakUzunluk)
mx = np.max(canakyaprakUzunluk)

X = np.arange(150, dtype = float)

for i in range (150):
    X[i] = (canakyaprakUzunluk[i]-mn)/(mx-mn)

print(X)

# %%
#non değerleri bulma işlemi

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/VeriOnIsleme.xlsx')
X = veriSeti.iloc[:,:-1].values
y = veriSeti.iloc[:,-1].values

yaklasikDeger = SimpleImputer(missing_values=np.nan,strategy="most_frequent") # {'most_frequent', 'median', 'constant', 'mean'} 
X[:,1:5] = yaklasikDeger.fit_transform(X[:,1:5])

print(yaklasikDeger.statistics_)

# %%

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/VeriOnIsleme.xlsx')
X = veriSeti.iloc[:,:-1].values
y = veriSeti.iloc[:,-1].values

yaklasikDeger = SimpleImputer(missing_values=np.nan,strategy="most_frequent") # {'most_frequent', 'median', 'constant', 'mean'} 
X[:,0:5] = yaklasikDeger.fit_transform(X[:,0:5])

print(yaklasikDeger.statistics_)

# %% Min max dönüşümü

import numpy as np
X = np.array([22,87,20,91,48,61,76,51,29,18])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y=scaler.fit_transform(X.reshape(-1,1))
# %% 

import numpy as np 
import sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,0]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y=scaler.fit_transform(X.reshape(-1,1))
# %%

import numpy as np 
import sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y=scaler.fit_transform(X.reshape(-1,1))































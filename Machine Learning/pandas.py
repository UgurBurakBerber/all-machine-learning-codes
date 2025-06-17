#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:40:20 2025

@author: ugurburak
"""

import pandas as pd

seri1 = pd.Series([2025,'Makine Öğrenmesi', -5.23])
print(seri1)

# %%

import pandas as pd

seri1 = pd.Series([2025,'Makine Öğrenmesi', -5.23],index = ['A','B','C'])
print(seri1)
print(seri1['B'])

# %%

notlar = {"Matematik": 70,'Bilgisayar': 90,"Yapay Zeka": 100,"İstatistik":85}

seri2 = pd.Series(notlar)
print(seri2)
print(seri2*1.05)
# %%

seri3 = pd.Series([10,15,20,25,25,20,10,15,45,50,20])

print(seri3.unique())
print(seri3.value_counts())
print(seri3[seri3>20])



































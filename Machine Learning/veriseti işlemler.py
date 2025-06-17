#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:59:03 2025

@author: ugurburak
"""

import pandas as pd
#veriSeti = pd.read_excel('EnergyStarScore.xlsx')
veriSeti = pd.read_excel('/Users/ugurburak/Desktop/Machine Learning/veri setleri/EnergyStarScore.xlsx')

veriSeti.info() # boşlukları vs gösteriyor

veriSeti.sample(3) 
veriSeti.head(3) # ilk 3 tanesi
veriSeti.tail(3) # son 3 tanesi
veriSeti[['Bina Adı','ESS']].head(7)

veriSeti[veriSeti.ESS>=95].head(3)

veriSeti[(veriSeti.ESS>=60) & veriSeti.ESS<=70].head(3)



veriSeti2 = veriSeti[["BA","BEKY","SGE","SGY","BTE","ESS"]]
veriSeti2.corr()

# %%


import seaborn as sb

sb.heatmap(veriSeti2.corr(),annot = True, fmt = ".2f" ,cmap = "Greens", linewidth = 0.01)




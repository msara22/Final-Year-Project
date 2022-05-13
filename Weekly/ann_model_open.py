# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:15:47 2022

@author: sarvinoz.toshpulotov
"""
import pickle
with open ('ann_model','rb') as f:
      mod= pickle.load(f)
      
y_pred= mod.predict([[5,	3,	0,	0,	6,2,	1,	2016,	76,	71.2,	66,	74,	70.3,	66,	100,	96.8,	89,	18,	7.8,	0,	29.8,	29.8,	29.7,	4.33
]])

print(y_pred)
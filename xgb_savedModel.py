# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:33:35 2022

@author: sarvinoz.toshpulotov
"""

import pickle
import pandas as pd
import numpy as np

# df = pd.read_csv("Dataset2.csv")
# day = df.iloc[:,4].values
# month = df.loc[:,'month'].values
# df1 = df.loc[(day == 6)].copy()
# df2 = df1[(df1.loc[:,'month'].values ==6)]





def xgb(hour_from, day_from, month_from, year_from):  
    df = pd.read_csv("C://Users//sarvinoz.toshpulotov//Desktop//App_flask//Dataset2.csv")
    loaded_model = pickle.load(open("random_model.pkl", 'rb'))
    month = df.loc[:,'month'].values
    day = df.loc[:,'day'].values

    year = df.loc[:,'year'].values
    choice= {1:"daily", 2:"weekly", 3:'monthly'}
    # print(day[0])
    # print(type(day[0]))
    # print(type(day_from))
    
    df1 = df.loc[(day == day_from)].copy()
    print(df1)
    if(df1.shape[0]!=0):
        print("hello")
        df2 = df1[(df1.loc[:,'month'].values ==month_from)]
    else: 
        df2 = df[(df.loc[:,'month'].values ==month_from)]
    print(df2)
    df3 = df2[(df2.loc[:,'year'].values == year_from)]
    final = df3[(df3.loc[:,'hour'].values == hour_from)]
    x = final.iloc[:,:-1].values
    print(final.values.tolist())
    result = loaded_model.predict(x)
    print(result)
        
    
    # if(df3.empty== True):
    #     print(df2)
    # else:
    #     print(df3)
 
         
        
        
    
        

    
    

xgb(15, 10, 11, 2016)
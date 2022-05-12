# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:31:38 2022

@author: sarvinoz.toshpulotov
"""

import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Dataset2.csv")
X, y =df.iloc[:,0:-1].values, df.iloc[:,-1].values
print(df)

# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0,1))
# x_scaled = sc.fit_transform(X)
# y_scaled = sc.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 20)


X_train =  df.iloc[:10945, 0:-1].values
y_train =  df.iloc[:10945, -1].values
X_test  = df.iloc[10946:, 0:-1].values
y_test =  df.iloc[10946:, -1].values

xgb_r = xg.XGBRegressor(objective="reg:squarederror", n_estimators = 23, seed = 15)
xgb_r.fit(X_train, y_train)




y_pred = xgb_r.predict(X_test)

y_pred_train = xgb_r.predict(X_train)

import pickle
pickle.dump(xgb_r, open("xbg_model.pkl", 'wb'))

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
mae= mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(round(mae,2))
print(round(mape,2))
print(round(mse,2))
print(round(rmse,2))

print("//////")

mae1= mean_absolute_error(y_train, y_pred_train)
mse1 = mean_squared_error(y_train, y_pred_train)
rmse1 = np.sqrt(mse1)
mape1 = mean_absolute_percentage_error(y_train, y_pred_train)
print(round(mae1,2))
print(round(mape1,2))
print(round(mse1,2))
print(round(rmse1,2))
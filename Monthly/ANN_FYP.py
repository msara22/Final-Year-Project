# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:18:56 2022

@author: sarvinoz.toshpulotov
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

data = pd.read_csv("C://Users//sarvinoz.toshpulotov//Desktop//FYP//Codes//att2//Dataset2.csv")


# X = df.iloc[:,8:-1].values
# y = df.iloc[:,-1].values


#Spliting the data into train and test set
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

X_train =  data.iloc[:10250, 0:-1].values
y_train =  data.iloc[:10250, -1].values
X_test  = data.iloc[10250:, 0:-1].values
y_test =  data.iloc[10250:, -1].values


#Initializing ANN(Creating sceleton of ANN)
ann = tf.keras.models.Sequential()
#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 17,input_dim = 24, activation="relu"))
#
#Adding the  second hidden layer
ann.add(tf.keras.layers.Dense(units = 17, activation="sigmoid"))
#Adding output layer
ann.add(tf.keras.layers.Dense(units=1))
#Compilinf the ANN(Reducing the loss)
ann.compile(optimizer = "adam",loss = "mean_absolute_error" )
#Training the ANN model on the Training Set
ann.fit(X_train, y_train, epochs = 20, verbose=1,  batch_size =32)
y_pred = ann.predict(X_test)
y_pred_train = ann.predict(X_train)

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
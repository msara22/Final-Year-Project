
# import numpy package for arrays and stuff
import numpy as np 
from sklearn import tree
  
# import matplotlib.pyplot for plotting our result
import matplotlib.pyplot as plt
  
# import pandas for importing csv files 
import pandas as pd 
import random
import pickle

df = pd.read_csv("Dataset2.csv")
X, y =df.iloc[:,0:-1].values, df.iloc[:,-1].values
print(df)

print(df.isnull().sum())
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values


df.columns
  
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.05, random_state = 0)

X_train =  df.iloc[:10944, 0:-1].values
y_train =  df.iloc[:10944, -1].values
X_test  = df.iloc[10944:, 0:-1].values
y_test =  df.iloc[10944:, -1].values



# import the regressor
from sklearn.ensemble import RandomForestRegressor
  
  # create regressor object
regressor = RandomForestRegressor(n_estimators = 56, random_state = 15)
 
# fit the regressor with x and y data
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train)

pickle.dump(regressor, open('random_model.pkl','wb'))

pickle_model = pickle.load( open( "random_model.pkl", 'rb'))
pickle_model.predict(X_test)
                                 

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


# import numpy package for arrays and stuff
import numpy as np 
  
# import matplotlib.pyplot for plotting our result
import matplotlib.pyplot as plt
  
# import pandas for importing csv files 
import pandas as pd 

data = pd.read_csv('Dataset2.csv')
CorData = data.corr(method='kendall')
with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    df = data.drop(columns = ["Temp_max", "Temp_avg", "Temp_min", "Dew_min", "Dew_avg", "Dew_max", "Press_min", "Hum_max", "day_of_week", "Wind_min"])

X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values


# import the regressor
from sklearn.neighbors import KNeighborsRegressor



from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

X_train =  data.iloc[:10802, 0:-1].values
y_train =  data.iloc[:10802, -1].values
X_test  = data.iloc[10802:, 0:-1].values
y_test =  data.iloc[10802:, -1].values

neigh = KNeighborsRegressor(n_neighbors=2)  

neigh.fit(X_train, y_train)

y_pred=neigh.predict(X_test)
y_pred_train = neigh.predict(X_train)

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
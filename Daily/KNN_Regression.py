
# import numpy package for arrays and stuff
import numpy as np 
  
# import matplotlib.pyplot for plotting our result
import matplotlib.pyplot as plt
  
# import pandas for importing csv files 
import pandas as pd 

data = pd.read_csv('C://Users//sarvinoz.toshpulotov//Desktop//FYP//Codes//att2//Dataset2.csv')
CorData = data.corr(method='kendall')
with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)
df = data.drop(columns = ["minute","Temp_max", "Temp_avg", "Temp_min", "Dew_min", "Dew_avg", "Dew_max", "Press_min", "Hum_max", "day_of_week", "Wind_min"])

print(df.isnull().sum())
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x_scaled = sc.fit_transform(X)
y_scaled = sc.fit_transform(y.reshape(-1,1))

# import the regressor
from sklearn.neighbors import KNeighborsRegressor



from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

X_train =  x_scaled[:10944, 0:-1]
y_train =  y_scaled[:10944, -1]
X_test  = x_scaled[10944:, 0:-1]
y_test =  y_scaled[10944:, -1]

neigh = KNeighborsRegressor(n_neighbors=2)  

neigh.fit(X_train, y_train)

y_pred=neigh.predict(X_test)
y_pred_train = neigh.predict(X_train)

y_test = sc.inverse_transform(y_test.reshape(-1,1))
y_pred = sc.inverse_transform(y_pred.reshape(-1,1))

y_train = sc.inverse_transform(y_train.reshape(-1,1))
y_pred_train = sc.inverse_transform(y_pred_train.reshape(-1,1))

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
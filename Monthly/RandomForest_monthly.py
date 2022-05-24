# import numpy package for arrays and stuff
import numpy as np 
  
# import matplotlib.pyplot for plotting our result
import matplotlib.pyplot as plt
  
# import pandas for importing csv files 
import pandas as pd 

# reading csv file
data = pd.read_csv('Dataset2.csv')

# finding the corellation of feature to each other
CorData = data.corr(method='kendall')
with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)
    
# removing unnecessary features
df = data.drop(columns = ["minute","Temp_max", "Temp_avg", "Temp_min", "Dew_min", "Dew_avg", "Dew_max", "Press_min", "Hum_max", "day_of_week", "Wind_min"])

# dividing the dataset into X and Y
print(df.isnull().sum())
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Preprocessing the data using MinMax Scaler, which is used for normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x_scaled = sc.fit_transform(X)
y_scaled = sc.fit_transform(y.reshape(-1,1))


# Dividing the dataset into training and testing data. As it is monthly, so the size of testing data will be 720
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

X_train =  x_scaled[:10247, 0:-1]
y_train =  y_scaled[:10247, -1]
X_test  = x_scaled[10247:, 0:-1]
y_test =  y_scaled[10247:, -1]


# import the regressor
from sklearn.ensemble import RandomForestRegressor
  
  # create regressor object
regressor = RandomForestRegressor(n_estimators = 56, random_state = 15) 

regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

# reversing the normalized dataset
y_test = sc.inverse_transform(y_test.reshape(-1,1))
y_pred = sc.inverse_transform(y_pred.reshape(-1,1))

y_train = sc.inverse_transform(y_train.reshape(-1,1))
y_pred_train = sc.inverse_transform(y_pred_train.reshape(-1,1))

#saving the model
import pickle
pickle.dump(regressor, open("rf_monthly.pkl", 'wb'))

#performance evaluation metrics to check the accuraty of the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
mae= mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Performance evaluation for testing data")
print(round(mae,2))
print(round(mape,2))
print(round(mse,2))
print(round(rmse,2))

print("Performance evaluation for training data")

mae1= mean_absolute_error(y_train, y_pred_train)
mse1 = mean_squared_error(y_train, y_pred_train)
rmse1 = np.sqrt(mse1)
mape1 = mean_absolute_percentage_error(y_train, y_pred_train)
print(round(mae1,2))
print(round(mape1,2))
print(round(mse1,2))
print(round(rmse1,2))

import matplotlib.pyplot as plt
plt.title("Actual vs Predicted values for testing data")
plt.plot(y_pred[:200], label = "Predicted")
plt.plot(y_test[:200], label = "Actual")
legend = plt.legend()
plt.xlabel("Time(Hours)")
plt.ylabel("Energy Consumed(kW)")
plt.show()

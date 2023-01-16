### Weather forecasting with Recurrent Neural Networks in Python ----
### (available online: https://medium.com/analytics-vidhya/weather-forecasting-with-recurrent-neural-networks-1eaa057d70c3)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('inst/extdata/training.csv', sep = '\t')
dataset = dataset.dropna(subset=["Temperature"])
dataset=dataset.reset_index(drop=True)
training_set = dataset['Temperature'].values.reshape(-1, 1)

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(training_set_scaled)-n_past-n_future+1):
    x_train.append(training_set_scaled[i : i + n_past , 0])     
    y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )

# build model
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout

regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future,activation='linear'))
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
regressor.fit(x_train, y_train, epochs=25,batch_size=32 )

# test data
testdataset = pd.read_csv('inst/extdata/testing.csv')
testdataset = testdataset.iloc[:30,1].values.reshape(-1, 1)
real_temperature = pd.read_csv('inst/extdata/testing.csv')
real_temperature = real_temperature.iloc[30:,1].values.reshape(-1, 1)

testing = sc.transform(testdataset)
testing = np.array(testing)
testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))

# predict
predicted_temperature = regressor.predict(testing)
predicted_temperature = sc.inverse_transform(predicted_temperature)
predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))

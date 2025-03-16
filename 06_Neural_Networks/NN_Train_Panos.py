#Necesasary Packages; numpy, pandas, sklearn, tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#Data handaling
path = r'C:\Users\infra\PycharmProjects\ME44312---Machine-Learning\green_taxi_data.parquet'   # Load the dataset from the parquet file
data = pd.read_parquet(path)                                                                                                                    # Read the loaded file

data = data[['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']]                                                                  # Select the relevant columns
data.head()

X = data[['trip_distance', 'fare_amount', 'passenger_count']].values #Inputs
Y = data[['tip_amount']].values #Output


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 42)
#X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 1)


#Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

NN = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=1000, activation='relu')

NN.fit(X_train,y_train)
NN_predict = NN.predict(X_test)

#Compute errors
print('MSE', mean_squared_error(y_test,NN_predict))
print('MAE', mean_absolute_error(y_test,NN_predict))

#Keras neural network

model = Sequential()
#Want to use an expotential linear unit instead of the usual relu
model.add( Dense(100, activation='relu', input_shape=(100) ) )
model.add( Dense(100), activation='relu')
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

history = model.fit(X_train,y_train,epochs = 300)
model.evaluate(X_test,y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


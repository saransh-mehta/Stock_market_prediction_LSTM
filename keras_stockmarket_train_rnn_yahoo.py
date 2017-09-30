import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# Importing the training set
training_set = pd.read_csv('Yahoo_stock.csv')

#first we will just take open prices
training_set = training_set.iloc[:,2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
x_train = training_set[0:1760]
y_train = training_set[1:1761]

# Reshaping
x_train = np.reshape(x_train, (176, 10, 1))
y_train = np.reshape(y_train, (176, 10, 1))

y_train.shape

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

model = Sequential()

model.add(LSTM(units = 10, return_sequences = True, input_shape = x_train.shape[1:], init='glorot_normal', inner_init='glorot_normal'))
model.add(Activation('relu'))

model.add(LSTM(units = 10, return_sequences = True, input_shape = x_train.shape[1:], init='glorot_normal', inner_init='glorot_normal'))
model.add(Activation('relu'))

model.add(Dense(units = 128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrics = ['accuracy'])

epochs = 500

for i in range(10):
	
    model.fit(x_train, y_train, epochs = epochs, validation_split = 0.1, verbose = 1)
    model.save('LSTM2_stockmarket_YAHOO_' + str((epochs * i + 500)) + '.h5') 
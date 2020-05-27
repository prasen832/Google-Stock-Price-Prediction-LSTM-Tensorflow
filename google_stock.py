import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, LeakyReLU, Input, Dropout, Dense, Add, Dropout, LSTM
from tensorflow.keras import Model, datasets, models
from tensorflow.keras.optimizers import Adam

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('google.csv', date_parser=True)

data.tail()

train_data = data[data['Date']<'2019-01-01'].copy()

test_data = data[data['Date']>='2019-01-01'].copy()

x_data = train_data.drop(['Date', 'Adj Close'], axis=1)

scaler = MinMaxScaler()

x_data = scaler.fit_transform(x_data)


x_train = []
y_train = []

# we will feed 60 samples and 61st is predicted. 
# In this manner we will learn LSTM network to predict based on last 60 samples

for i in range(60, train_data.shape[0]):
    x_train.append(x_data[i-60:i])
    y_train.append(x_data[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

def stock():
    
    I = Input(shape=[x_train.shape[1],5])
    
    L1 = LSTM(units=50, activation='relu', return_sequences=True)(I)
    D1 = Dropout(0.2)(L1)
    
    L2 = LSTM(units=60, activation='relu', return_sequences=True)(D1)
    D2 = Dropout(0.2)(L2)
    
    L3 = LSTM(units=70, activation='relu', return_sequences=True)(D2)
    D3 = Dropout(0.2)(L3)
    
    L4 = LSTM(units=100, activation='relu')(D3)
    D4 = Dropout(0.2)(L4)
    
    out = Dense(1)(D4)
    
    model = Model(inputs=I, outputs=out)
    
    return model

model = stock()
model.summary()

model.compile(optimizer='adam', loss='MSE')

model.fit(x_train, y_train, batch_size=10, epochs=1)

last_60 = train_data.tail(60)

xtest = xtest.drop(['Date', 'Adj Close'], axis=1)

xt = scaler.transform(xtest)

x_test = []
y_test = []

for i in range(60,xt.shape[0]):
    x_test.append(xt[i-60:i])
    y_test.append(xt[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

pred = model.predict(x_test)

# Bringing scale value to normal

# p = scaler.inverse_transform(pred)
# y = scaler.inverse_transform(y_test)

scaler.scale_

scale = 1/8.18605127e-04

p = pred*scale
y = y_test*scale

plt.figure()
plt.plot(p, color='green', label='Predicted')
plt.plot(y, color='red', label='Actual')
plt.title('Google stock prices')
plt.xlabel('Date Progression')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
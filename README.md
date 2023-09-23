# Google Stock Prediction Using LSTM Tensorflow

![stock](Stock.jpg)

## Overview
1. RNN and LSTM are sequential learning models where LSTM is an upgraded model of RNN.
2. The target of this project is to predict the next day's Google stock  open price using LSTM model.
3. An LSTM-based neural network is designed with Keras in Tensorflow 2 to predict the next day's stock open price.
4. The prediction stock price graph is exactly following the actual price graph.

## Dataset
The data file (CSV) is available above.

## EDA

1. Load the dataset using Pandas.
2. Observe a few rows of the dataset carefully.
3. Check for a number of feature columns present in it.
4. Use info() to check datatypes of all feature columns it also shows various info about the dataset.
5. Check for, if null values present in the dataset. 
6. Use describe() to check various statistical information about data.

## Feature Engineering

1. Make Date column as pandas DATETIME datatype.
2. Create day, month and year as 3 new features form Date column.
3. For training data we will take data before 2019-01-01.
4. For testing data we will take data on and after 2019-01-01.
5. Delete Date column as we no longer needed it.
6. Do Feature normalization using MinMaxScaler()
7. x_train should contain previous 60 samples with all features.  
8. y_train should contain 61th sample of 'Open' value as we are predicting stock open price for next day.

## LSTM Model

1. Create an LSTM neural network model.
2. Compile the model with optimizer='adam'and loss='MSE'
3. Fit the model with x_train, y_train, batch_size=100, epochs=30
4. Observe whether the loss is reducing with each epoch of training or not.
5. If the loss is not reducing then the model needs to be modified with new Hyperparameters.
6. If the loss is reducing then the model is working perfectly.
7. At the end of training the loss should be as minimal as possible.

## LSTM Model Performance

The **Regression Model** performance should be **evaluated** on **Mean Absolute Percentage Error.**

#### The Percentage Accuracy for Test data is  %

#### The Mean Absolute Percentage Error for Test data  %

## Prediction Plot

## Conclusion 

#### The prediction graph is exactly following the actual graph bus still accuracy improvement is needed.

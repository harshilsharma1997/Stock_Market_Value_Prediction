#Importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#Importing dataset
df = pd.read_csv('google.csv') 

#setting date as index for x-axis
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#printing first 25 values of dataset
print(df.head(25))

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'],linewidth=2)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

#creating test and train data
train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
plt.figure(figsize=(16,8))

#displaying test and train data
plt.title('Google Prices')
plt.xlabel('Dates',fontsize=18)
plt.ylabel('Prices',fontsize=18)
plt.plot(df['Close'], 'blue', label='Training Data')
plt.plot(test_data['Close'], 'green', label='Testing Data')
plt.legend()

#function for moving average
def func(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))
train_ar = train_data['Close'].values
test_ar = test_data['Close'].values

#making predictions based on pevious data
history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    
#comparizon between actual and predicted values
result = pd.DataFrame({'Actual':test_ar,'Predicted':predictions})
print(result)

#diplaying analytics    
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = func(test_ar, predictions)
print('Mean absolute percentage error: %.3f' % error2)
print('Accuracy obtained: ',  100.000 - error2)


#Comparing prices and displaying
plt.figure(figsize=(16,8))
plt.plot(test_data.index, predictions, color='blue',marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
plt.title('Google Prices Prediction')
plt.xlabel('Dates',fontsize=18)
plt.ylabel('Prices',fontsize=18)
plt.legend()

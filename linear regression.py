import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style

# style.use('ggplot')
plt.style.use('fivethirtyeight')

#Importing the dataset
df = pd.read_csv('google.csv') 

#Setting index as date and date format
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

print(df.head(25))

df.describe()

x = df[['Open','High','Low','Volume']].values
y = df['Close'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'],linewidth=2)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

regressor = LinearRegression()

regressor.fit(x_train,y_train)

print(regressor.coef_)

y_pred = regressor.predict(x_test)

result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})

result.head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#graph=df.head(20)
#graph.plot(kind='bar')

plt.figure(figsize=(16,8))
plt.title('Actual Price')
plt.plot(result['Actual'])
# plt.plot(result['Predicted'])
# plt.plot(y_test)
# plt.plot(y_pred)

# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Close',fontsize=18)

plt.figure(figsize=(16,8))
plt.title('Predicted Price')
plt.plot(result['Actual'])

plt.figure(figsize=(16,16))
plt.title('Actual and predicted Price')
plt.plot(result['Actual'])
plt.plot(result['Predicted'])



#Import the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(style="ticks", color_codes=True)
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor

# style.use('ggplot')
plt.style.use('fivethirtyeight')

#Importing the dataset
df = pd.read_csv('google.csv') 

#Setting index as date and date format
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

print(df.head(25))

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'],linewidth=2)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

x = df[['Open','High','Low','Volume']].values
y = df['Close'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.75,random_state=0)

rmse_val = [] #to store rmse values for different k
for K in range(40):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k =' , K , 'is:', error)

curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

KNN=KNeighborsRegressor(n_neighbors=120)
KNN.fit(x_train,y_train)
y_pred=model.predict(x_test)

result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
print(result)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('KNN score :', KNN.score(x_test, y_test))
print('Model Accuracy in %age :', 100*KNN.score(x_test, y_test))

sns.pairplot(result,x_vars=['Actual'],y_vars='Predicted',height=7,aspect=1.5,kind='reg')







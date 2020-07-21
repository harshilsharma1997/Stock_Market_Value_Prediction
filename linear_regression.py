#Import the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns;

# style.use('ggplot')
plt.style.use('fivethirtyeight')

#Importing the dataset
df = pd.read_csv('google.csv') 

#setting date as index for x-axis
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#displaying first 25 values in dataframe df
print(df.head(25))


sns.set(style="ticks", color_codes=True)
sns.pairplot(df,x_vars=['Open'],y_vars='Close',height=8,aspect=1.5,kind='reg',plot_kws={'line_kws':{'color':'red' ,'linewidth':'2'}})
sns.pairplot(df,x_vars=['Low'],y_vars='High',height=8,aspect=1.5,kind='reg',plot_kws={'line_kws':{'color':'red' ,'linewidth':'2'}})
sns.pairplot(df, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))

print(df.describe())

x = df[['Open','High','Low']].values
y = df['Close'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)

plt.figure(figsize=(12,6))
plt.title('Close Price History',fontsize=18)
plt.plot(df['Close'],linewidth=2)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

regressor = LinearRegression()

z=regressor.fit(x_train,y_train)

#displaying regressor coefficients
print(regressor.coef_)

y_pred = regressor.predict(x_test)


#displaying result for predicted values
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
print(result.head(25))

z.score(x_test, y_test)

#displaying analytics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Z score :', z.score(x_test, y_test))
print('Model Accuracy in %age :', 100*z.score(x_test, y_test))

#displaying various plots 
plt.figure(figsize=(12,6))
plt.title('Open Price USD ($)',fontsize=18)
plt.plot(result['Actual'])
plt.plot(result['Predicted'])

plt.figure(figsize=(12,6))
plt.title('Predicted Price USD ($)',fontsize=18)
plt.plot(result['Actual'])
plt.show()

plt.figure(figsize=(12,6))
plt.title('Actual and predicted Price USD ($)',fontsize=18)
plt.plot(result['Actual'])
plt.plot(result['Predicted'])
plt.show()

plt.figure(figsize=(12,6))
plt.title('Actual and predicted Price USD ($)',fontsize=18)
plt.scatter(result.Actual,result.Predicted)
plt.xlabel('Actual Price USD ($)',fontsize=18)
plt.ylabel('Close  Price USD ($)',fontsize=18)
plt.show()

sns.pairplot(result,x_vars=['Actual'],y_vars='Predicted',height=8,aspect=1.5,kind='reg',plot_kws={'line_kws':{'color':'red' ,'linewidth':'2'}})

m= df[['Open','High','Low']].values
y_pred2 = regressor.predict(m)
print('Prediction: '),
y_pred2

graph=result.head(20)
graph.plot(kind='bar',figsize=(16,8))
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dfr = pd.read_csv('C:/Users/DELL/Desktop/20242/Hoc may/Baitaplon/output.csv', low_memory=False)

# Create the data of independent variables
dependent_variable = 'Sales'
# Create a list of independent variables

X = dfr.drop('Sales',axis=1,inplace=False)

# Create the dependent variable data
y = dfr[dependent_variable].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#Checking the score on train set.
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

r2_train= r2_score(y_train, y_pred_train)
print(r2_train)

r2_test= r2_score(y_test, y_pred_test)
print(r2_test)


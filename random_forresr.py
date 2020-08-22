# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:08:09 2020

@author: Gauri
"""
#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#input
df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:-1].values
Y = df.iloc[:, -1].values

Y = Y.reshape(len(Y), 1)
#no feature scaling needed for decision trees YAYAYAY

#training

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

#predict
regressor.predict([[6.5]])

#visualise

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff ( Random Forest)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
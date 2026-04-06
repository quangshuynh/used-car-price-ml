"""
train.py

Training severl models (Linear Regression, KNN Regressor,
Decision Tree Regressor, Random Forest Regressor, Neural Network (MLPRegressor), Support Vector Regressor) 
on the used car dataset

Author: Kai Fan kf5601
"""
# todo:
# - Linear Regression
# - KNN Regressor
# - Decision Tree Regressor
# - Random Forest Regressor
# - Neural Network (MLPRegressor) with hidden layers (32, 16)
# - Support Vector Regressor (SVR)
#
# IMPORTANT: Use this from Quang's code
#     preprocessor.fit_transform(X_train)
#     preprocessor.transform(X_val)

# Import necessary libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
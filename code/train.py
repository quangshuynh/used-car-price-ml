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
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from preprocessing import clean_and_prepare_data

# test file access
# Get the directory of the current script, then go up one level
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir + "\\data\\raw", "Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv")
try:
    with open(file_path, 'r') as file:
        print("File access successful!")
except FileNotFoundError:
    print("File not found. Please check the path:", file_path)
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


def linear_Regression(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def knn_Regressor(X_train, y_train, X_val, y_val):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def decision_Tree_Regressor(X_train, y_train, X_val, y_val):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def random_Forest_Regressor(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def svr(X_train, y_train, X_val, y_val):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def mlp_Regressor(X_train, y_train, X_val, y_val):
    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return evaluate_model(y_val, y_pred)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def main():
    # Load and preprocess the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir + "\\data\\raw", "Used Car Price Prediction Dataset export 2026-03-20 19-46-48.csv")
    
    df = pd.read_csv(file_path)
    X, y, preprocessor = clean_and_prepare_data(df)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate each model
    models = {
        "Linear Regression": linear_Regression,
        "KNN Regressor": knn_Regressor,
        "Decision Tree Regressor": decision_Tree_Regressor,
        "Random Forest Regressor": random_Forest_Regressor,
        "Support Vector Regressor": svr,
        "MLP Regressor": mlp_Regressor
    }
    

    for name, func in models.items():
        mse, mae, r2 = func(X_train, y_train, X_val, y_val)
        print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
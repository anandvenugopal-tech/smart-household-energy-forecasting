import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib

def train_model():
    # load data
    df = pd.read_csv('C:\Machine_Learning_Projects\energy-load-forecasting\smart-household-energy-forecasting\data\processed\household_features.csv')
    
    # Define features and target
    X = df.drop(['Datetime','Global_active_power'], axis = 1)
    y = df['Global_active_power']

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, shuffle = False)

    print('Training model...')

    model = RandomForestRegressor(
        n_estimators = 100,
        max_depth = 10,
        min_samples_split = 5,
        min_samples_leaf = 2, 
        n_jobs = -1,
        random_state = 0
    )

    # train model
    model.fit(X_train, y_train)

    # save the model
    os.makedirs('models', exist_ok = True)
    joblib.dump(model, "models/randomforest_model.pkl")
    
    print('Model saved at models/random_forest_model.pkl')

if __name__ == "__main__":
    train_model()
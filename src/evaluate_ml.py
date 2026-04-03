import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate():
    # load data
    df = pd.read_csv('C:\Machine_Learning_Projects\energy-load-forecasting\smart-household-energy-forecasting\data\processed\household_features.csv')
    
    # Define features and target
    X = df.drop(['Datetime','Global_active_power'], axis = 1)
    y = df['Global_active_power']

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, shuffle = False)

    model = joblib.load('models/randomforest_model.pkl')

    y_pred = model.predict(X_test)

    # evaluation
    print(f"MAE: {mean_absolute_error(y_pred, y_test)}")
    print(f"MSE: {mean_squared_error(y_pred, y_test)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_pred, y_test))}")


if __name__ == "__main__":
    evaluate()



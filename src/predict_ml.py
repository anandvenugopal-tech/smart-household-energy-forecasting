import pandas as pd
import joblib

def predict():
    model = joblib.load('models/randomforest_model.pkl')
    df = pd.read_csv('C:\Machine_Learning_Projects\energy-load-forecasting\smart-household-energy-forecasting\data\processed\household_features.csv')
    test_data = df.iloc[-1:]
    X = test_data.drop(['Global_active_power', 'Datetime'], axis = 1)
    prediction = model.predict(X)
    print('Predicted power: ', prediction[0])

if __name__ == "__main__":
    predict()

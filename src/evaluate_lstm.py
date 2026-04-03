import torch
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from train_lstm import LSTMModel, create_sequence, process_data, train_test_split

model = LSTMModel()
model.load_state_dict(torch.load("models/energy_lstm_model.pth"))
model.eval()

scaler = joblib.load('models/scaler')

data = process_data()

X, y = create_sequence(data, 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
X_test = torch.tensor(X_test, dtype = torch.float32)


with torch.no_grad():
    pred = model(X_test).numpy()

pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test, pred)

print("LSTM MAE: ", mae)


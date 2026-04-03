import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 50, batch_first = True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
    

scaler = MinMaxScaler()

def process_data():
    df = pd.read_csv('C:\Machine_Learning_Projects\energy-load-forecasting\smart-household-energy-forecasting\data\processed\data.csv')
    data = df['Global_active_power'].values
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)

    return data

def create_sequence(data, window_size = 60):

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)

def train_test_split(X, y, train_size):
    
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test

data = process_data()
X, y = create_sequence(data, 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")


torch.save(model.state_dict(), 'models/energy_lstm_model.pth')

joblib.dump(scaler, 'models/scaler.gz')








    

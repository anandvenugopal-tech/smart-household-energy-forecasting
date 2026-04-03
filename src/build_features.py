import pandas as pd

def build_features():

    df = pd.read_csv('C:\Machine_Learning_Projects\energy-load-forecasting\smart-household-energy-forecasting\data\processed\data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors = 'coerce')

    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Weekday'] = df['Datetime'].dt.weekday
    df['Hour'] = df['Datetime'].dt.hour

    df['lag_1'] = df['Global_active_power'].shift(1)
    df['lag_60'] = df['Global_active_power'].shift(60)
    df['lag_1440'] = df['Global_active_power'].shift(1440)

    df['rolling_60'] = df['Global_active_power'].rolling(60).mean()
    df['rolling_1440'] = df['Global_active_power'].rolling(1440).mean()

    df.dropna(inplace = True)

    df.to_csv('data/processed/household_features.csv', index = False)

if __name__ == "__main__":
    build_features()


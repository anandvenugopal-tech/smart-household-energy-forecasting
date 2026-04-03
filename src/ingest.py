"""
Loads the raw UCI Individual Household Power Consumption dataset, combines Date and Time 
into a single Datetime column, and reads the data into a DataFrame for further processing.
"""

import pandas as pd

def load_data():
    """
    Load the Individual Household Power Consumption dataset 
    """

    df = pd.read_csv(
        r"data\raw\household_power_consumption.txt",
        sep = ";",
        parse_dates = {'Datetime': ['Date', 'Time']}, 
        infer_datetime_format= True,
        na_values = "?",
        low_memory = False
        )
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())


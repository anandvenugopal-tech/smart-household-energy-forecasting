"""
Cleans the dataset by removing missing values, sorting by timestamp, resetting the index, 
and saving the cleaned version into data/processed/ for EDA and modeling.
"""

#import libraries, modules and functions
import pandas as pd
from ingest import load_data

def clean_data():
    """
    This funciton helps to clean the data and convert to csv file and save in data/processed.
    """

    df = load_data()
    df = df.sort_values('Datetime')
    df = df.reset_index(drop = True)
    df.to_csv(f"data/processed/data.csv", index = False)
    print(f"Saved the file")

if __name__ == "__main__":
    clean_data()
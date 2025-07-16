import pandas as pd
import os

# Define to manage filenames and their relative paths
DATA_FILES = {
    "credit": "creditcard/creditcard.csv",
    "fraud": "Fraud_Data.csv",
    "IpAddresss": "IpAddress_to_Country.csv"
}

def get_file_path(file_key):
    # Assuming this script is inside 'src' or 'scripts', adjust if needed
    current_dir = os.getcwd()
    # Base data directory relative to current dir
    base_data_dir = os.path.join(current_dir, "..", "Data")
    return os.path.join(base_data_dir, DATA_FILES[file_key])

class CSVDataloader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)

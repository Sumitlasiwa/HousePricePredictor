# src/data_loader.py

import pandas as pd
from src.config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    CLEANED_TRAIN_DATA_PATH,
    CLEANED_VALIDATION_DATA_PATH,
    CLEANED_TEST_DATA_PATH,
)

def load_raw_data():
    """Loads raw train, validation, and test datasets."""
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VALIDATION_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    return df_train, df_val, df_test

def load_cleaned_data():
    """Loads cleaned train, validation, and test datasets."""
    df_train = pd.read_csv(CLEANED_TRAIN_DATA_PATH)
    df_val = pd.read_csv(CLEANED_VALIDATION_DATA_PATH)
    df_test = pd.read_csv(CLEANED_TEST_DATA_PATH)
    return df_train, df_val, df_test

import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from scipy import stats
from src.config import (
    CLEANED_TRAIN_DATA_PATH,
    CLEANED_VALIDATION_DATA_PATH,
    CLEANED_TEST_DATA_PATH,
)


def feature_engineering():
    """Perform feature engineering on the raw datasets and save cleaned datasets."""
    
    df_train, df_val, df_test = load_raw_data()
    
    categorical_features = [ feature for feature in df_train.columns if df_train[feature].dtype == 'object']
    
    #removing furnishingstatus because it is ordinal categorical feature
    categorical_features = [feature for feature in categorical_features if feature != 'furnishingstatus']
    
    binary_mapping = {'yes':1, 'no':0}
    df_names = [df_train, df_val, df_test]
    for df_name in df_names:
        for feature in categorical_features:
            df_name[feature] = df_name[feature].map(binary_mapping)
            
            
    #ordinal encoding for furnishingstatus
    furnishing_mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    for df_name in df_names:
        df_name["furnishingstatus"] = df_name["furnishingstatus"].map(furnishing_mapping)
        
    #log transformation for skewed data
    for df_name in df_names:
        df_name['area'] = np.log1p(df_name['area'])
        df_name['bedrooms'] = np.log1p(df_name['bedrooms'])
        df_name['price'] = np.log1p(df_name['price'])
        
   
    #handling outliers using z-score method
    z_scores = np.abs(stats.zscore(df_train))
    df_train = df_train[(z_scores < 3).all(axis=1)]
    
        # Save cleaned datasets
    df_train.to_csv(CLEANED_TRAIN_DATA_PATH, index=False)
    df_val.to_csv(CLEANED_VALIDATION_DATA_PATH, index=False)
    df_test.to_csv(CLEANED_TEST_DATA_PATH, index=False)

    print("✅ Feature engineering complete and cleaned data saved.")
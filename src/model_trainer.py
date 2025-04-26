# src/model_trainer.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_cleaned_data
from src.config import RANDOM_STATE

def train_and_evaluate_models():
    """Train multiple models and evaluate their performance."""

    df_train, df_val, df_test = load_cleaned_data()

    X_train = df_train.drop('price', axis=1)
    y_train = df_train['price']

    X_val = df_val.drop('price', axis=1)
    y_val = df_val['price']

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(random_state=RANDOM_STATE)
    }

    best_rmse = float('inf')
    best_model = None
    best_model_name = ""
    
    # Training and Evaluation
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)

        print(f"Model: {name}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.2f}")
        print("-" * 30)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    # Save best model
    if best_model:
        os.makedirs('models', exist_ok=True)  # create folder if not exists
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

    print(f"✅ Best model '{best_model_name}' saved to 'models/best_model.pkl' with RMSE {best_rmse:.2f}")
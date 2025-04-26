# main.py

from src.feature_engineering import feature_engineering
from src.model_trainer import train_and_evaluate_models

def main():
    print("🚀 Starting feature engineering...")
    feature_engineering()

    print("\n🚀 Starting model training and evaluation...")
    train_and_evaluate_models()

if __name__ == "__main__":
    main()

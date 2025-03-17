import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans the dataset.
    - Converts Mileage and Price columns to numeric.
    - Converts dates with dayfirst=True.
    - Calculates 'COE Years Left' and 'Bike_Age'.
    - Fixes inconsistencies in categorical features.
    """
    # Load the dataset
    data = pd.read_excel(file_path)
    logging.info("Dataset loaded successfully.")

    # Clean Classification
    data['Classification'] = data['Classification'].str.replace(" ", "").str.upper()
    valid_classes = {'CLASS2B', 'CLASS2A', 'CLASS2'}
    data = data[data['Classification'].isin(valid_classes)]
    data['Classification'] = data['Classification'].astype('category')

    # Clean Mileage
    data['Mileage'] = pd.to_numeric(
        data['Mileage'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Mileage'])

    # Clean Price and filter positive values
    data['Price'] = pd.to_numeric(
        data['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data[data['Price'] > 0]  # Ensure positive prices for log transform

    # Clean Engine Capacity
    data['Engine Capacity'] = pd.to_numeric(
        data['Engine Capacity'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Engine Capacity'])

    # Clean No. of owners
    data['No. of owners'] = (
        data['No. of owners']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)  # Extract digits
        .astype(float)  # Convert to numeric
    )
    data = data.dropna(subset=['No. of owners'])

    # Date handling
    data['COE Expiry Date'] = pd.to_datetime(data['COE Expiry Date'], dayfirst=True)
    data['Registration Date'] = pd.to_datetime(data['Registration Date'], dayfirst=True)

    # Calculate "COE Years Left" and "Bike_Age"
    today = pd.to_datetime('today')
    data['COE Years Left'] = (data['COE Expiry Date'] - today).dt.days / 365.25
    data['Bike_Age'] = today.year - data['Registration Date'].dt.year

    # Feature engineering: Mileage per COE Year
    data['Mileage_per_COE_Year'] = data['Mileage'] / (data['COE Years Left'] + 1e-6)

    # Encode categorical features (Brand and Category)
    data['Brand'] = data['Brand'].astype('category')
    data['Category'] = data['Category'].astype('category')

    # Final feature set
    features = [
        'Classification', 'Mileage', 'COE Years Left', 'No. of owners',
        'Engine Capacity', 'Bike_Age', 'Mileage_per_COE_Year', 'Brand', 'Category'
    ]
    data = data.dropna(subset=features + ['Price'])
    
    return data

def split_data(data: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into train and test sets.
    """
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tune_model(X_train, y_train, cat_features: list):
    """
    Tune CatBoost model using RandomizedSearchCV.
    """
    param_grid = {
        'depth': [6, 8, 10],  # Deeper trees for better performance
        'learning_rate': [0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5],
    }

    model = CatBoostRegressor(
        iterations=2000,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,  # Number of parameter combinations to try
        cv=5,       # 5-fold cross-validation
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
    )
    grid_search.fit(X_train, y_train, cat_features=cat_features)
    return grid_search.best_estimator_

def evaluate_model(model: CatBoostRegressor, X_test, y_test, features: list):
    """
    Evaluate the model using R2, MSE, RMSE, and MAE, and print feature importances.
    """
    preds_log = model.predict(X_test)
    preds = np.exp(preds_log)  # Reverse log transform
    
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    logging.info(f"R^2 on the test set: {r2:.2f}")
    logging.info(f"MSE on the test set: {mse:.2f}")
    logging.info(f"RMSE on the test set: {rmse:.2f}")
    logging.info(f"MAE on the test set: {mae:.2f}")

    # Display feature importances
    importances = model.get_feature_importance()
    for feat, imp in zip(features, importances):
        logging.info(f"{feat}: {imp:.2f}")

def main():
    file_path = 'combined_dataset_latest.xlsx'
    features = [
        'Classification', 'Mileage', 'COE Years Left', 'No. of owners',
        'Engine Capacity', 'Bike_Age', 'Mileage_per_COE_Year', 'Brand', 'Category'
    ]
    cat_features = ['Classification', 'Brand', 'Category']  # Categorical features
    target = 'Price'

    # Load and clean data
    data = clean_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, features, target)
    
    # Log-transform the target
    y_train_log = np.log(y_train)
    
    # Train model
    model = tune_model(X_train, y_train_log, cat_features)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, features)
    
    # Save model
    with open('catboost_model_optimized.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info("✅ Optimized model saved!")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
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
    - Adds interaction features.
    - Removes outliers in Price.
    """
    # Load the dataset
    data = pd.read_excel(file_path)
    logging.info("Dataset loaded successfully.")

    # Clean Classification
    data['Classification'] = data['Classification'].str.replace(" ", "").str.upper()
    valid_classes = {'CLASS2B', 'CLASS2A', 'CLASS2'}
    data = data[data['Classification'].isin(valid_classes)]
    data['Classification'] = data['Classification'].astype('category')

    # Clean Mileage and log-transform
    data['Mileage'] = pd.to_numeric(
        data['Mileage'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Mileage'])
    data['Mileage_log'] = np.log1p(data['Mileage'])  # Log-transform Mileage for better normalization

    # Clean Price and filter positive values
    data['Price'] = pd.to_numeric(
        data['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data[data['Price'] > 0]  # Ensure positive prices for log transform

    # Outlier handling for Price
    price_q1 = data['Price'].quantile(0.25)
    price_q3 = data['Price'].quantile(0.75)
    iqr = price_q3 - price_q1
    data = data[(data['Price'] >= price_q1 - 1.5 * iqr) & (data['Price'] <= price_q3 + 1.5 * iqr)]

    # Clean Engine Capacity and apply log transformation
    data['Engine Capacity'] = pd.to_numeric(
        data['Engine Capacity'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Engine Capacity'])
    data['Engine_Capacity_log'] = np.log1p(data['Engine Capacity'])  # Log-transform Engine Capacity

    # Clean No. of owners
    data['No. of owners'] = (
        data['No. of owners']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )
    data = data.dropna(subset=['No. of owners'])

    # Date handling
    data['COE Expiry Date'] = pd.to_datetime(data['COE Expiry Date'], dayfirst=True)
    data['Registration Date'] = pd.to_datetime(data['Registration Date'], dayfirst=True)

    # Calculate "COE Years Left" and "Bike_Age"
    today = pd.to_datetime('today')
    data['COE Years Left'] = (data['COE Expiry Date'] - today).dt.days / 365.25
    data['Bike_Age'] = today.year - data['Registration Date'].dt.year

    # Feature engineering: Interaction features
    data['Engine_Capacity_x_Bike_Age'] = data['Engine Capacity'] * data['Bike_Age']
    data['Mileage_x_COE_Years_Left'] = data['Mileage'] * data['COE Years Left']
    data['Mileage_per_COE_Year'] = data['Mileage'] / (data['COE Years Left'] + 1e-6)
    
    # Polynomial features (degree 2 interactions)
    data['Mileage_squared'] = data['Mileage'] ** 2
    data['Engine_Capacity_squared'] = data['Engine Capacity'] ** 2
    data['Mileage_x_Engine_Capacity'] = data['Mileage'] * data['Engine Capacity']

    # Encode categorical features (Brand and Category)
    data['Brand'] = data['Brand'].astype('category')
    data['Category'] = data['Category'].astype('category')

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
        'depth': [6, 8, 10, 12, 14],  # Increased depth
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Lower learning rate for better convergence
        'l2_leaf_reg': [1, 3, 5, 10, 15],
        'random_strength': [1, 2, 3, 4, 5],
        'bagging_temperature': [0, 0.5, 1, 3, 5],  # Adjusting bagging to introduce more randomness
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'iterations': [5000, 10000, 15000],  # Larger iterations to allow more learning
        'subsample': [0.7, 0.8, 0.9, 1.0],  # Introduce more randomness during training
        'colsample_bylevel': [0.7, 0.8, 0.9],  # Control feature subsampling
        'max_bin': [128, 256, 512]  # Number of bins used for feature quantization
    }

    model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        od_type='Iter',  # Early stopping
        od_wait=100,
        verbose=False
    )

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,  # Increased number of iterations for a wider search
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    grid_search.fit(X_train, y_train, cat_features=cat_features)
    return grid_search.best_estimator_

def evaluate_model(model: CatBoostRegressor, X_test, y_test, features: list):
    """
    Evaluate the model using R2, MSE, RMSE, and MAE, and print feature importances.
    """
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)  # Reverse np.log1p transform

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
        'Classification', 'Mileage_log', 'COE Years Left', 'No. of owners',
        'Engine_Capacity_log', 'Bike_Age', 'Mileage_per_COE_Year', 'Brand', 'Category',
        'Engine_Capacity_x_Bike_Age', 'Mileage_x_COE_Years_Left', 
        'Mileage_squared', 'Engine_Capacity_squared', 'Mileage_x_Engine_Capacity'
    ]
    cat_features = ['Classification', 'Brand', 'Category']
    target = 'Price'

    data = clean_data(file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data, features, target)

    # Log1p transform of target
    y_train_log = np.log1p(y_train)

    # Tune model
    model = tune_model(X_train, y_train_log, cat_features)

    # Evaluate model
    evaluate_model(model, X_test, y_test, features)

    # Save optimized model
    with open('catboost_model_optimized_v2.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info("✅ Optimized model saved!")

if __name__ == "__main__":
    main()

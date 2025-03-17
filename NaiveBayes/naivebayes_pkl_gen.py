import pandas as pd
import numpy as np
import time
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
    Cleaning includes:
    - Converting Mileage and Price columns to numeric.
    - Converting dates with dayfirst=True.
    - Calculating 'COE Years Left'.
    """
    # Load the dataset
    data = pd.read_excel(file_path)
    logging.info("Dataset loaded successfully.")

    # Clean the Mileage column: remove non-numeric characters and convert to float safely.
    data['Mileage'] = pd.to_numeric(
        data['Mileage'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Mileage'])

    # Clean the Price column: remove non-numeric characters and convert to float safely.
    data['Price'] = pd.to_numeric(
        data['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    )
    data = data.dropna(subset=['Price'])

    # Convert date columns to datetime format with dayfirst=True
    data['COE Expiry Date'] = pd.to_datetime(data['COE Expiry Date'], dayfirst=True)
    data['Registration Date'] = pd.to_datetime(data['Registration Date'], dayfirst=True)

    # Calculate "COE Years Left" based on today's date
    today = pd.to_datetime('today')
    data['COE Years Left'] = (data['COE Expiry Date'] - today).dt.days / 365.25

    # Drop rows with missing values in the required feature columns
    features = ['Classification', 'Mileage', 'COE Years Left', 'No. of owners']
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
    Tune CatBoost model using RandomizedSearchCV for efficiency and accuracy balance.
    Implements early stopping to prevent unnecessary long training.
    """
    param_grid = {
        'iterations': [500, 1000, 1500],  # Reduced range for efficiency
        'depth': [6, 8, 10],  # Balanced depth values
        'learning_rate': [0.01, 0.05, 0.1],  # Different learning rates
        'l2_leaf_reg': [3, 5, 7]  # Regularization values
    }

    model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        task_type="GPU",  # Enable GPU if available
        verbose=0  # Suppress detailed training output
    )

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,  # Test only 15 random parameter sets
        cv=5,  # Keep CV=5 for best accuracy
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    start_time = time.time()  # Start time tracking
    grid_search.fit(X_train, y_train, cat_features=cat_features)
    total_time = time.time() - start_time  # Total training time

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return grid_search.best_estimator_


def evaluate_model(model: CatBoostRegressor, X_test, y_test, features: list):
    """
    Evaluate the model using R2, MSE, RMSE, and MAE, and print feature importances.
    """
    preds = model.predict(X_test)
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
    features = ['Classification', 'Mileage', 'COE Years Left', 'No. of owners']
    target = 'Price'
    cat_features = ['Classification']

    data = clean_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, features, target)

    # Tune and train the model
    model = tune_model(X_train, y_train, cat_features)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, features)


if __name__ == "__main__":
    main()

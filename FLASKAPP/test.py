import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app_v2 import load_dataset, models  # Import necessary functions and models from app_v2

# Ensure interactive mode is on for matplotlib to display plots
plt.ion()

def plot_residuals(y_true, y_pred, save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_error_distribution(y_true, y_pred, save_path=None):
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[indices])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_hexbin_actual_vs_predicted(y_true, y_pred, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hexbin(y_true, y_pred, gridsize=50, cmap='Blues')
    plt.colorbar(label='Count')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs. Predicted Prices (Hexbin)')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_feature_distribution(df, save_path=None):
    # Plot the distribution of all numerical features
    plt.figure(figsize=(12, 8))
    df.select_dtypes(include=[np.number]).hist(bins=30, figsize=(12, 8))
    plt.suptitle('Feature Distribution')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_correlation_heatmap(df, save_path=None):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("DataFrame contains no numeric columns for correlation.")
    
    # Compute the correlation matrix
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def evaluate_model(model_name):
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found.")
    model = models[model_name]
    df = load_dataset(sample=True)  # Load dataset

    # Print available columns & model's expected features
    print(f"📌 Available columns in df: {df.columns.tolist()}")
    print(f"📌 Model trained with features: {model.feature_names_in_.tolist()}")

    # Rename columns if needed
    feature_mapping = {
        'Engine Capacity': 'Engine_Capacity',
        'Registration Date': 'Registration_Date',
        'COE Expiry Date': 'COE_Expiry_Date',
        'Mileage': 'Mileage',
        'No. of owners': 'No._of_owners'
    }
    df = df.rename(columns=feature_mapping)

    # Get exact features expected by the model
    expected_features = model.feature_names_in_.tolist()

    # Ensure all expected features exist in df
    missing_features = [col for col in expected_features if col not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}. Adding default values.")
        for feature in missing_features:
            df[feature] = 0  # Assign default value

    # Select only expected features for X_test
    X_test = df[expected_features]
    
    # Convert categorical features to numeric (One-Hot Encoding or Label Encoding)
    for col in X_test.select_dtypes(include=['object']).columns:
        print(f"🔄 Encoding categorical column: {col}")
        X_test[col] = X_test[col].astype("category").cat.codes  # Label encoding

    y_test = df['Price']

    print(f"📌 Final X_test features: {X_test.columns.tolist()}")  # Debugging output
    print(f"📌 Data types in X_test:\n{X_test.dtypes}")  # Debugging output

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}')

    # Generate plots
    plot_residuals(y_test, y_pred, save_path='residual_plot.png')
    plot_error_distribution(y_test, y_pred, save_path='error_distribution.png')
    plot_feature_importance(model, expected_features, save_path='feature_importance.png')
    plot_hexbin_actual_vs_predicted(y_test, y_pred, save_path='hexbin_plot.png')

    # Optional: plot feature distribution and correlation heatmap
    plot_feature_distribution(df, save_path='feature_distribution.png')
    plot_correlation_heatmap(df, save_path='correlation_heatmap.png')

    return y_pred

# Run the evaluation
if __name__ == "__main__":
    evaluate_model("xgboost")

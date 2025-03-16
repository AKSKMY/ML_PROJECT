import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------------------------------------------------------
# 1) LOAD THE SAME DATASET
# -------------------------------------------------------------------------
# Adjust as needed if your file is in another location.
file_name = "combined_dataset_latest.xlsx"
possible_folders = ["Datasets", "..", "../Datasets", "./"]
file_path = None

for folder in possible_folders:
    potential_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder, file_name))
    if os.path.exists(potential_path):
        file_path = potential_path
        break

if not file_path:
    raise FileNotFoundError(f"❌ Could not find {file_name} in {possible_folders}")

print(f"✅ Using dataset for inference: {file_path}")
df = pd.read_excel(file_path)

# -------------------------------------------------------------------------
# 2) REPLICATE THE SAME DATA CLEANING AS IN TRAINING
# -------------------------------------------------------------------------
print("📊 Actual columns in dataset:", df.columns.tolist())

# Standardize column names (strip whitespace)
df.columns = df.columns.str.strip()

# The same features you used in training
features = [
    'Bike Name', 'Brand', 'Model', 'Engine Capacity', 'Classification',
    'Registration Date', 'COE Expiry Date', 'Mileage', 'No. of owners', 'Category'
]
target = 'Price'

# Ensure required columns exist
missing_features = [col for col in features + [target] if col not in df.columns]
if missing_features:
    raise KeyError(f"❌ Missing columns in dataset: {missing_features}")

# Handle missing values for numeric columns by median (same approach as training)
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove "SGD$" or non-numeric from "Price" and convert to float
df['Price'] = (
    df['Price']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
    .astype(float)
)

# Make sure "Model" is kept as string
df['Model'] = df['Model'].astype(str)

# Fix non-numeric values in "Mileage" (identical to training)
df['Mileage'] = (
    df['Mileage']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)
df['Mileage'].replace('', np.nan, inplace=True)
df['Mileage'] = df['Mileage'].astype(float)
df['Mileage'].fillna(df['Mileage'].median(), inplace=True)

# Fix non-numeric values in "Engine Capacity"
df['Engine Capacity'] = (
    df['Engine Capacity']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)
df['Engine Capacity'].replace('', np.nan, inplace=True)
df['Engine Capacity'] = df['Engine Capacity'].astype(float)
df['Engine Capacity'].fillna(df['Engine Capacity'].median(), inplace=True)

# Fix non-numeric values in "No. of owners"
df['No. of owners'] = (
    df['No. of owners']
    .astype(str)
    .str.extract(r'(\d+)')  # Extract only digits
)
df['No. of owners'].replace('', np.nan, inplace=True)
df['No. of owners'] = df['No. of owners'].astype(float)
df['No. of owners'].fillna(df['No. of owners'].median(), inplace=True)

# Convert date columns to just the year (ints)
df['Registration Date'] = pd.to_datetime(df['Registration Date'], errors='coerce').dt.year
df['COE Expiry Date'] = pd.to_datetime(df['COE Expiry Date'], errors='coerce').dt.year

# -------------------------------------------------------------------------
# 3) RE-APPLY THE SAME LABEL ENCODERS TO THE SAME COLUMNS
#    (In training, you label-encoded 'Brand' and 'Category' only)
# -------------------------------------------------------------------------
label_encoders_path = os.path.join("saved_models", "label_encoders.pkl")
if not os.path.exists(label_encoders_path):
    raise FileNotFoundError(f"❌ Could not find label_encoders.pkl at {label_encoders_path}")

label_encoders = joblib.load(label_encoders_path)

for col in ['Brand', 'Category']:
    if col in df.columns and col in label_encoders:
        le = label_encoders[col]
        df[col] = le.transform(df[col])
    else:
        print(f"⚠️ Warning: Column '{col}' missing or not in label_encoders.")

# -------------------------------------------------------------------------
# 4) PREPARE X AND y THE SAME WAY AS TRAINING
#    (EXCLUDING 'Bike Name', 'Model', 'Classification' from X)
# -------------------------------------------------------------------------
X = df[features].drop(columns=['Bike Name', 'Model', 'Classification'])
y = df[target]

# If any new NaNs appear, fill them
if X.isna().sum().sum() > 0:
    X.fillna(X.median(numeric_only=True), inplace=True)
if y.isna().sum() > 0:
    y.fillna(y.median(), inplace=True)

# -------------------------------------------------------------------------
# 5) LOAD THE SCALER AND APPLY TO THE SAME NUMERIC COLUMNS
# -------------------------------------------------------------------------
scaler_path = os.path.join("saved_models", "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Could not find scaler.pkl at {scaler_path}")

scaler = joblib.load(scaler_path)

# The numeric columns used in training
numeric_features = ['Engine Capacity', 'Registration Date', 'COE Expiry Date', 'Mileage', 'No. of owners']
# Scale them in-place
X[numeric_features] = scaler.transform(X[numeric_features])

# Check if we need to apply polynomial features
poly_path = os.path.join("saved_models", "poly_features.pkl")
if os.path.exists(poly_path):
    print("✅ Loading polynomial features transformer")
    poly = joblib.load(poly_path)
    X = poly.transform(X)

print("\n✅ Data preprocessing complete. Final shape of X:", X.shape)

# -------------------------------------------------------------------------
# 6) LOAD EACH MODEL, PREDICT, AND COMPUTE REGRESSION METRICS
# -------------------------------------------------------------------------
models_info = {
    "Random Forest": "random_forest_regressor.pkl",
    "XGBoost":       "xgboost_regressor.pkl",
    "LightGBM":      "lightgbm_regressor.pkl",
    "SVM":           "svm_regressor.pkl"
}

results = []

for model_name, filename in models_info.items():
    model_path = os.path.join("saved_models", filename)
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_name} - file not found: {model_path}")
        continue

    model = joblib.load(model_path)
    predictions = model.predict(X)  # We used X with the same shape/columns as in training

    # Check if SVM uses log transform and apply inverse transform if needed
    if model_name == "SVM":
        metadata_path = os.path.join("saved_models", "svm_model_metadata.pkl")
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                if metadata.get("log_transform", False):
                    print(f"✅ Applying inverse log transform to {model_name} predictions")
                    predictions = np.expm1(predictions)  # Reverse log transformation
            except Exception as e:
                print(f"⚠️ Error loading model metadata: {e}")

    # Calculate metrics
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)

    results.append({
        "Model":  model_name,
        "R^2":    r2,
        "MSE":    mse,
        "RMSE":   rmse,
        "MAE":    mae
    })

# -------------------------------------------------------------------------
# 7) DISPLAY THE RESULTS
# -------------------------------------------------------------------------
if results:
    results_df = pd.DataFrame(results)
    print("\nPerformance of All Regressor Models:")
    print(results_df.to_string(index=False))
else:
    print("\nNo models were evaluated because none were found or loaded.")
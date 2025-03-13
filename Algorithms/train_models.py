import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import joblib

# ✅ Get absolute path dynamically
dataset_filenames = ["combined_dataset_latest.xlsx"]
dataset_folders = ["../Datasets", "./", "../"]
file_path = None

# ✅ Search for dataset in multiple locations
for folder in dataset_folders:
    for filename in dataset_filenames:
        potential_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder, filename))
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    if file_path:
        break

# ❌ If dataset is not found, raise an error
if not file_path:
    raise FileNotFoundError("❌ Dataset not found. Ensure it exists in the correct directory.")

print(f"✅ Using dataset: {file_path}")

# ✅ Load dataset
df = pd.read_excel(file_path)

# ✅ Print actual column names (for debugging)
print("📊 Actual columns in dataset:", df.columns.tolist())

# ✅ Standardize column names
df.columns = df.columns.str.strip()

# ✅ Define correct column names
features = ['Bike Name', 'Brand', 'Model', 'Engine Capacity', 'Classification',
            'Registration Date', 'COE Expiry Date', 'Mileage', 'No. of owners', 'Category']
target = 'Price'

# ✅ Ensure all required columns exist
missing_features = [col for col in features + [target] if col not in df.columns]
if missing_features:
    raise KeyError(f"❌ Missing columns in dataset: {missing_features}")

# ✅ Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# ✅ Remove "SGD$" from "Price" and convert to float
df['Price'] = df['Price'].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

# ✅ Keep "Model" as a string
df['Model'] = df['Model'].astype(str)  # ⬅ Ensures Model stays as a string

# ✅ Fix non-numeric values in "Mileage"
df['Mileage'] = df['Mileage'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df['Mileage'].replace('', np.nan, inplace=True)  # ✅ Replace empty strings with NaN
df['Mileage'] = df['Mileage'].astype(float)
df['Mileage'].fillna(df['Mileage'].median(), inplace=True)  # ✅ Fill NaN with median value

# ✅ Fix non-numeric values in "Engine Capacity"
df['Engine Capacity'] = df['Engine Capacity'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df['Engine Capacity'].replace('', np.nan, inplace=True)
df['Engine Capacity'] = df['Engine Capacity'].astype(float)
df['Engine Capacity'].fillna(df['Engine Capacity'].median(), inplace=True)

# ✅ Fix non-numeric values in "No. of owners"
df['No. of owners'] = df['No. of owners'].astype(str).str.extract('(\d+)')
df['No. of owners'].replace('', np.nan, inplace=True)
df['No. of owners'] = df['No. of owners'].astype(float)
df['No. of owners'].fillna(df['No. of owners'].median(), inplace=True)

# ✅ Convert date columns to just the year
df['Registration Date'] = pd.to_datetime(df['Registration Date'], errors='coerce').dt.year
df['COE Expiry Date'] = pd.to_datetime(df['COE Expiry Date'], errors='coerce').dt.year

# ✅ Encode categorical features (excluding "Model")
categorical_features = ['Brand', 'Category']  # ⬅ "Model" is removed from encoding
label_encoders = {}

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:
        print(f"⚠️ Warning: Column '{col}' not found in dataset")

# ✅ Extract features and target (EXCLUDE "Model" from training)
X = df[features].drop(columns=['Bike Name', 'Model', 'Classification'])  # ⬅ Price is NOT dropped anymore
y = df[target]

# ✅ Ensure no NaN values are left in the dataset
if X.isna().sum().sum() > 0:
    print("⚠️ Warning: NaN values found in X, filling them with median values.")
    X.fillna(X.median(numeric_only=True), inplace=True)

if y.isna().sum() > 0:
    print("⚠️ Warning: NaN values found in y, filling them with median values.")
    y.fillna(y.median(), inplace=True)

# ✅ Define numeric features (excluding "Model")
numeric_features = ['Engine Capacity', 'Registration Date', 'COE Expiry Date', 'Mileage', 'No. of owners']

# ✅ Standardize only numeric columns (excluding "Model")
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Debugging: Print X shape to confirm "Model" is removed
print(f"🔹 X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("🔍 Features used for training:", X_train.columns.tolist())

# ✅ Train and evaluate models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(C=10, kernel='rbf', gamma='scale')  # ✅ Ensure SVM is included
}

# ✅ Create directory to save models
models_directory = "saved_models"
os.makedirs(models_directory, exist_ok=True)

# ✅ Train and save all models
for name, model in models.items():
    print(f"🔄 Training {name} model...")
    model.fit(X_train, y_train)

    # ✅ Save trained model
    model_filename = f"{models_directory}/{name.lower().replace(' ', '_')}_regressor.pkl"
    joblib.dump(model, model_filename)
    print(f"💾 Saved: {model_filename}")

# ✅ Save preprocessing objects
joblib.dump(label_encoders, os.path.join(models_directory, "label_encoders.pkl"))
joblib.dump(scaler, os.path.join(models_directory, "scaler.pkl"))

print("\n✅ All models trained and saved successfully!")

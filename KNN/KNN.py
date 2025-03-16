import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Update dataset path
file_path = 'combined_dataset_latest.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Define column assumptions:
# df.columns[1] -> Bike Brand
# df.columns[2] -> Bike Model
# df.columns[5] -> Mileage
# 'Price' is the target column.
brand_col = df.columns[1]
bike_model_col = df.columns[2]
mileage_col = df.columns[5]
target_col = 'Price'

# Clean the Price column: remove currency symbols and commas, then convert to numeric
df[target_col] = df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Clean the Mileage column: remove non-numeric characters and convert to numeric
df[mileage_col] = df[mileage_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df[mileage_col] = pd.to_numeric(df[mileage_col], errors='coerce')

# Ensure categorical columns are strings
df[brand_col] = df[brand_col].astype(str)
df[bike_model_col] = df[bike_model_col].astype(str)

# Drop rows with missing target or mileage values
df = df.dropna(subset=[target_col, mileage_col])

# Use bike brand, bike model, and mileage as features
X = df[[brand_col, bike_model_col, mileage_col]]
y = df[target_col]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: one-hot encode categorical features and scale numeric features
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), [brand_col, bike_model_col]),
    ('num', StandardScaler(), [mileage_col])
])

# Create a pipeline with the preprocessor and KNN regressor
model = make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=5))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics in a table-like format
print("\nPerformance of KNN Regressor Model:")
print(f"{'Model':<10} {'R^2':<8} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
print(f"{'KNN':<10} {r2:<8.4f} {mse:<12.2f} {rmse:<12.2f} {mae:<12.2f}")

def predict_bike_price(brand, bike_model, mileage):
    # Create a DataFrame for the input data ensuring the column names match those used in training
    input_df = pd.DataFrame({
        brand_col: [brand],
        bike_model_col: [bike_model],
        mileage_col: [mileage]
    })
    predicted_price = model.predict(input_df)[0]
    return predicted_price

# Get user inputs
brand = input("\nEnter the bike brand: ").strip()
bike_model = input("Enter the bike model: ").strip()
mileage = input("Enter the mileage: ").strip()

try:
    mileage = float(mileage)
except ValueError:
    print("Mileage must be a numeric value.")
    mileage = 0.0

result = predict_bike_price(brand, bike_model, mileage)
print(f"Predicted Price: SGD {result:.2f}")

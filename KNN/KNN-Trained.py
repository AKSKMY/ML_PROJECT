import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

# 1. Load and Clean the Dataset
file_path = 'combined_dataset_latest.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Assume columns[1] -> Brand, columns[2] -> Model, columns[5] -> Mileage, 'Price' is target
brand_col = df.columns[1]
bike_model_col = df.columns[2]
mileage_col = df.columns[5]
target_col = 'Price'

# Clean 'Price' column: remove non-numeric chars
df[target_col] = df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Clean 'Mileage' column: remove non-numeric chars
df[mileage_col] = df[mileage_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df[mileage_col] = pd.to_numeric(df[mileage_col], errors='coerce')

# Ensure categorical columns are strings
df[brand_col] = df[brand_col].astype(str)
df[bike_model_col] = df[bike_model_col].astype(str)

# Drop rows with missing Price or Mileage
df.dropna(subset=[target_col, mileage_col], inplace=True)

# 2. Prepare Features (X) and Target (y)
X = df[[brand_col, bike_model_col, mileage_col]]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build a Pipeline (OneHot + StandardScaler + KNN)
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), [brand_col, bike_model_col]),
    ('num', StandardScaler(), [mileage_col])
])

model = make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=5))

# 4. Train the Model
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# 5. Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPerformance of KNN Regressor Model:")
print(f"{'Model':<10} {'R^2':<8} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'TrainTime(s)':<15}")
print(f"{'KNN':<10} {r2:<8.4f} {mse:<12.2f} {rmse:<12.2f} {mae:<12.2f} {training_time:<15.2f}")

# 6. Save the Trained Model as .pkl
joblib.dump(model, 'knn_regressor.pkl')
print("\nKNN model saved to knn_regressor.pkl")

# 7. (Optional) Predict a Price from User Input
def predict_bike_price(brand, bike_model, mileage):
    # Create a DataFrame for the new data
    input_df = pd.DataFrame({
        brand_col: [brand],
        bike_model_col: [bike_model],
        mileage_col: [mileage]
    })
    return model.predict(input_df)[0]

brand = input("\nEnter the bike brand: ").strip()
bike_model = input("Enter the bike model: ").strip()
mileage_input = input("Enter the mileage: ").strip()

try:
    mileage_input = float(mileage_input)
except ValueError:
    print("Mileage must be numeric; using 0.0 as default.")
    mileage_input = 0.0

price_prediction = predict_bike_price(brand, bike_model, mileage_input)
print(f"Predicted Price: SGD {price_prediction:.2f}")

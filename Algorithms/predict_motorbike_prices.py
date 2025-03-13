import joblib
import numpy as np
import os

# Define model directory
models_directory = "saved_models"

# Load available trained models
models = {}
available_models = ["random_forest", "xgboost", "lightgbm", "svm"]

for model_name in available_models:
    model_path = os.path.join(models_directory, f"{model_name}_regressor.pkl")
    if os.path.exists(model_path):
        models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded {model_name.upper()} model.")

# Load encoders and scaler
label_encoders = joblib.load(os.path.join(models_directory, "label_encoders.pkl"))
scaler = joblib.load(os.path.join(models_directory, "scaler.pkl"))

# Function to preprocess input data
def preprocess_input(sample_input):
    """Encodes categorical variables and scales numeric features for model prediction."""
    
    # Encode categorical variables
    for col in ['Bike Brand', 'Variant/Model Year', 'Market Segment']:
        if col in label_encoders:
            sample_input[col] = label_encoders[col].transform([sample_input[col]])[0]
    
    # Prepare feature array
    numeric_features = ['Engine Size (cc)', 'Year of Registration', 'COE Expiry Year', 'Total Mileage (km)', 'Number of Previous Owners']
    sample_values = np.array([sample_input[col] for col in sample_input]).reshape(1, -1)
    
    # Standardize numerical features
    sample_values[:, :5] = scaler.transform(sample_values[:, :5])
    
    return sample_values

# Function to predict motorbike price using all models
def predict_motorbike_price(sample_input):
    """Predicts the resale price using multiple models and selects the best prediction."""
    
    X_input = preprocess_input(sample_input)
    
    predictions = {}
    
    for model_name, model in models.items():
        predicted_price = model.predict(X_input)[0]
        predictions[model_name] = predicted_price
        print(f"🔹 {model_name.upper()} Predicted Price: SGD {predicted_price:.2f}")

    # Select the model with the most reasonable prediction
    best_model = min(predictions, key=predictions.get)  # Select model predicting lowest price
    print(f"\n✅ Best Model Selected: {best_model.upper()}")
    print(f"💰 Predicted Secondhand Motorbike Price: SGD {predictions[best_model]:.2f}")

    return predictions[best_model]

# Example input for prediction
sample_motorbike = {
    'Bike Brand': "Yamaha",
    'Variant/Model Year': "R15 2020",
    'Engine Size (cc)': 155,
    'Year of Registration': 2020,
    'COE Expiry Year': 2030,
    'Total Mileage (km)': 25000,
    'Number of Previous Owners': 1,
    'Market Segment': "Budget"
}

# Run prediction
predicted_price = predict_motorbike_price(sample_motorbike)

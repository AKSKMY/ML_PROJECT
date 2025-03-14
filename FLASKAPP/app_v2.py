from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import sys
import importlib.util

# Get the absolute path to the parent directory (ML_PROJECT)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths to templates and static folders in the parent directory
template_dir = os.path.join(parent_dir, 'templates')
static_dir = os.path.join(parent_dir, 'static')

# Initialize Flask app with correct template and static paths
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'motorbike_price_prediction'

# Print paths for debugging
print(f"🔍 App directory: {os.path.abspath(__file__)}")
print(f"🔍 Template directory: {template_dir}")
print(f"🔍 Static directory: {static_dir}")

# Path to models and preprocessing files
models_directory = os.path.join(parent_dir, "saved_models")
available_models = ["random_forest", "xgboost", "lightgbm", "svm"]

# Load models
models = {}
for model_name in available_models:
    model_path = os.path.join(models_directory, f"{model_name}_regressor.pkl")
    if os.path.exists(model_path):
        try:
            models[model_name] = joblib.load(model_path)
            print(f"✅ Loaded {model_name.upper()} model.")
        except Exception as e:
            print(f"⚠️ Error loading {model_name} model: {e}")

# Check for SVM model directory
SVM_RESULTS_DIR = os.path.join(parent_dir, "SVM", "results")
os.makedirs(SVM_RESULTS_DIR, exist_ok=True)

# Try to load encoders and scaler
try:
    label_encoders = joblib.load(os.path.join(models_directory, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(models_directory, "scaler.pkl"))
    print("✅ Loaded preprocessing objects.")
except Exception as e:
    print(f"⚠️ Error loading preprocessing objects: {e}")
    label_encoders = {}
    scaler = None

# Read selected_model.txt if it exists
try:
    with open(os.path.join(parent_dir, "selected_model.txt"), "r") as f:
        default_model = f.read().strip().lower()
        if default_model in available_models:
            print(f"✅ Read selected model from file: {default_model}")
        else:
            default_model = "random_forest"
            print(f"⚠️ Unknown model in selected_model.txt. Using default: {default_model}")
except:
    default_model = "random_forest"
    print(f"⚠️ Could not read selected_model.txt. Using default: {default_model}")

# User authentication (simple dictionary)
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# Admin control panel filters
admin_selected_filters = {
    "license_class": True,
    "mileage_range": True,
    "coe_left_range": True,
    "previous_owners": True
}

# System statistics
system_stats = {
    "prediction_count": 0,
    "last_retrained": "Never",
    "system_load": "Low"
}

# Cache for model performance metrics
model_metrics_cache = {}

# ------------------------ HELPER FUNCTIONS ------------------------

def load_dataset(sample=False):
    """Try to load the motorcycle dataset for analysis"""
    # Look for dataset files in common locations
    dataset_names = ["combined_dataset_latest.xlsx", "Latest_Dataset.xlsx", "bike_data.xlsx"]
    search_dirs = [
        os.path.join(parent_dir, "Datasets"),
        os.path.join(parent_dir, "NewStuff"),
        parent_dir,
        os.path.dirname(os.path.abspath(__file__))
    ]
    
    for search_dir in search_dirs:
        for dataset_name in dataset_names:
            potential_path = os.path.join(search_dir, dataset_name)
            if os.path.exists(potential_path):
                try:
                    df = pd.read_excel(potential_path)
                    if sample and len(df) > 100:
                        # Return a sample for visualization purposes
                        return df.sample(n=100, random_state=42)
                    return df
                except Exception as e:
                    print(f"⚠️ Error loading {potential_path}: {e}")
    
    # If no dataset is found, create a synthetic one for demo purposes
    print("⚠️ No dataset found. Creating synthetic data for demonstration.")
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic data
    brands = ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "KTM", "Triumph", "Harley-Davidson"]
    models = ["CBR", "R1", "Ninja", "GSX-R", "Panigale", "S1000RR", "RC", "Street Triple", "Sportster"]
    categories = ["Sport", "Naked", "Cruiser", "Touring", "Scooter"]
    
    data = {
        "Brand": np.random.choice(brands, n_samples),
        "Model": np.random.choice(models, n_samples),
        "Engine Capacity": np.random.randint(125, 1200, n_samples),
        "Registration Date": np.random.randint(2010, 2024, n_samples),
        "COE Expiry Date": np.random.randint(2024, 2034, n_samples),
        "Mileage": np.random.randint(1000, 50000, n_samples),
        "No. of owners": np.random.randint(1, 4, n_samples),
        "Category": np.random.choice(categories, n_samples),
        "Price": np.random.randint(5000, 25000, n_samples)
    }
    
    return pd.DataFrame(data)

def calculate_model_metrics(model_name, force_recalculate=False):
    """Calculate or retrieve metrics for a model"""
    # Return cached metrics if available and not forcing recalculation
    if model_name in model_metrics_cache and not force_recalculate:
        return model_metrics_cache[model_name]
    
    # If model doesn't exist, return None
    if model_name not in models:
        return None
    
    try:
        # Load the dataset
        df = load_dataset()
        
        # Identify price column (target)
        price_columns = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
        for col in price_columns:
            if col in df.columns:
                target_col = col
                break
        else:
            target_col = df.columns[-1]
        
        # Clean price column if needed
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        
        # Extract features (all columns except price)
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Apply label encoding for categorical features
        for col in feature_cols:
            if col in label_encoders:
                df[col] = df[col].astype(str)
                df[col] = label_encoders[col].transform(df[col])
        
        # Split data into features and target
        X = df[feature_cols]
        y = df[target_col]
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = models[model_name].predict(X_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        # Calculate accuracy as percentage of predictions within 20% of actual value
        accuracy = np.mean(np.abs(predictions - y) / y <= 0.2) * 100
        
        # Create scatter plot of actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'{model_name.upper()}: Actual vs Predicted Motorcycle Prices')
        plt.tight_layout()
        img_path = os.path.join(SVM_RESULTS_DIR, f'{model_name}_actual_vs_predicted.png')
        plt.savefig(img_path)
        plt.close()
        
        # Create error distribution plot
        errors = y - predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name.upper()}: Error Distribution')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        error_path = os.path.join(SVM_RESULTS_DIR, f'{model_name}_error_distribution.png')
        plt.savefig(error_path)
        plt.close()
        
        # Store metrics in cache
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy': float(accuracy),
            'img_path': img_path,
            'error_path': error_path
        }
        
        model_metrics_cache[model_name] = metrics
        return metrics
    
    except Exception as e:
        print(f"⚠️ Error calculating metrics for {model_name}: {e}")
        return None

def calculate_all_model_metrics(force_recalculate=False):
    """Calculate metrics for all models"""
    results = {}
    for model_name in models:
        metrics = calculate_model_metrics(model_name, force_recalculate)
        if metrics:
            results[model_name] = metrics
    return results

def predict_price(input_data, model_name=default_model):
    """Predict price using the specified model with enhanced SVM responsiveness"""
    print(f"📊 Making prediction with {model_name} model")
    print(f"📊 Input data: {input_data}")
    
    if model_name not in models:
        print(f"❌ Model {model_name} not found")
        return None, "Model not found"
    
    try:
        # Determine the features needed by the model
        model_features = None
        if hasattr(models[model_name], 'feature_names_in_'):
            model_features = list(models[model_name].feature_names_in_)
            print(f"✅ Model expects these features: {model_features}")
        
        # If we can't determine from model, check expected number of features
        if not model_features:
            # For SVR models, try to get n_features_in_
            if hasattr(models[model_name], 'n_features_in_'):
                n_features = models[model_name].n_features_in_
                print(f"✅ Model expects {n_features} features")
                
                # Based on the error message, SVR expects 7 features including Brand and Category
                if n_features == 7:
                    model_features = ['Engine Capacity', 'Registration Date', 'COE Expiry Date', 
                                     'Mileage', 'No. of owners', 'Brand', 'Category']
                elif n_features == 5:
                    model_features = ['Engine Capacity', 'Registration Date', 'COE Expiry Date', 
                                     'Mileage', 'No. of owners']
                else:
                    print(f"⚠️ Unknown feature count: {n_features}, using all input features")
                    model_features = list(input_data.keys())
            else:
                # Default to all input features
                model_features = list(input_data.keys())
                print(f"⚠️ Could not determine expected features, using all input features: {model_features}")
        
        # Create a feature vector with the right features in the right order
        feature_vector = []
        for feature in model_features:
            if feature in input_data:
                feature_vector.append(input_data[feature])
            else:
                print(f"⚠️ Missing feature: {feature}, using default value")
                # Use default values for missing features
                if feature == 'Engine Capacity':
                    feature_vector.append(150)
                elif feature == 'Registration Date':
                    feature_vector.append(2020)
                elif feature == 'COE Expiry Date':
                    feature_vector.append(2030)
                elif feature == 'Mileage':
                    feature_vector.append(10000)
                elif feature == 'No. of owners':
                    feature_vector.append(1)
                elif feature == 'Brand' or feature == 'Category':
                    feature_vector.append(0)
                else:
                    feature_vector.append(0)
        
        # Convert to numpy array
        X = np.array(feature_vector).reshape(1, -1)
        print(f"✅ Feature vector created with shape: {X.shape}")
        
        # Apply scaling if we have a scaler and the right number of features
        # Get scaler feature count
        scaler_feature_count = 0
        if hasattr(scaler, 'n_features_in_'):
            scaler_feature_count = scaler.n_features_in_
        
        if scaler is not None and X.shape[1] == scaler_feature_count:
            # Standard case - scaler and feature count match
            X_scaled = scaler.transform(X)
            print("✅ Applied standard scaling")
        elif scaler is not None and X.shape[1] > scaler_feature_count:
            # Handle case where model needs more features than scaler knows about
            # Scale only the features the scaler knows about
            scaler_features = model_features[:scaler_feature_count]
            print(f"⚠️ Scaling only these features: {scaler_features}")
            
            # Extract just the features for scaling
            X_to_scale = X[:, :scaler_feature_count]
            X_scaled_partial = scaler.transform(X_to_scale)
            
            # Reconstruct the full feature vector with scaled values
            X_scaled = np.zeros(X.shape)
            X_scaled[:, :scaler_feature_count] = X_scaled_partial
            X_scaled[:, scaler_feature_count:] = X[:, scaler_feature_count:]
            print("✅ Applied partial scaling")
        elif scaler is not None and X.shape[1] < scaler_feature_count:
            # This shouldn't happen given our feature extraction, but just in case
            print("⚠️ Feature count less than scaler expects, cannot scale properly")
            X_scaled = X  # Skip scaling
        else:
            # No scaler or other issue
            X_scaled = X
            print("⚠️ No scaling applied")
        
        # Get base prediction from model
        base_prediction = models[model_name].predict(X_scaled)[0]
        print(f"✅ Base prediction: ${base_prediction:.2f}")
        
        # For SVM model, apply adjustments to ensure input changes are reflected in the prediction
        if model_name.lower() == 'svm':
            print("🔄 Applying SVM-specific adjustments for responsiveness")
            
            # Store original prediction
            predicted_price = base_prediction
            
            # 1. Adjust for COE years left
            if 'COE Expiry Date' in input_data:
                current_year = 2025  # Current year
                coe_expiry = input_data['COE Expiry Date']
                years_left = max(0, coe_expiry - current_year)
                
                # Apply significant adjustment based on COE years left (4% per year)
                coe_factor = 1.0 + (years_left * 0.04)
                predicted_price *= coe_factor
                print(f"📅 Applied COE adjustment: {years_left} years left → factor: {coe_factor:.2f}")
            
            # 2. Adjust for number of owners (8% reduction per additional owner)
            if 'No. of owners' in input_data:
                num_owners = input_data['No. of owners']
                if num_owners > 1:
                    owner_factor = 1.0 - ((num_owners - 1) * 0.08)
                    predicted_price *= owner_factor
                    print(f"👥 Applied owner adjustment: {num_owners} owners → factor: {owner_factor:.2f}")
            
            # 3. Adjust for mileage (higher mileage = lower price)
            if 'Mileage' in input_data:
                mileage = input_data['Mileage']
                if mileage > 20000:
                    # Higher mileage reduces price (up to 20% reduction)
                    mileage_factor = 1.0 - min(0.2, (mileage - 20000) / 100000)
                    predicted_price *= mileage_factor
                    print(f"🛣️ Applied mileage adjustment: {mileage}km → factor: {mileage_factor:.2f}")
            
            # 4. Adjust for engine capacity (bigger engine = higher price)
            if 'Engine Capacity' in input_data:
                engine_cc = input_data['Engine Capacity']
                if engine_cc > 400:  # Adjust for bigger bikes
                    engine_factor = 1.0 + min(0.3, (engine_cc - 400) / 1000)
                    predicted_price *= engine_factor
                    print(f"🔧 Applied engine adjustment: {engine_cc}cc → factor: {engine_factor:.2f}")
                    
            print(f"✅ Adjusted SVM prediction: ${predicted_price:.2f}")
        else:
            # For other models, use the base prediction as is
            predicted_price = base_prediction
        
        # Update prediction count
        system_stats["prediction_count"] += 1
        
        print(f"✅ Final prediction: ${predicted_price:.2f}")
        return predicted_price, None
    
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

# ------------------------ LOGIN SYSTEM ------------------------

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('user_dashboard') if session.get('role') == 'user' else url_for('admin_panel'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username]['password'] == password:
            session['user_id'] = username
            session['role'] = users[username]['role']
            return redirect(url_for('user_dashboard') if session['role'] == 'user' else url_for('admin_panel'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

# ------------------------ ADMIN PANEL ------------------------

@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        admin_selected_filters["license_class"] = 'license_class' in request.form
        admin_selected_filters["mileage_range"] = 'mileage_range' in request.form
        admin_selected_filters["coe_left_range"] = 'coe_left_range' in request.form
        admin_selected_filters["previous_owners"] = 'previous_owners' in request.form
        flash("Settings updated successfully.", "success")
    
    return render_template('admin.html', filters=admin_selected_filters)

@app.route('/get_filters')
def get_filters():
    return jsonify(admin_selected_filters)

# ------------------------ MODEL API ENDPOINTS ------------------------

@app.route('/update_model', methods=['POST'])
def update_model():
    """Updates the selected model for predictions."""
    global default_model
    
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
    
    selected_model = request.form.get('model')

    if selected_model in models:
        default_model = selected_model
        
        # Write to selected_model.txt
        try:
            with open(os.path.join(parent_dir, "selected_model.txt"), "w") as f:
                f.write(selected_model)
        except Exception as e:
            print(f"⚠️ Error writing to selected_model.txt: {e}")
        
        flash(f"✅ Model updated to {selected_model.upper()}!", "success")
    else:
        flash("❌ Invalid model selection!", "danger")

    return redirect(url_for('admin_panel'))

@app.route('/model_status')
def model_status():
    """Returns available models and the currently selected model."""
    available = [model_name for model_name in models]
    
    return jsonify({
        "available_models": available,
        "selected_model": default_model
    })

@app.route('/api/model_metrics')
def api_model_metrics():
    """Returns metrics for all models or a specific model."""
    model_name = request.args.get('model')
    force_recalculate = request.args.get('force_recalculate', '0') == '1'
    
    if model_name and model_name in models:
        metrics = calculate_model_metrics(model_name, force_recalculate)
        return jsonify(metrics)
    else:
        metrics = calculate_all_model_metrics(force_recalculate)
        return jsonify(metrics)

@app.route('/api/system_stats')
def api_system_stats():
    """Returns system statistics."""
    return jsonify(system_stats)

@app.route('/api/prediction', methods=['POST'])
def api_prediction():
    """API endpoint for making predictions."""
    model_name = request.json.get('model', default_model)
    input_data = request.json.get('input_data', {})
    
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400
    
    predicted_price, error = predict_price(input_data, model_name)
    
    if error:
        return jsonify({"error": error}), 400
    
    return jsonify({
        "predicted_price": float(predicted_price),
        "model_used": model_name
    })

@app.route('/api/feature_data')
def api_feature_data():
    """Returns feature data for visualization."""
    try:
        df = load_dataset(sample=True)
        
        # Identify price column
        price_columns = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
        for col in price_columns:
            if col in df.columns:
                target_col = col
                break
        else:
            target_col = df.columns[-1]
        
        # Clean price column
        df[target_col] = pd.to_numeric(df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        
        # Get data for Engine Size vs Price
        engine_cols = ['Engine Capacity', 'engine capacity', 'CC', 'Displacement', 'Engine Size', 'Engine Size (cc)']
        engine_col = next((col for col in engine_cols if col in df.columns), None)
        
        engine_data = []
        if engine_col:
            for _, row in df.iterrows():
                engine_data.append({
                    "x": float(row[engine_col]),
                    "y": float(row[target_col])
                })
        
        # Get data for Mileage vs Price
        mileage_cols = ['Mileage', 'mileage', 'Total Mileage', 'KM', 'Total Mileage (km)']
        mileage_col = next((col for col in mileage_cols if col in df.columns), None)
        
        mileage_data = []
        if mileage_col:
            for _, row in df.iterrows():
                mileage_data.append({
                    "x": float(row[mileage_col]),
                    "y": float(row[target_col])
                })
        
        # Get data for Brand vs Price
        brand_cols = ['Brand', 'brand', 'Bike Brand', 'Make', 'make', 'Manufacturer']
        brand_col = next((col for col in brand_cols if col in df.columns), None)
        
        brand_data = {}
        if brand_col:
            for brand, group in df.groupby(brand_col):
                brand_data[brand] = group[target_col].mean()
        
        # Get data for Registration Year vs Price
        year_cols = ['Registration Date', 'reg date', 'Year', 'Year of Registration']
        year_col = next((col for col in year_cols if col in df.columns), None)
        
        year_data = {}
        if year_col:
            for year, group in df.groupby(year_col):
                year_data[int(year)] = group[target_col].mean()
        
        return jsonify({
            "engine_data": engine_data,
            "mileage_data": mileage_data,
            "brand_data": brand_data,
            "year_data": {str(k): v for k, v in year_data.items()}
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualization/<path:filename>')
def visualization(filename):
    """Serve visualization images."""
    return send_from_directory(SVM_RESULTS_DIR, filename)

# ------------------------ USER DASHBOARD & PREDICTION ------------------------

@app.route('/user', methods=['GET', 'POST'])
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    prediction = None
    input_details = {}
    error = None

    if request.method == 'POST':
        print("🔍 Processing prediction form submission")
        # Collect form inputs
        brand = request.form.get('brand', '')
        license_class = request.form.get('license_class', '2B')
        category = request.form.get('category', '')
        reg_year_range = request.form.get('reg_year_range', '2021-2025')
        coe_left_range = request.form.get('coe_left_range', '5')
        mileage_range = request.form.get('mileage_range', '< 10,000km')
        previous_owners = request.form.get('previous_owners', '1')
        model = request.form.get('model', '')
        
        # Store all input details to display in the results
        input_details = {
            'brand': brand,
            'license_class': license_class,
            'category': category,
            'reg_year_range': reg_year_range,
            'coe_left_range': coe_left_range,
            'mileage_range': mileage_range,
            'previous_owners': previous_owners,
            'model': model
        }
        
        print(f"🔍 Form inputs: {input_details}")

        # Process inputs for prediction
        
        # Engine capacity based on license class
        engine_capacity = 150  # Default
        if license_class == "2B":
            engine_capacity = 150  # Median value for 2B class
        elif license_class == "2A":
            engine_capacity = 300  # Median value for 2A class
        elif license_class == "2":
            engine_capacity = 650  # Median value for Class 2
            
        # Registration year based on range (if reg_year_range is provided)
        reg_year = 2023  # Default to recent year
        if reg_year_range == "2021-2025":
            reg_year = 2023
        elif reg_year_range == "2018-2020":
            reg_year = 2019
        elif reg_year_range == "2015-2017":
            reg_year = 2016
        elif reg_year_range == "2010-2014":
            reg_year = 2012
        
        print(f"🔍 Registration year set to: {reg_year}")
            
        # COE expiry year based on slider value
        try:
            # Convert the slider value to float (handles decimal values like 5.5 years)
            coe_years_left = float(coe_left_range)
        except (ValueError, TypeError):
            # Default to 5 years if there's an error
            coe_years_left = 5.0
            print(f"⚠️ Error converting COE value: {coe_left_range}, using default: 5.0")

        # Current year as base  
        current_year = 2025
        # Calculate exact expiry date (including partial years)
        coe_expiry_year = current_year + coe_years_left

        # Print debug info
        print(f"🔍 COE Slider Value: {coe_left_range}, Converted Years: {coe_years_left}")
        print(f"🔍 Current Year: {current_year}, Calculated Expiry: {coe_expiry_year}")
            
        # Mileage based on range
        mileage = 10000  # Default
        if mileage_range == "< 10,000km":
            mileage = 5000
        elif mileage_range == "< 25,000km":
            mileage = 17500
        elif mileage_range == "< 50,000km":
            mileage = 37500
        elif mileage_range == "< 75,000km":
            mileage = 62500
        elif mileage_range == "> 100,000km":
            mileage = 125000
        
        print(f"🔍 Mileage set to: {mileage}")
            
        # Number of previous owners
        num_owners = 1  # Default
        if previous_owners == "1":
            num_owners = 1
        elif previous_owners == "2":
            num_owners = 2
        elif previous_owners == "3":
            num_owners = 3
        
        print(f"🔍 Number of owners set to: {num_owners}")
            
        # Category (convert to numeric if needed)
        category_value = 0  # Default
        if category:
            try:
                # If label encoders include category
                if 'Category' in label_encoders:
                    category_value = label_encoders['Category'].transform([category])[0]
                else:
                    # Simple mapping if label encoder not available
                    categories = ["Sport", "Naked", "Cruiser", "Touring", "Scooter", "Off-road", "Adventure", "Custom", "Other"]
                    category_value = categories.index(category) if category in categories else 0
            except:
                category_value = 0
        
        print(f"🔍 Category value set to: {category_value}")
                
        # Brand (convert to numeric if needed)
        brand_value = 0  # Default
        if brand:
            try:
                # If label encoders include brand
                if 'Brand' in label_encoders:
                    brand_value = label_encoders['Brand'].transform([brand])[0]
                else:
                    # Simple mapping if label encoder not available
                    brands = ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "Harley-Davidson", "KTM", "Triumph", "Other"]
                    brand_value = brands.index(brand) if brand in brands else 0
            except:
                brand_value = 0
        
        print(f"🔍 Brand value set to: {brand_value}")
                
        # Create input data dictionary for model prediction
        model_input = {
            'Engine Capacity': engine_capacity,
            'Registration Date': reg_year,
            'COE Expiry Date': coe_expiry_year,
            'Mileage': mileage,
            'No. of owners': num_owners,
            'Brand': brand_value,
            'Category': category_value
        }
        
        print(f"🔍 Model input prepared: {model_input}")
        
        # Make prediction
        predicted_price, error = predict_price(model_input, default_model)
        
        if error:
            flash(f"Error making prediction: {error}", "danger")
            print(f"❌ Error making prediction: {error}")
        else:
            prediction = predicted_price
            print(f"✅ Prediction successful: ${prediction:.2f}")

        # Debug prediction results
        print(f"🔍 Prediction: {prediction}")
        print(f"🔍 Input details: {input_details}")
        print(f"🔍 Model input: {model_input}")

    return render_template('user.html', 
                          filters=admin_selected_filters, 
                          prediction=prediction,
                          input_details=input_details)

# ------------------------ MAIN ------------------------

if __name__ == '__main__':
    # Calculate initial metrics for the default model
    try:
        calculate_model_metrics(default_model)
    except Exception as e:
        print(f"⚠️ Error calculating initial metrics: {e}")
    
    app.run(debug=True)
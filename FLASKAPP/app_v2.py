from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import sys
import importlib.util

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'motorbike_price_prediction'

# Path to models and preprocessing files
models_directory = "saved_models"
available_models = ["random_forest", "xgboost", "lightgbm", "svm"]

# Load models
models = {}
for model_name in available_models:
    model_path = os.path.join(models_directory, f"{model_name}_regressor.pkl")
    if os.path.exists(model_path):
        models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded {model_name.upper()} model.")

# Get the absolute path of the Algorithms directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS_DIR = os.path.join(BASE_DIR, 'Algorithms')

# Ensure Python can find scripts in the Algorithms directory
if ALGORITHMS_DIR not in sys.path:
    sys.path.append(ALGORITHMS_DIR)

# Load predict_motorbike_prices dynamically
predict_motorbike_prices_path = os.path.join(ALGORITHMS_DIR, "predict_motorbike_prices.py")

if os.path.exists(predict_motorbike_prices_path):
    spec = importlib.util.spec_from_file_location("predict_motorbike_prices", predict_motorbike_prices_path)
    predict_motorbike_prices = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_motorbike_prices)
    print("✅ Successfully loaded predict_motorbike_prices.py")
else:
    print("❌ Error: predict_motorbike_prices.py not found in Algorithms/")

# Load encoders and scaler
label_encoders = joblib.load(os.path.join(models_directory, "label_encoders.pkl"))
scaler = joblib.load(os.path.join(models_directory, "scaler.pkl"))

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

# ------------------------ USER DASHBOARD & PREDICTION ------------------------

@app.route('/user', methods=['GET', 'POST'])
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    prediction = None

    if request.method == 'POST':
        license_class = request.form.get('license_class')
        mileage_range = request.form.get('mileage_range')
        coe_left_range = request.form.get('coe_left_range')
        previous_owners = request.form.get('previous_owners')

        # Process user input for prediction
        sample_input = {
            'Engine Capacity': 155 if license_class == "2b" else (400 if license_class == "2a" else 600),
            'COE Expiry Date': 2030 - int(coe_left_range.split('-')[0]),  # Convert range to number
            'Total Mileage (km)': int(mileage_range.split('-')[0].replace(',', '')),
            'Number of Previous Owners': 1 if previous_owners == "1" else (2 if previous_owners == "2" else 3),
            'Brand': 0,  # Default value for now
            'Category': 1  # Default category for now
        }

        # Scale the input
        numeric_features = ['Engine Capacity', 'COE Expiry Date', 'Total Mileage (km)', 'Number of Previous Owners']
        sample_input_scaled = scaler.transform(np.array([sample_input[f] for f in numeric_features]).reshape(1, -1))

        # Predict using the best model
        selected_model = models["random_forest"]
        predicted_price = selected_model.predict(sample_input_scaled)[0]

        return render_template('user.html', filters=admin_selected_filters, prediction=predicted_price,
                               license_class=license_class, mileage_range=mileage_range,
                               coe_left_range=coe_left_range, previous_owners=previous_owners)

    return render_template('user.html', filters=admin_selected_filters)

# ------------------------ MODEL SELECTION ------------------------

@app.route('/update_model', methods=['POST'])
def update_model():
    """Updates the selected model for predictions."""
    selected_model = request.form.get('model')

    if selected_model in models:
        session['selected_model'] = selected_model
        flash(f"✅ Model updated to {selected_model.upper()}!", "success")
    else:
        flash("❌ Invalid model selection!", "danger")

    return redirect(url_for('admin_panel'))

@app.route('/model_status')
def model_status():
    """Returns available models and the currently selected model."""
    model_files = [f.replace("_regressor.pkl", "") for f in os.listdir(models_directory) if f.endswith("_regressor.pkl")]
    selected_model = session.get('selected_model', 'random_forest')

    return jsonify({
        "available_models": model_files,
        "selected_model": selected_model
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sys
import os
import subprocess
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from Algorithms.ml_models import TrainPredictor
    train_predictor = TrainPredictor()
    predictor_initialized = True
except Exception as e:
    print(f"Error initializing predictor: {e}")
    predictor_initialized = False

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'train_delay_prediction_key'

# Path to Naive Bayes Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ML_PROJECT directory
MODEL_PATH = os.path.join(BASE_DIR, "Algorithms", "naive_bayes.pkl")

# Load Naive Bayes Model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        naive_bayes_model = pickle.load(model_file)
    model_loaded = True
    print(f"Naive Bayes model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading Naive Bayes model: {e}")
    naive_bayes_model = None
    model_loaded = False

# Default model and graph selections
current_model = "Random Forest"
current_graph = "bar"

# Valid users (in production, this should be a database)
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

@app.route('/')
def home():
    if not predictor_initialized:
        flash('Error: Prediction model could not be loaded. Please check dependencies.', 'danger')
        
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

@app.route('/admin')
def admin_panel():
    if 'user_id' in session and session.get('role') == 'admin':
        return render_template('admin.html', available_models=train_predictor.available_models if predictor_initialized else [])
    return redirect(url_for('login'))

@app.route('/user')
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('user.html')

@app.route('/update_model', methods=['POST'])
def update_model():
    global current_model
    if 'user_id' in session and session.get('role') == 'admin':
        new_model = request.form.get('model')
        if new_model in train_predictor.available_models:
            current_model = new_model
            train_predictor.load_selected_model()
            flash(f'Model changed to {current_model}', 'success')
        else:
            flash(f'Invalid model selection.', 'danger')
    return redirect(url_for('admin_panel'))

@app.route('/update_graph', methods=['POST'])
def update_graph():
    global current_graph
    if 'user_id' in session and session.get('role') == 'admin':
        new_graph = request.form.get('graph')
        if new_graph in ["bar", "line", "scatter"]:
            current_graph = new_graph
            flash(f'Graph changed to {current_graph}', 'success')
        else:
            flash('Invalid graph selection', 'danger')
    return redirect(url_for('admin_panel'))

@app.route('/get_selected_model')
def get_selected_model():
    return jsonify({"selected": current_model, "available": train_predictor.available_models if predictor_initialized else []})

@app.route('/get_selected_graph')
def get_selected_graph():
    return jsonify({"selected": current_graph})

@app.route('/model_predict', methods=['POST'])
def model_predict():
    if not predictor_initialized:
        return jsonify({"error": "Prediction service unavailable"}), 503

    alert = request.form.get('alert', '')
    try:
        feature1 = float(request.form.get('feature1', 0))
        feature2 = float(request.form.get('feature2', 0))
    except ValueError:
        return jsonify({"error": "Invalid input"}), 400

    try:
        delay, severity = train_predictor.predict(alert, feature1, feature2)
        return jsonify({"delay": float(delay), "severity": severity, "model_used": current_model})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/naivebayes_predict', methods=['POST'])
def naivebayes_predict():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 403

    start_station = request.form.get('start_station')
    end_station = request.form.get('end_station')
    travel_day = request.form.get('travel_day')

    if not all([start_station, end_station, travel_day]):
        return jsonify({"error": "All fields are required."}), 400

    if not model_loaded:
        return jsonify({"error": "Model is not available. Please check the server logs."}), 500

    input_text = f"{start_station} {end_station} {travel_day}"

    try:
        # Predict probability
        prediction_prob = naive_bayes_model.predict_proba([input_text])[0][1] * 100

        # Calculate model accuracy
        dataset_path = os.path.join(BASE_DIR, "NaiveBayes", "Dataset_Latest.xlsx")
        df = pd.read_excel(dataset_path, engine='openpyxl')

        stations_col = df.columns[8]
        day_col = df.columns[9]

        df['stations'] = df[stations_col].apply(lambda x: ' '.join(str(x).split('\n')))
        df['day'] = df[day_col]
        df['features'] = df['stations'] + ' ' + df['day']

        _, X_test, _, y_test = train_test_split(df['features'], df['day'], test_size=0.2, random_state=42)
        y_pred = naive_bayes_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "prediction": f"{prediction_prob:.2f}",
        "accuracy": f"{accuracy:.2f}",
        "travel_day": travel_day
    })


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

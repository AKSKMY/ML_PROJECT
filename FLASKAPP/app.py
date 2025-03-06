from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from Algorithms.ml_models import TrainPredictor
    # Initialize the predictor
    train_predictor = TrainPredictor()
    predictor_initialized = True
except Exception as e:
    print(f"Error initializing predictor: {e}")
    predictor_initialized = False

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'train_delay_prediction_key'

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
        if session.get('role') == 'admin':
            return redirect(url_for('admin_panel'))
        else:
            return redirect(url_for('user_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if username in users and users[username]['password'] == password:
            session['user_id'] = username
            session['role'] = users[username]['role']
            
            # Redirect based on role
            if session['role'] == 'admin':
                return redirect(url_for('admin_panel'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('login.html')

@app.route('/admin')
def admin_panel():
    if 'user_id' in session and session.get('role') == 'admin':
        if not predictor_initialized:
            flash('Warning: Model predictor not initialized. Prediction functionality will be limited.', 'warning')
        return render_template('admin.html', 
                              available_models=train_predictor.available_models if predictor_initialized else [])
    return redirect(url_for('login'))

@app.route('/user')
def user_dashboard():
    if 'user_id' in session:
        if not predictor_initialized:
            flash('Warning: Model predictor not initialized. Prediction functionality will be limited.', 'warning')
        return render_template('user.html')
    return redirect(url_for('login'))

@app.route('/update_model', methods=['POST'])
def update_model():
    global current_model
    if 'user_id' in session and session.get('role') == 'admin':
        if not predictor_initialized:
            flash('Error: Cannot change model, predictor not initialized.', 'danger')
            return redirect(url_for('admin_panel'))
            
        new_model = request.form.get('model')
        
        # Validate model selection against available models
        if new_model in train_predictor.available_models:
            current_model = new_model
            
            # Get absolute path to Algorithms directory
            algorithms_dir = os.path.join(parent_dir, "Algorithms")
            
            # Save the selected model to a file for the ML predictor
            with open(os.path.join(algorithms_dir, "selected_model.txt"), "w") as file:
                file.write(current_model)
            
            # Reload the model in the predictor
            train_predictor.load_selected_model()
            
            flash(f'Model changed to {current_model}', 'success')
        else:
            flash(f'Invalid model selection. Available models: {", ".join(train_predictor.available_models)}', 'danger')
    
    return redirect(url_for('admin_panel'))

@app.route('/update_graph', methods=['POST'])
def update_graph():
    global current_graph
    if 'user_id' in session and session.get('role') == 'admin':
        new_graph = request.form.get('graph')
        
        # Validate graph selection
        valid_graphs = ["bar", "line", "scatter"]
        if new_graph in valid_graphs:
            current_graph = new_graph
            flash(f'Graph changed to {current_graph}', 'success')
        else:
            flash('Invalid graph selection', 'danger')
    
    return redirect(url_for('admin_panel'))

@app.route('/get_selected_model')
def get_selected_model():
    if predictor_initialized:
        return jsonify({"selected": current_model, 
                        "available": train_predictor.available_models})
    return jsonify({"selected": "None", "available": []})

@app.route('/get_selected_graph')
def get_selected_graph():
    return jsonify({"selected": current_graph})

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor_initialized:
        return jsonify({
            "error": "Prediction service not available. Please check model dependencies."
        }), 503
    
    # Get prediction parameters from form
    alert = request.form.get('alert', '')
    try:
        feature1 = float(request.form.get('feature1', 0))
        feature2 = float(request.form.get('feature2', 0))
    except ValueError:
        feature1, feature2 = 0, 0
    
    try:
        # Make prediction using the loaded model
        delay, severity = train_predictor.predict(alert, feature1, feature2)
        
        # Return prediction result
        return jsonify({
            "delay": float(delay),
            "severity": severity,
            "model_used": current_model
        })
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/model_status')
def model_status():
    """Return status of all available models"""
    if predictor_initialized:
        return jsonify({
            "status": "ok",
            "available_models": train_predictor.available_models,
            "selected_model": train_predictor.selected_model
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Model predictor not initialized"
        }), 503

if __name__ == '__main__':
    app.run(debug=True)
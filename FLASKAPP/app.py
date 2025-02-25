from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import os
import sys
import joblib
from werkzeug.utils import secure_filename

# Add the parent directory (ML_PROJECT) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.ml_models import TrainPredictor  # Import ML model class

app = Flask(__name__, template_folder="../templates")
app.secret_key = 'INF2008Project'

# File paths for storing model and graph selections
MODEL_FILE = "../Algorithms/selected_model.txt"
GRAPH_FILE = "../Algorithms/selected_graph.txt"

# Set default values if the files don't exist
if not os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "w") as f:
        f.write("Random Forest")

if not os.path.exists(GRAPH_FILE):
    with open(GRAPH_FILE, "w") as f:
        f.write("bar")

# Load the trained model
model = TrainPredictor()

# User credentials
admin_username = 'admin'
admin_password = 'admin123'
user_username = 'user'
user_password = 'user123'

# Login route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if role == 'admin' and username == admin_username and password == admin_password:
            session['username'] = username
            session['role'] = role  # Set role as admin
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        elif role == 'user' and username == user_username and password == user_password:
            session['username'] = username
            session['role'] = role  # Set role as user
            flash('Login successful!', 'success')
            return redirect(url_for('user'))

        else:
            flash('Invalid credentials, please try again.', 'danger')

    return render_template('login.html')

# Logout route
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Route for administrator
@app.route('/admin', methods=["GET", "POST"])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('admin.html')

# User page with prediction functionality
@app.route('/user', methods=["GET", "POST"])
def user():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    prediction = None
    severity = None

    if request.method == "POST":
        train_alert = request.form["alert_text"]
        feature1 = float(request.form["feature1"])
        feature2 = float(request.form["feature2"])

        predicted_delay, severity_label = model.predict(train_alert, feature1, feature2)

        prediction = f"Predicted Delay: {predicted_delay:.2f} minutes"
        severity = f"Severity: {severity_label}"

    return render_template("user.html", prediction=prediction, severity=severity)

# Route to update selected graph type (Admin Only)
@app.route('/update_graph', methods=['POST'])
def update_graph():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Unauthorized action.", "danger")
        return redirect(url_for('login'))

    selected_graph = request.form.get("graph", "bar")

    # Save graph type to file
    with open(GRAPH_FILE, "w") as f:
        f.write(selected_graph)

    flash(f"Visualization updated to {selected_graph}!", "success")
    return redirect(url_for('index'))

# Route to get selected graph type (User View)
@app.route('/get_selected_graph', methods=['GET'])
def get_selected_graph():
    with open(GRAPH_FILE, "r") as f:
        selected_graph = f.read().strip()
    return jsonify({"selected": selected_graph})

# Route to update selected model (Admin Only)
@app.route('/update_model', methods=['POST'])
def update_model():
    if 'username' not in session or session.get('role') != 'admin':
        flash("Unauthorized action.", "danger")
        return redirect(url_for('login'))
    
    selected_model = request.form.get("model", "Random Forest")

    # Save selected model to file
    with open(MODEL_FILE, "w") as f:
        f.write(selected_model)

    flash(f"Model updated to {selected_model}!", "success")
    return redirect(url_for('index'))

# Route to get the current selected model
@app.route('/get_selected_model', methods=['GET'])
def get_selected_model():
    with open(MODEL_FILE, "r") as f:
        selected_model = f.read().strip()
    return jsonify({"selected": selected_model})

if __name__ == "__main__":
    app.run(debug=True)

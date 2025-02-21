from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import os
import sys
from werkzeug.utils import secure_filename
# Add the parent directory (ML_PROJECT) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithms.ml_models import TrainPredictor  # Import ML model class


app = Flask(__name__, template_folder="../templates")

app.secret_key = 'INF2008Project'

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "Latest_Dataset.xlsx"))

# Initialize and train the ML models
model = TrainPredictor()
model.train_model(dataset_path)  # Train on startup

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

        # Check if admin credentials are valid
        if role == 'admin' and username == admin_username and password == admin_password:
            session['username'] = username
            session['role'] = role  # Set role as admin
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        # Check if user credentials are valid
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
    global current_csv_path
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == "POST":
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            current_csv_path = file_path
    
    # # Read the current CSV file
    # df = pd.read_csv(current_csv_path)
    
    # # Separate DataFrame based on unique values in 'Z' column
    # separated_dfs = separate_df(df, 'Z')

    # # Create plots for each group
    # plots = {}
    # plot_titles = {2.0: 'Level 1', 7.0: 'Level 2', 12.0: 'Level 3'}  # Adjust according to actual Z values
    # plot_colors = {2.0: 'red', 7.0: 'green', 12.0: 'blue'}  # Define colors for each level
    # for key, value in separated_dfs.items():
    #     title = plot_titles.get(key, f'Group {key}')  # Default title if key is not in plot_titles
    #     color = plot_colors.get(key, 'black')
    #     fig = px.scatter(value, x='X', y='Y', text='Item', title=title, color_discrete_sequence=[color])
    #     plots[key] = pio.to_html(fig, full_html=False)
    
    # items = df['Item'].tolist()
    
    # return render_template('admin.html', plots=plots, plot_titles=plot_titles, items=items)

    return render_template('admin.html')

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

# Default visualization type
VISUALIZATION_TYPE = "bar"

# Route to update visualization (Admin Only)
@app.route('/update_visualization', methods=['POST'])
def update_visualization():
    if 'username' not in session or session.get('role') != 'admin':  # Check both username and role
        flash("Unauthorized action.", "danger")
        return redirect(url_for('login'))  # Redirect to login if not authorized
    
    global VISUALIZATION_TYPE
    VISUALIZATION_TYPE = request.form.get("visualization", "bar")
    flash("Visualization updated successfully!", "success")
    return redirect(url_for('index'))

# Route to get current visualization (User View)
@app.route('/get_visualization', methods=['GET'])
def get_visualization():
    return jsonify({"selected": VISUALIZATION_TYPE})


if __name__ == "__main__":
    app.run(debug=True)
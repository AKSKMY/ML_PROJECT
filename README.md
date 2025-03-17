# ML_PROJECT GROUP 14
# USER MANUAL

# User Manual for Flask Machine Learning Web Application (Price Prediction Function) Excluding other individual data visualisation

This manual explains how to set up, use, and manage your Flask-based web application, including detailed steps for training and testing machine learning models.

---

## 1. Environment Setup

### Prerequisites
- Python (Recommended version: Python 3.10)
- Flask
- Required Python libraries:
  pip install -r requirements.txt


## 2. Running the Flask Application

### Starting the Server
- Navigate to the application directory containing `app_v2.py`.
- Execute:
  ```bash
  python app_v2.py
  ```
- The application will start at `http://localhost:5000`.

## 3. Web Interface Usage

### Login Page
- Navigate to `http://localhost:5000/login`
- Enter the username and password using demo accounts
- User roles:
  - **Admin**: View machine learning algorithms accuracy and overview.
  - **User**: Accesses prediction functionalities only.

### Admin Page
- Accessible via `http://localhost:5000/admin`
- Capabilities:
  - Select Active Model for /user.html motorcycle prediction model.
  - Manage active user filters.
  - View Machine Learning model status.

### User Dashboard
- Accessible after successful login.
- Users can enter motorbike attributes to get price predictions.

## 3. Training the Machine Learning Models

### Step-by-Step Training

#### Data Preparation/Cleaning
- Ensure your dataset is in a CSV format.
- Place the CSV file in the working directory.
- Clean dataset such as in our case removing SOLD entries.

#### Execute Training Script
- Run the provided script:
```bash
python train_models.py
```
- The script:
  - Performs data preprocessing.
  - Splits data into training and testing sets.
  - Trains multiple machine learning models: Linear Regression, Decision Tree, K-Nearest Neighbors, Support Vector Machine, and Random Forest.
  - Evaluates each model using accuracy scores and selects the best-performing model.
  - Saves trained models using `joblib`.

#### Model Evaluation
- After training, the script provides a performance summary including accuracy and other metrics.

### Saved Model
- Models are saved in the `.pkl` format, ready to be loaded for predictions.

## 4. Making Predictions

### Prediction Script
- The `predict_motorbike_prices.py` script used by `train_model.py` predict motorbike prices based on input parameters.
```bash
python predict_motorbike_prices.py
```
- This script loads the saved model and makes predictions based on user inputs.

## 4. Understanding Performance Metrics
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Effectiveness of identifying positive instances.
- **F1-score**: Balanced metric of Precision and Recall.

## 5. Troubleshooting

### Common Errors and Solutions
- **Issue**: Incorrect or nonsensical predictions.
  - Verify correct preprocessing (feature scaling, normalization).
- **Solution**: 
1) Check normalization steps in training script, ensure consistency between training and prediction phases.
2) Drop columns that shouldn't be included in training.

## 5. Predicting Motorbike Prices

- Run prediction script:
```bash
python predict_motorbike_prices.py
```
- Follow on-screen prompts to input data and receive predictions.

## 5. Troubleshooting
- **Issue**: Flask server not starting.
  - **Solution**: Ensure Flask is installed (`pip install flask`). Check port availability.
- **Issue**: Incorrect predictions or model errors.
  - **Solution**: Re-check data preprocessing steps and model training consistency. Or change features used in training the models.

## 6. Testing our other models
- 

---

This manual provides clear instructions on how to set up your Flask application, manage users, train machine learning models, evaluate their performance, and predict motorbike prices.
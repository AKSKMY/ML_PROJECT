# Motorcycle Price Prediction Web Application

A Flask-based machine learning web application for predicting motorcycle prices in the Singapore market. This project integrates several ML algorithms—including Random Forest, XGBoost, LightGBM, SVM, and (optionally) CatBoost—to provide accurate price predictions based on motorbike attributes. The application features a user-friendly web interface with distinct roles for regular users and administrators, dynamic performance visualizations, and a robust training pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Running the Application](#running-the-application)
5. [Web Interface Usage](#web-interface-usage)
   - [Login Page](#login-page)
   - [User Dashboard](#user-dashboard)
   - [Admin Panel](#admin-panel)
6. [Training the Machine Learning Models](#training-the-machine-learning-models)
   - [General Training (`train_models.py`)](#general-training-train_modelspy)
   - [CatBoost Model Training (`CatBoosting.py`)](#catboost-model-training-catboostingpy)
   - [SVM Model Training (`svm_motorbike_trainer.py`)](#svm-model-training-svm_motorbike_trainerpy)
7. [Making Predictions](#making-predictions)
8. [Performance Metrics & Visualizations](#performance-metrics--visualizations)
9. [Troubleshooting & FAQs](#troubleshooting--faqs)
10. [Project Structure](#project-structure)
11. [License](#license)

---

## Overview

This project is designed to predict the price of a used motorbike by leveraging a variety of machine learning models. It preprocesses real-world data, trains several models, and dynamically selects the best-performing one for use. Users can access the application via a web interface while administrators can monitor model performance, change the active model, and adjust input filters.

---

## Prerequisites & Environment Setup

- **Python:** This project requires **Python 3.11.x**. (Note that newer Python versions may lead to compatibility issues with some dependencies.)
- **Web Framework:** Flask
- **Other Libraries:** NumPy, Pandas, Scikit-learn, XGBoost, LightGBM, (optionally) CatBoost, Matplotlib, Seaborn, Joblib, and more (see [Installing Dependencies](#installing-dependencies)).

Ensure you have the correct Python version installed before proceeding.

---

## Installing Dependencies

A dedicated script, `dependency_installer.py`, is provided to help you install all required dependencies.

1. **Run the installer:**
   ```bash
   python dependency_installer.py
   ```
2. **Notes:**
   - The script checks for essential packages such as scikit-learn, pandas, matplotlib, seaborn, xgboost, lightgbm, and others.
   - CatBoost installation is handled separately since it requires Rust to be installed (or a pre-built binary version).
   - Follow the prompts if your Python version is not 3.11.x; it is highly recommended to use Python 3.11.x for full compatibility.

---

## Running the Application

To launch the web application:

1. **Start the Flask server:**
   - Navigate to the directory containing `app_v2.py`.
   - Run:
     ```bash
     python app_v2.py
     ```
2. The server will start on `http://localhost:5000`.

---

## Web Interface Usage

### Login Page
- **URL:** `http://localhost:5000/login`
- **Demo Credentials:**
  - **Admin:** Username: `admin` | Password: `admin123`
  - **User:** Username: `user` | Password: `user123`
- Choose your role using the provided buttons before logging in.

### User Dashboard
- After logging in as a user, you’ll see a form to enter motorbike attributes (e.g., engine capacity, registration year, mileage, COE years left, etc.).
- Upon submission, the application displays the predicted price along with a brief summary of the inputs used.

### Admin Panel
- **URL:** `http://localhost:5000/admin`
- **Features:**
  - **Model Control:** View a list of loaded models (Random Forest, XGBoost, LightGBM, SVM, and optionally CatBoost) and select the active model for predictions.
  - **Performance Metrics:** Monitor key metrics such as MAE, RMSE, R² score, and prediction accuracy.
  - **User Filter Controls:** Enable or disable input filters (e.g., license class, mileage range, COE left, previous owners) for the user interface.
  - **Visualizations:** Access dynamic plots that compare model performance and visualize prediction errors.

---

## Training the Machine Learning Models

The project includes multiple training scripts to build and save different ML models.

### General Training (`train_models.py`)
- **Purpose:** Preprocesses data from `combined_dataset_latest.xlsx`, trains models (Random Forest, XGBoost, LightGBM, and SVM), evaluates performance, and saves the trained models using Joblib.
- **Usage:**
  ```bash
  python train_models.py
  ```

### CatBoost Model Training (`CatBoosting.py`)
- **Purpose:** Specifically handles data cleaning, feature engineering (including log-transformations and polynomial features), hyperparameter tuning via RandomizedSearchCV, and training of a CatBoostRegressor model.
- **Usage:**
  ```bash
  python CatBoosting.py
  ```
- **Note:** CatBoost is optional and will only work if its dependencies (including Rust or a pre-built binary) are met.

### SVM Model Training (`svm_motorbike_trainer.py`)
- **Purpose:** Trains an SVM model using a robust pipeline that includes outlier removal, hyperparameter tuning via RandomizedSearchCV, and optional log-transformation of target values. The script also saves the entire model package (including feature names, scaler, and label encoders) for use during prediction.
- **Usage:**
  ```bash
  python svm_motorbike_trainer.py
  ```

---

## Making Predictions

Predictions can be obtained either by:

- **Using the Web Interface:** Users fill in the form on the dashboard, and the application (via `app_v2.py`) loads the selected model to make a prediction.
- **Running a Prediction Script:** (If available) You can run a dedicated prediction script (e.g., `predict_motorbike_prices.py`) that loads a saved model and predicts prices based on command-line input.

The prediction process ensures that input data is preprocessed consistently with the training phase, including scaling, encoding, and feature engineering.

---

## Performance Metrics & Visualizations

- **Admin Panel:** Displays a summary of performance metrics (MAE, RMSE, R², accuracy) for the active model.
- **Model Comparison:** A table in the admin panel shows metrics for all available models.
- **Dynamic Visualizations:** The application generates charts (scatter plots of actual vs. predicted prices, error distributions, residual plots, and feature importance graphs) to help evaluate model performance.
- **Note:** Visualizations are generated and stored in a dedicated results directory and can be refreshed via the admin interface.

---

## Troubleshooting & FAQs

- **Incorrect Predictions:**
  - Verify that data preprocessing (scaling, encoding, feature engineering) is consistent between training and prediction.
  - Check for any missing or improperly formatted columns in your dataset.
- **Flask Server Issues:**
  - Ensure that Flask and all required dependencies are installed.
  - Verify that the port (default 5000) is available.
- **CatBoost Installation:**
  - If CatBoost fails to install, ensure that Rust is installed or try installing the pre-built binary using the options provided by `dependency_installer.py`.
- **Python Version Mismatch:**
  - This project requires Python 3.11.x. If you are using a different version, you may experience compatibility issues.

---

## Project Structure

Below is an overview of the main files and directories in the project:

- **app_v2.py:** Main Flask application file that sets up routes, loads models, and handles predictions.
- **train_models.py:** Script for training multiple models (Random Forest, XGBoost, LightGBM, SVM).
- **CatBoosting.py:** Script for training a CatBoost model with advanced feature engineering.
- **svm_motorbike_trainer.py:** SVM-specific training script that packages preprocessing objects along with the model.
- **login.html, user.html, admin.html:** HTML templates for the login page, user dashboard, and admin panel, respectively.
- **dependency_installer.py:** Script to automatically install required dependencies.
- **saved_models/** (directory): Contains trained model files (in `.pkl` format), scalers, label encoders, and other preprocessing objects.
- **results/** (directory): Stores visualizations and performance plots generated during model evaluation.
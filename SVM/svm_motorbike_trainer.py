"""
SVM Model Trainer for Motorcycle Price Prediction
This script focuses on training and optimizing an SVM model specifically
for motorcycle price prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time

class SVMMotorbikeTrainer:
    def __init__(self, visualize=True):
        """
        Initialize the SVM Motorbike Price Predictor

        Parameters:
        -----------
        visualize : bool
            Whether to create visualization plots
        """
        # Define paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = self.find_dataset()
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Results directory for visualizations
        if visualize:
            self.results_dir = os.path.join(self.base_dir, "SVM", "results")
            os.makedirs(self.results_dir, exist_ok=True)
        
        # SVM Model parameters
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.visualize = visualize
        
        print(f"🔍 Dataset path: {self.dataset_path}")
        print(f"💾 Models directory: {self.models_dir}")
        if visualize:
            print(f"📊 Results directory: {self.results_dir}")

    def find_dataset(self):
        """Find the latest dataset file"""
        dataset_names = ["combined_dataset_latest.xlsx", "Latest_Dataset.xlsx", "bike_data.xlsx"]
        
        # First, check the immediate directories
        search_dirs = [".", "Datasets", "../Datasets", "../../Datasets", "NewStuff", "../NewStuff"]
        
        # Get the absolute path to the project root (parent of SVM directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Print debugging information
        print(f"🔍 Looking for datasets in directories relative to: {self.base_dir}")
        print(f"🔍 Project root directory: {project_root}")
        
        # Check all combinations of search directories and dataset names
        for search_dir in search_dirs:
            search_path = os.path.join(self.base_dir, search_dir)
            print(f"🔍 Checking directory: {search_path}")
            
            for dataset_name in dataset_names:
                potential_path = os.path.join(search_path, dataset_name)
                if os.path.exists(potential_path):
                    print(f"✅ Found dataset at: {potential_path}")
                    return potential_path
        
        # Explicit check for known project paths
        explicit_paths = [
            os.path.join(project_root, "Datasets", "bike_data.xlsx"),
            os.path.join(project_root, "Datasets", "combined_dataset_latest.xlsx"),
            os.path.join(project_root, "Datasets", "Latest_Dataset.xlsx"),
            os.path.join(project_root, "NewStuff", "bike_data.xlsx"),
            os.path.join(project_root, "NewStuff", "combined_dataset_latest.xlsx"),
            os.path.join(project_root, "KNN & NN", "Latest_Dataset.xlsx"),
            os.path.join(project_root, "LogisticRegression", "Latest_Dataset.xlsx")
        ]
        
        for path in explicit_paths:
            print(f"🔍 Checking explicit path: {path}")
            if os.path.exists(path):
                print(f"✅ Found dataset at: {path}")
                return path
        
        # If still not found, try a more exhaustive search
        print("🔍 Performing deeper search for dataset files...")
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file in dataset_names:
                    path = os.path.join(root, file)
                    print(f"✅ Found dataset at: {path}")
                    return path
                # Also check for Excel files that might be relevant
                elif file.endswith('.xlsx') and any(keyword in file.lower() for keyword in ['bike', 'motor', 'vehicle']):
                    path = os.path.join(root, file)
                    print(f"🔍 Found potentially relevant dataset at: {path}")
                    return path
        
        # If we get here, we couldn't find the dataset
        print("❌ No dataset files found after exhaustive search.")
        
        # Create a synthetic dataset as a last resort
        print("⚠️ Creating a synthetic dataset for demonstration purposes...")
        synthetic_path = os.path.join(self.base_dir, "synthetic_bike_data.xlsx")
        
        import numpy as np
        import pandas as pd
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            "Brand": np.random.choice(["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati"], n_samples),
            "Model": np.random.choice(["CBR", "R1", "Ninja", "GSX-R", "Panigale"], n_samples),
            "Engine Capacity": np.random.randint(125, 1200, n_samples),
            "Registration Date": np.random.randint(2010, 2024, n_samples),
            "COE Expiry Date": np.random.randint(2024, 2034, n_samples),
            "Mileage": np.random.randint(1000, 50000, n_samples),
            "No. of owners": np.random.randint(1, 4, n_samples),
            "Category": np.random.choice(["Sport", "Naked", "Cruiser", "Touring", "Scooter"], n_samples),
            "Price": np.random.randint(5000, 25000, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_excel(synthetic_path, index=False)
        
        print(f"✅ Created synthetic dataset at: {synthetic_path}")
        return synthetic_path

    def load_data(self):
        """Load and preprocess the motorcycle dataset"""
        print("🔄 Loading and preprocessing data...")
        
        try:
            # Try different Excel engines
            for engine in ['openpyxl', 'xlrd']:
                try:
                    df = pd.read_excel(self.dataset_path, engine=engine)
                    break
                except Exception as e:
                    print(f"⚠️ Engine {engine} failed: {e}")
            else:
                raise ValueError("Could not read Excel file with any engine")
            
            print(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        # Print actual column names for debugging
        print("📋 Actual columns in dataset:", df.columns.tolist())
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Try to identify price column (different datasets might use different names)
        price_columns = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
        for col in price_columns:
            if col in df.columns:
                target_col = col
                break
        else:
            # If no clear price column, use the last column and hope it's price
            target_col = df.columns[-1]
            print(f"⚠️ No clear price column found, using {target_col} as target")
        
        # Clean price column (remove currency symbols and commas)
        df[target_col] = df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Handle missing values in target
        if df[target_col].isna().sum() > 0:
            print(f"⚠️ Found {df[target_col].isna().sum()} missing values in target column")
            df = df.dropna(subset=[target_col])
            print(f"✅ Dropped rows with missing target values, {df.shape[0]} rows remaining")
        
        # Try to identify important feature columns
        
        # For brand
        brand_cols = ['Brand', 'brand', 'Bike Brand', 'Make', 'make', 'Manufacturer']
        brand_col = next((col for col in brand_cols if col in df.columns), None)
        
        # For model
        model_cols = ['Model', 'model', 'Variant', 'variant', 'Model Year', 'Variant/Model Year']
        model_col = next((col for col in model_cols if col in df.columns), None)
        
        # For engine capacity
        engine_cols = ['Engine Capacity', 'engine capacity', 'CC', 'Displacement', 'Engine Size', 'Engine Size (cc)']
        engine_col = next((col for col in engine_cols if col in df.columns), None)
        
        # For registration date/year
        reg_cols = ['Registration Date', 'reg date', 'Year', 'Year of Registration']
        reg_col = next((col for col in reg_cols if col in df.columns), None)
        
        # For COE expiry
        coe_cols = ['COE Expiry Date', 'COE expiry', 'COE Expiry Year']
        coe_col = next((col for col in coe_cols if col in df.columns), None)
        
        # For mileage
        mileage_cols = ['Mileage', 'mileage', 'Total Mileage', 'KM', 'Total Mileage (km)']
        mileage_col = next((col for col in mileage_cols if col in df.columns), None)
        
        # For owners
        owners_cols = ['No. of owners', 'Owners', 'Previous Owners', 'Number of Previous Owners']
        owners_col = next((col for col in owners_cols if col in df.columns), None)
        
        # For category/classification
        category_cols = ['Category', 'category', 'Type', 'Classification', 'Market Segment']
        category_col = next((col for col in category_cols if col in df.columns), None)
        
        # Collect all identified columns
        feature_cols = [col for col in [brand_col, model_col, engine_col, reg_col, coe_col, mileage_col, owners_col, category_col] 
                      if col is not None]
        
        print(f"✅ Identified feature columns: {feature_cols}")
        print(f"✅ Target column: {target_col}")
        
        # Prepare dataframe for modeling
        df_clean = df[feature_cols + [target_col]].copy()
        
        # Clean and prepare each column
        for col in df_clean.columns:
            # Skip target column
            if col == target_col:
                continue
                
            # Clean numeric columns
            if col in [engine_col, mileage_col, owners_col]:
                # Extract numeric values
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Fill missing values with median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Handle date columns
            if col in [reg_col, coe_col]:
                try:
                    # Try to convert to datetime first
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce').dt.year
                    # Fill missing values with median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                except Exception as e:
                    print(f"⚠️ Error processing date column {col}: {e}")
                    # If conversion fails, try to extract year with regex
                    df_clean[col] = df_clean[col].astype(str).str.extract(r'(\d{4})').astype(float)
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # Encode categorical columns
            if col in [brand_col, model_col, category_col]:
                df_clean[col] = df_clean[col].astype(str)
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
        
        print("✅ Data cleaning complete")
        
        # Final check for any remaining NaN values
        if df_clean.isna().sum().sum() > 0:
            print("⚠️ There are still NaN values in the cleaned dataframe. Filling with appropriate values...")
            # For numeric columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            cat_cols = df_clean.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Split features and target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Data split and scaled: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def train_model(self, X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True):
        """Train an SVM model for motorcycle price prediction"""
        print("\n🔄 Training SVM model for motorcycle price prediction...")
        start_time = time.time()
        
        if tune_hyperparams:
            print("📊 Performing hyperparameter tuning for SVM...")
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['linear', 'rbf', 'poly'],
                'epsilon': [0.01, 0.1, 0.2]
            }
            
            # Use smaller param grid for faster execution if needed
            # param_grid = {
            #     'C': [10, 100],
            #     'gamma': ['scale', 0.1],
            #     'kernel': ['rbf']
            # }
            
            # Use GridSearchCV to find the best parameters
            grid_search = GridSearchCV(
                SVR(),
                param_grid,
                cv=5,
                scoring='neg_mean_absolute_error',
                verbose=1,
                n_jobs=-1  # Use all available cores
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            self.model = grid_search.best_estimator_
            print(f"✅ Best parameters found: {grid_search.best_params_}")
        else:
            print("📊 Training SVM with default parameters...")
            # Use default parameters
            self.model = SVR(C=100, gamma='scale', kernel='rbf')
            self.model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print("\n📈 Model Evaluation Results:")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save model and preprocessing objects
        print("\n💾 Saving model and preprocessing objects...")
        joblib.dump(self.model, os.path.join(self.models_dir, "svm_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(self.models_dir, "label_encoders.pkl"))
        
        # Create a "selected_model.txt" file with "SVM" to set it as the default model
        with open(os.path.join(self.base_dir, "selected_model.txt"), "w") as f:
            f.write("SVM")
        
        print("✅ Model and preprocessing objects saved successfully")
        
        # Visualize results if enabled
        if self.visualize:
            self.visualize_results(y_test, y_pred, feature_names)
        
        return {
            'model': self.model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time
        }
    
    def visualize_results(self, y_test, y_pred, feature_names):
        """Create visualizations of model performance"""
        print("\n📊 Creating visualizations...")
        
        # 1. Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('SVM Model: Actual vs Predicted Motorcycle Prices')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_actual_vs_predicted.png'))
        
        # 2. Error Distribution
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('SVM Model: Error Distribution')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_error_distribution.png'))
        
        # 3. Feature importance (SVR doesn't provide feature importance directly)
        # We can approximate importance by evaluating model with permutated features
        
        # 4. Residual Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=errors)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residual')
        plt.title('SVM Model: Residual Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_residual_plot.png'))
        
        # 5. Create a model summary report
        performance_summary = f"""
# SVM Model for Motorcycle Price Prediction

## Model Overview
- **Algorithm**: Support Vector Machine (SVR)
- **Implementation**: scikit-learn
- **Kernel**: {self.model.kernel}
- **C (Regularization)**: {self.model.C}
- **Gamma**: {self.model.gamma}
- **Epsilon**: {self.model.epsilon}

## Performance Metrics
- **Mean Absolute Error (MAE)**: ${mean_absolute_error(y_test, y_pred):.2f}
- **Root Mean Squared Error (RMSE)**: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}
- **R² Score**: {r2_score(y_test, y_pred):.4f}

## Feature Summary
The model was trained using the following features:
{", ".join(feature_names)}

## Visualization Summary
- **Actual vs Predicted**: Shows the relationship between actual and predicted motorcycle prices
- **Error Distribution**: Shows the distribution of prediction errors
- **Residual Plot**: Shows the relationship between predictions and their errors

## Model Advantages
- SVM can capture non-linear relationships in the data
- Robust to outliers when properly tuned
- Works well with both categorical (encoded) and numerical features

## Recommendations
- Consider feature engineering to capture more complex relationships
- Experiment with different kernel functions for potentially better performance
- Regularly update the model as new data becomes available
"""
        
        with open(os.path.join(self.results_dir, 'svm_performance_summary.md'), 'w') as f:
            f.write(performance_summary)
        
        print("✅ Visualizations created successfully")
    
    def compare_with_other_models(self):
        """Compare SVM with other models in the system"""
        print("\n📊 Comparing SVM with other models...")
        
        # Load metrics for other available models
        other_models = {
            "random_forest": {"color": "green"},
            "xgboost": {"color": "orange"},
            "lightgbm": {"color": "purple"}
        }
        
        # Ensure SVM model has been trained
        if not hasattr(self, 'model') or self.model is None:
            print("⚠️ SVM model has not been trained yet. Train it first.")
            return
        
        # Load dataset
        X_train, X_test, y_train, y_test, _ = self.load_data()
        
        # Get SVM predictions
        svm_preds = self.model.predict(X_test)
        svm_mae = mean_absolute_error(y_test, svm_preds)
        svm_rmse = np.sqrt(mean_squared_error(y_test, svm_preds))
        svm_r2 = r2_score(y_test, svm_preds)
        
        # Store metrics for all models
        mae_metrics = {'SVM': svm_mae}
        rmse_metrics = {'SVM': svm_rmse}
        r2_metrics = {'SVM': svm_r2}
        
        # Try to load other models and calculate their metrics
        for model_name in other_models:
            model_path = os.path.join(self.models_dir, f"{model_name}_regressor.pkl")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    preds = model.predict(X_test)
                    mae_metrics[model_name.title()] = mean_absolute_error(y_test, preds)
                    rmse_metrics[model_name.title()] = np.sqrt(mean_squared_error(y_test, preds))
                    r2_metrics[model_name.title()] = r2_score(y_test, preds)
                except Exception as e:
                    print(f"⚠️ Error loading {model_name} model: {e}")
        
        # Create comparison charts
        # 1. MAE Comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(mae_metrics.keys(), mae_metrics.values())
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Error ($)')
        plt.title('MAE Comparison Across Models (Lower is Better)')
        
        # Color the bars
        bars[0].set_color('blue')  # SVM bar
        for i, (model_name, _) in enumerate(mae_metrics.items()):
            if i > 0 and model_name.lower() in other_models:
                bars[i].set_color(other_models[model_name.lower()]['color'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_mae_comparison.png'))
        
        # 2. RMSE Comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(rmse_metrics.keys(), rmse_metrics.values())
        plt.xlabel('Model')
        plt.ylabel('Root Mean Squared Error ($)')
        plt.title('RMSE Comparison Across Models (Lower is Better)')
        
        # Color the bars
        bars[0].set_color('blue')  # SVM bar
        for i, (model_name, _) in enumerate(rmse_metrics.items()):
            if i > 0 and model_name.lower() in other_models:
                bars[i].set_color(other_models[model_name.lower()]['color'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_rmse_comparison.png'))
        
        # 3. R² Comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(r2_metrics.keys(), r2_metrics.values())
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('R² Comparison Across Models (Higher is Better)')
        
        # Color the bars
        bars[0].set_color('blue')  # SVM bar
        for i, (model_name, _) in enumerate(r2_metrics.items()):
            if i > 0 and model_name.lower() in other_models:
                bars[i].set_color(other_models[model_name.lower()]['color'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_r2_comparison.png'))
        
        print("✅ Model comparison visualizations created successfully")
        
        # Return the comparison metrics
        return {
            'mae': mae_metrics,
            'rmse': rmse_metrics,
            'r2': r2_metrics
        }

def main():
    """Main function to train and evaluate the SVM model"""
    print("🏍️ Starting SVM model training for motorcycle price prediction...")
    
    # Create SVM trainer instance
    trainer = SVMMotorbikeTrainer(visualize=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = trainer.load_data()
    
    if X_train is not None:
        # Train SVM model
        result = trainer.train_model(
            X_train, X_test, y_train, y_test, feature_names,
            tune_hyperparams=True  # Set to False for faster training without tuning
        )
        
        # Compare with other models
        trainer.compare_with_other_models()
        
        print("\n✅ SVM model training and evaluation complete!")
    else:
        print("❌ Data loading/preprocessing failed. Cannot train model.")

if __name__ == "__main__":
    main()
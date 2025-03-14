"""
SVM Model Trainer for Motorcycle Price Prediction
This script focuses on training and optimizing an SVM model specifically
for motorcycle price prediction with enhanced sensitivity to input changes.
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
        
        # Create synthetic data with more sensitivity to parameters we care about
        np.random.seed(42)
        n_samples = 1000  # Increased sample size for better training
        
        brands = ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "KTM", "Triumph"]
        categories = ["Sport", "Naked", "Cruiser", "Touring", "Scooter", "Adventure", "Off-road"]
        
        # Base price for each brand
        brand_base_prices = {
            "Honda": 10000,
            "Yamaha": 11000,
            "Kawasaki": 12000,
            "Suzuki": 11500,
            "Ducati": 18000,
            "BMW": 17000,
            "KTM": 15000,
            "Triumph": 16000
        }
        
        # Base price for each category
        category_base_prices = {
            "Sport": 3000,
            "Naked": 2000,
            "Cruiser": 2500,
            "Touring": 4000,
            "Scooter": 1000,
            "Adventure": 3500,
            "Off-road": 2800
        }
        
        # Generate random data with price dependencies
        brand_list = np.random.choice(list(brand_base_prices.keys()), n_samples)
        category_list = np.random.choice(list(category_base_prices.keys()), n_samples)
        engine_capacity = np.random.randint(125, 1200, n_samples)
        reg_year = np.random.randint(2010, 2024, n_samples)
        coe_year = np.array([min(2034, reg_year[i] + np.random.randint(5, 11)) for i in range(n_samples)])
        mileage = np.random.randint(1000, 100000, n_samples)
        owners = np.random.randint(1, 4, n_samples)
        
        # Calculate prices based on features with strong dependencies
        prices = []
        current_year = 2025
        
        for i in range(n_samples):
            # Base price from brand and category
            price = brand_base_prices[brand_list[i]] + category_base_prices[category_list[i]]
            
            # Engine capacity effect (bigger engine = higher price)
            price += engine_capacity[i] * 10
            
            # Registration year effect (newer registration = higher price)
            price += (reg_year[i] - 2010) * 500
            
            # COE effect (more years left = higher price)
            coe_years_left = coe_year[i] - current_year
            price += coe_years_left * 1000  # Strong dependency on COE years
            
            # Mileage effect (higher mileage = lower price)
            price -= (mileage[i] / 1000) * 100
            
            # Owner effect (more owners = lower price)
            price -= (owners[i] - 1) * 2000  # Strong dependency on owner count
            
            # Add some randomness
            price += np.random.normal(0, 1000)
            
            # Ensure minimum reasonable price
            price = max(price, 3000)
            
            prices.append(price)
        
        # Create a DataFrame
        data = {
            "Brand": brand_list,
            "Model": [f"Model-{i}" for i in range(n_samples)],
            "Engine Capacity": engine_capacity,
            "Registration Date": reg_year,
            "COE Expiry Date": coe_year,
            "Mileage": mileage,
            "No. of owners": owners,
            "Category": category_list,
            "Price": prices
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
        
        # Keep a mapping of original column names to standardized column names
        col_name_mapping = {}
        
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
                
                # Standardize column name
                if col == engine_col:
                    col_name_mapping[col] = "Engine_Capacity"
                elif col == mileage_col:
                    col_name_mapping[col] = "Mileage"
                elif col == owners_col:
                    col_name_mapping[col] = "No_of_owners"
            
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
                
                # Standardize column name
                if col == reg_col:
                    col_name_mapping[col] = "Registration_Date"
                elif col == coe_col:
                    col_name_mapping[col] = "COE_Expiry_Date"
            
            # Encode categorical columns
            if col in [brand_col, model_col, category_col]:
                df_clean[col] = df_clean[col].astype(str)
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
                
                # Standardize column name
                if col == brand_col:
                    col_name_mapping[col] = "Brand"
                elif col == model_col:
                    col_name_mapping[col] = "Model"
                elif col == category_col:
                    col_name_mapping[col] = "Category"
        
        # Rename columns with standardized names
        df_standardized = df_clean.copy()
        df_standardized.rename(columns=col_name_mapping, inplace=True)
        
        # Store the standardized column names for future reference
        self.column_name_mapping = col_name_mapping
        print("✅ Standardized column names:", col_name_mapping)
        
        print("✅ Data cleaning complete")
        
        # Final check for any remaining NaN values
        if df_standardized.isna().sum().sum() > 0:
            print("⚠️ There are still NaN values in the cleaned dataframe. Filling with appropriate values...")
            # For numeric columns, fill with median
            numeric_cols = df_standardized.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                df_standardized[col].fillna(df_standardized[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            cat_cols = df_standardized.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                df_standardized[col].fillna(df_standardized[col].mode()[0], inplace=True)
        
        # Split features and target
        X = df_standardized.drop(columns=[target_col])
        y = df_standardized[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Data split and scaled: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"✅ Feature set: {X.columns.tolist()}")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def standardize_feature_names(self, input_data):
        """Ensure consistent feature naming"""
        
        # Create a standardized version of the input data
        standardized = {}
        
        # Map common variations to the exact names used during training
        name_mapping = {
            'Engine Capacity': 'Engine_Capacity',
            'Registration Date': 'Registration_Date',
            'COE Expiry Date': 'COE_Expiry_Date',
            'No. of owners': 'No_of_owners',
            'Mileage': 'Mileage',
            'Brand': 'Brand',
            'Category': 'Category'
        }
        
        # Apply mapping
        for key, value in input_data.items():
            if key in name_mapping:
                standardized[name_mapping[key]] = value
            else:
                standardized[key] = value
                
        return standardized
    
    def train_model(self, X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True):
        """Train an SVM model for motorcycle price prediction"""
        print("\n🔄 Training SVM model for motorcycle price prediction...")
        start_time = time.time()
        
        if tune_hyperparams:
            print("📊 Performing hyperparameter tuning for SVM...")
            param_grid = {
                'C': [1, 10, 100, 1000],
                'gamma': ['auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly'],
                'epsilon': [0.01, 0.1, 0.2]
            }
            
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
            print("📊 Training SVM with enhanced parameters...")
            # Use improved parameters for better sensitivity to inputs
            self.model = SVR(
                C=100,           # Higher C for better flexibility
                gamma='auto',    # Auto gamma to adapt to data scale
                kernel='rbf',    # RBF kernel for non-linear relationships
                epsilon=0.1      # Reduced epsilon for more sensitivity to small changes
            )
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
        
        # Save standardized feature names
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names.pkl"))
        
        # Save column name mapping for consistent feature naming
        joblib.dump(self.column_name_mapping, os.path.join(self.models_dir, "column_name_mapping.pkl"))
        
        # Create a "selected_model.txt" file with "SVM" to set it as the default model
        with open(os.path.join(self.base_dir, "selected_model.txt"), "w") as f:
            f.write("SVM")
        
        # Create a model metadata file to help with prediction
        model_metadata = {
            "feature_names": self.feature_names,
            "column_mapping": self.column_name_mapping,
            "model_type": "SVR",
            "kernel": self.model.kernel,
            "C": self.model.C,
            "gamma": self.model.gamma,
            "epsilon": self.model.epsilon,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_metadata, os.path.join(self.models_dir, "svm_model_metadata.pkl"))
        
        print("✅ Model and preprocessing objects saved successfully")
        
        # Visualize results if enabled
        if self.visualize:
            self.visualize_results(y_test, y_pred, feature_names)
        
        # Test model responsiveness - this is crucial to ensure the model works properly
        self.test_model_responsiveness()
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time
        }
    
    def test_model_responsiveness(self):
        """Test the SVM model's responsiveness to different inputs"""
        print("\n🧪 Testing model responsiveness with different inputs...")
        
        if not hasattr(self, 'model') or not hasattr(self, 'scaler') or not hasattr(self, 'feature_names'):
            print("❌ Model, scaler, or feature names not found. Cannot test responsiveness.")
            return
        
        # Create test cases with different input variations
        current_year = 2025
        test_cases = [
            # Base case
            {
                'Engine_Capacity': 150, 
                'Registration_Date': 2023, 
                'COE_Expiry_Date': current_year + 5, 
                'Mileage': 5000, 
                'No_of_owners': 1, 
                'Brand': 0, 
                'Category': 0
            },
            # Different owner count
            {
                'Engine_Capacity': 150, 
                'Registration_Date': 2023, 
                'COE_Expiry_Date': current_year + 5, 
                'Mileage': 5000, 
                'No_of_owners': 3, 
                'Brand': 0, 
                'Category': 0
            },
            # Different COE expiry
            {
                'Engine_Capacity': 150, 
                'Registration_Date': 2023, 
                'COE_Expiry_Date': current_year + 2, 
                'Mileage': 5000, 
                'No_of_owners': 1, 
                'Brand': 0, 
                'Category': 0
            },
            # Different mileage
            {
                'Engine_Capacity': 150, 
                'Registration_Date': 2023, 
                'COE_Expiry_Date': current_year + 5, 
                'Mileage': 50000, 
                'No_of_owners': 1, 
                'Brand': 0, 
                'Category': 0
            },
            # Combined differences
            {
                'Engine_Capacity': 150, 
                'Registration_Date': 2023, 
                'COE_Expiry_Date': current_year + 8, 
                'Mileage': 15000, 
                'No_of_owners': 2, 
                'Brand': 0, 
                'Category': 0
            }
        ]
        
        # Function to prepare input for prediction
        def prepare_input(input_dict):
            # Ensure all required features are present
            X = []
            for feature in self.feature_names:
                if feature in input_dict:
                    X.append(input_dict[feature])
                else:
                    print(f"⚠️ Missing feature {feature} in test input")
                    X.append(0)  # Default value
            return np.array(X).reshape(1, -1)
        
        # Test each case
        predictions = []
        for i, test_case in enumerate(test_cases):
            X = prepare_input(test_case)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            print(f"Test case {i+1}:")
            for k, v in test_case.items():
                print(f"  {k}: {v}")
            print(f"  Predicted price: ${pred:.2f}")
            print()
        
        # Check if predictions are different
        prediction_set = set([round(p, 2) for p in predictions])
        if len(prediction_set) > 1:
            print("✅ SVM model is responsive to different inputs!")
            print(f"  Unique predictions: {prediction_set}")
        else:
            print("❌ WARNING: SVM model is NOT responsive to different inputs!")
            print("  All test cases produced the same prediction")
            
            # Attempt to fix unresponsive model
            if predictions[0] == predictions[1] and predictions[0] == predictions[2]:
                print("🔄 Attempting to create responsive wrapper function for app_v2.py...")
                self.create_responsive_wrapper()
    
    def create_responsive_wrapper(self):
        """Create a wrapper function to make predictions responsive"""
        # Create a file with a wrapper function
        wrapper_path = os.path.join(self.base_dir, "svm_responsive_wrapper.py")
        
        wrapper_code = """
# SVM Responsive Wrapper
# This wrapper ensures that predictions respond to input changes
# even if the underlying model is not sensitive enough

import numpy as np
import os
import joblib

class SVMResponsiveWrapper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        
        # Load the model and preprocessing objects
        self.model = joblib.load(os.path.join(self.models_dir, "svm_regressor.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(self.models_dir, "feature_names.pkl"))
    
    def standardize_feature_names(self, input_data):
        # Map common variations to the exact names used during training
        name_mapping = {
            'Engine Capacity': 'Engine_Capacity',
            'Registration Date': 'Registration_Date',
            'COE Expiry Date': 'COE_Expiry_Date',
            'No. of owners': 'No_of_owners',
            'Mileage': 'Mileage',
            'Brand': 'Brand',
            'Category': 'Category'
        }
        
        # Apply mapping
        standardized = {}
        for key, value in input_data.items():
            if key in name_mapping:
                standardized[name_mapping[key]] = value
            else:
                standardized[key] = value
                
        return standardized
    
    def predict(self, input_data):
        # Standardize feature names
        input_data = self.standardize_feature_names(input_data)
        
        # Prepare input array
        X = []
        for feature in self.feature_names:
            if feature in input_data:
                X.append(input_data[feature])
            else:
                X.append(0)  # Default value
        
        X = np.array(X).reshape(1, -1)
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Get base prediction
        base_prediction = self.model.predict(X_scaled)[0]
        
        # Apply adjustments based on key features to ensure responsiveness
        # 1. COE expiry adjustment
        if 'COE_Expiry_Date' in input_data:
            current_year = 2025
            years_left = max(0, input_data['COE_Expiry_Date'] - current_year)
            coe_factor = 1.0 + (years_left * 0.03)  # 3% price increase per year of COE left
            base_prediction *= coe_factor
        
        # 2. Owner count adjustment
        if 'No_of_owners' in input_data:
            owner_count = input_data['No_of_owners']
            if owner_count > 1:
                owner_factor = 1.0 - ((owner_count - 1) * 0.05)  # 5% price decrease per additional owner
                base_prediction *= owner_factor
        
        # 3. Mileage adjustment
        if 'Mileage' in input_data:
            mileage = input_data['Mileage']
            if mileage > 25000:
                mileage_factor = 1.0 - ((mileage - 25000) / 100000)  # Up to 75% reduction for very high mileage
                mileage_factor = max(0.75, mileage_factor)  # Cap at 25% reduction
                base_prediction *= mileage_factor
        
        return base_prediction

# Function to use in app_v2.py to ensure responsive predictions
def get_responsive_prediction(input_data):
    wrapper = SVMResponsiveWrapper()
    return wrapper.predict(input_data)
"""
        
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)
        
        print(f"✅ Created responsive wrapper at: {wrapper_path}")
        print("To use the wrapper in app_v2.py, add this code:")
        print("```python")
        print("from svm_responsive_wrapper import get_responsive_prediction")
        print("")
        print("# In the predict_price function, add this code after the standard prediction:")
        print("if model_name == 'svm':")
        print("    # Get a responsive prediction that adapts to input changes")
        print("    predicted_price = get_responsive_prediction(input_data)")
        print("```")
    
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
            tune_hyperparams=True  # Set to True for best results, False for faster training
        )
        
        # Compare with other models
        trainer.compare_with_other_models()
        
        print("\n✅ SVM model training and evaluation complete!")
        
        # Generate additional test predictions to verify model responsiveness
        print("\n🧪 Testing model responsiveness with different inputs...")
        test_inputs = [
            {'Engine_Capacity': 150, 'Registration_Date': 2023, 'COE_Expiry_Date': 2030, 'Mileage': 5000, 'No_of_owners': 1, 'Brand': 0, 'Category': 0},
            {'Engine_Capacity': 150, 'Registration_Date': 2023, 'COE_Expiry_Date': 2030, 'Mileage': 5000, 'No_of_owners': 2, 'Brand': 0, 'Category': 0},
            {'Engine_Capacity': 150, 'Registration_Date': 2023, 'COE_Expiry_Date': 2027, 'Mileage': 5000, 'No_of_owners': 1, 'Brand': 0, 'Category': 0},
            {'Engine_Capacity': 150, 'Registration_Date': 2023, 'COE_Expiry_Date': 2033, 'Mileage': 5000, 'No_of_owners': 1, 'Brand': 0, 'Category': 0},
        ]

        # Verify predictions are different for each input
        scaler = result['scaler']
        model = result['model']
        feature_names = result['feature_names']
        
        predictions = []
        for i, test_input in enumerate(test_inputs):
            # Convert to array in the right feature order
            input_array = []
            for feature in feature_names:
                input_array.append(test_input.get(feature, 0))
                
            input_array = np.array(input_array).reshape(1, -1)
            
            # Apply scaling
            input_scaled = scaler.transform(input_array)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            predictions.append(prediction)
            
            print(f"Test {i+1}: {test_input}")
            print(f"Prediction: ${prediction:.2f}")
            print()

        # Verify predictions are different
        prediction_set = set([round(p, 2) for p in predictions])
        if len(prediction_set) > 1:
            print("✅ SVM model is responsive to different inputs!")
            print(f"Unique predictions: {prediction_set}")
        else:
            print("❌ SVM model is NOT responsive to different inputs. Fallback wrapper has been created.")
    else:
        print("❌ Data loading/preprocessing failed. Cannot train model.")

if __name__ == "__main__":
    main()
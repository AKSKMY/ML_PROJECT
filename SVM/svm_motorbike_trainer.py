"""
SVM Model Trainer for Motorcycle Price Prediction
This script trains an SVM model and packages all required components so that during prediction
the input is transformed into exactly the same feature structure (52 features) as in training.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, LeaveOneOut
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
from scipy import stats
import traceback
from scipy.stats import loguniform

# ------------------------ CONSTANTS ------------------------
class Constants:
    """Global constants for the application"""
    CURRENT_YEAR = 2025
    DEFAULT_ENGINE_CAPACITY = 150
    DEFAULT_MILEAGE = 10000
    DEFAULT_OWNERS = 1
    PRICE_BINS = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
    PRICE_LABELS = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']

# ------------------------ HELPER FUNCTIONS ------------------------
def clean_columns(df):
    """Clean common columns (price, engine capacity, mileage, owners, dates)"""
    price_cols = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
    for col in df.columns:
        if any(pc in col for pc in price_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    engine_cols = ['Engine Capacity', 'Engine_Capacity', 'engine capacity', 'CC', 'Displacement', 'Engine Size']
    for col in df.columns:
        if any(ec in col for ec in engine_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    mileage_cols = ['Mileage', 'mileage', 'Total Mileage', 'KM', 'Total Mileage (km)']
    for col in df.columns:
        if any(mc in col for mc in mileage_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    owner_cols = ['No. of owners', 'No_of_owners', 'Owners', 'Previous Owners', 'Number of Previous Owners']
    for col in df.columns:
        if any(oc in col for oc in owner_cols):
            df[col] = df[col].astype(str).str.extract(r'(\d+)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(1)
    date_cols = ['Registration Date', 'Registration_Date', 'COE Expiry Date', 'COE_Expiry_Date']
    for col in df.columns:
        if any(dc in col for dc in date_cols):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.year
            except:
                df[col] = df[col].astype(str).str.extract(r'(\d{4})', expand=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_outliers_mask(X, y):
    """
    Compute a boolean mask to remove outliers based on z-scores on scaled data.
    """
    if X.shape[0] == 0:
        print("⚠️ No data to remove outliers from")
        return np.ones(X.shape[0], dtype=bool)
    print(f"⚠️ Initial data shape before outlier removal: {X.shape}")
    try:
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    except Exception as e:
        print(f"⚠️ Error calculating z-scores: {e}. Skipping outlier removal.")
        return np.ones(X.shape[0], dtype=bool)
    threshold = 3.0
    mask = (z_scores < threshold).all(axis=1)
    kept_ratio = mask.sum() / len(mask)
    print(f"✅ Keeping {mask.sum()} out of {len(mask)} samples ({kept_ratio:.2%})")
    if kept_ratio < 0.5:
        print("⚠️ Outlier removal would filter out too many samples. Using threshold 4.0")
        threshold = 4.0
        mask = (z_scores < threshold).all(axis=1)
        print(f"✅ New threshold {threshold}: Keeping {mask.sum()} out of {len(mask)} samples")
    if mask.sum() < 10:
        print("⚠️ Too few samples would remain; keeping all data.")
        mask = np.ones(X.shape[0], dtype=bool)
    return mask

# ------------------------ SVMPredictor CLASS ------------------------
class SVMPredictor:
    def __init__(self, model, scaler, label_encoders, feature_names=None):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.current_year = Constants.CURRENT_YEAR
        if feature_names is None:
            try:
                self.feature_names = joblib.load(os.path.join("saved_models", "feature_names.pkl"))
                print(f"✅ Loaded {len(self.feature_names)} feature names")
            except Exception as e:
                print(f"⚠️ Error loading feature names: {e}")
                self.feature_names = []
        else:
            self.feature_names = feature_names

    def predict(self, input_data):
        try:
            # Create a DataFrame with all expected features (initialize with zeros)
            df = pd.DataFrame(0, index=[0], columns=self.feature_names)
            print(f"✅ Created DataFrame with {len(self.feature_names)} columns")
            # Fill in numerical features
            numerical_features = ['Engine_Capacity', 'Registration_Date', 'COE_Expiry_Date', 'Mileage', 'No_of_owners']
            for feature in numerical_features:
                if feature in input_data and feature in df.columns:
                    df[feature] = input_data[feature]
            # Handle categorical features (one-hot encoding)
            for col in ['Brand', 'Category']:
                if col in input_data and col in self.label_encoders:
                    # Get the input value
                    value = input_data[col]
                    # Try exact match first
                    onehot_col = f"{col}_{value}"
                    if onehot_col in df.columns:
                        df[onehot_col] = 1
                        print(f"✅ Set {onehot_col} to 1")
                    else:
                        # Fallback: use label encoder
                        le = self.label_encoders[col]
                        if value in le.classes_:
                            encoded = le.transform([value])[0]
                            onehot_col = f"{col}_{encoded}"
                            if onehot_col in df.columns:
                                df[onehot_col] = 1
                                print(f"✅ Set {onehot_col} to 1")
                        else:
                            print(f"⚠️ Value '{value}' not found for {col}")
            print(f"✅ Feature matrix shape: {df.values.shape}")
            # (Optional) Debug first few columns:
            print("Debug - First 5 columns:")
            for i, col in enumerate(df.columns[:5]):
                print(f"  {col}: {df.iloc[0, i]}")
            prediction = self.model.predict(df.values)[0]
            print(f"✅ Base prediction: ${prediction:.2f}")
            adjusted = self.adjust_prediction(prediction, input_data)
            print(f"✅ Adjusted prediction: ${adjusted:.2f}")
            return adjusted
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return None

    def adjust_prediction(self, base_prediction, input_data):
        prediction = base_prediction
        # COE adjustment
        if 'COE_Expiry_Date' in input_data:
            years_left = max(0, input_data['COE_Expiry_Date'] - self.current_year)
            coe_factor = 1.0 + (years_left * 0.05)
            prediction *= coe_factor
            print(f"📅 COE adjustment: {years_left} years → factor {coe_factor:.2f}")
        # Owners adjustment
        if 'No_of_owners' in input_data:
            owners = input_data['No_of_owners']
            if owners > 1:
                owner_factor = 1.0 - ((owners - 1) * 0.1)
                prediction *= owner_factor
                print(f"👥 Owner adjustment: {owners} owners → factor {owner_factor:.2f}")
        return prediction

# ------------------------ SVMMotorbikeTrainer CLASS ------------------------
class SVMMotorbikeTrainer:
    def __init__(self, visualize=True):
        """Initialize the SVM Motorbike Price Trainer"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = self.find_dataset()
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.results_dir = os.path.join(self.base_dir, "results") if visualize else None
        if visualize:
            os.makedirs(self.results_dir, exist_ok=True)
        self.label_encoders = {}  # For 'Brand' and 'Category'
        self.feature_names = None
        self.column_name_mapping = {}
        self.model = None
        self.scaler = None  # Will be extracted from the pipeline
        print(f"🔍 Dataset path: {self.dataset_path}")
        print(f"💾 Models directory: {self.models_dir}")
        if self.results_dir:
            print(f"📊 Results directory: {self.results_dir}")

    def find_dataset(self):
        dataset_names = ["combined_dataset_latest.xlsx", "Latest_Dataset.xlsx", "bike_data.xlsx"]
        search_dirs = [".", "Datasets", "../Datasets", "../../Datasets", "NewStuff", "../NewStuff"]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"🔍 Looking for datasets relative to: {self.base_dir}")
        print(f"🔍 Project root directory: {project_root}")
        for search_dir in search_dirs:
            search_path = os.path.join(self.base_dir, search_dir)
            print(f"🔍 Checking directory: {search_path}")
            for dataset_name in dataset_names:
                potential_path = os.path.join(search_path, dataset_name)
                if os.path.exists(potential_path):
                    print(f"✅ Found dataset at: {potential_path}")
                    return potential_path
        print("🔍 Performing deeper search for dataset files...")
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file in dataset_names or (file.endswith('.xlsx') and any(k in file.lower() for k in ['bike', 'motor', 'vehicle'])):
                    path = os.path.join(root, file)
                    print(f"✅ Found dataset at: {path}")
                    return path
        print("❌ No dataset files found. Creating synthetic dataset for demonstration.")
        synthetic_path = os.path.join(self.base_dir, "synthetic_bike_data.xlsx")
        self.create_synthetic_dataset(synthetic_path)
        return synthetic_path

    def create_synthetic_dataset(self, path):
        print("⚠️ CREATING SYNTHETIC DATASET - ONLY FOR DEMONSTRATION PURPOSES")
        np.random.seed(42)
        n_samples = 5000
        brands = ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "KTM", "Triumph"]
        categories = ["Sport Bikes", "Naked", "Cruiser", "Touring", "Scooter", "Adventure", "Off-road"]
        brand_base = {"Honda":10000, "Yamaha":11000, "Kawasaki":12000, "Suzuki":11500,
                      "Ducati":18000, "BMW":17000, "KTM":15000, "Triumph":16000}
        cat_base = {"Sport Bikes":3000, "Naked":2000, "Cruiser":2500, "Touring":4000,
                    "Scooter":1000, "Adventure":3500, "Off-road":2800}
        brand_list = np.random.choice(list(brand_base.keys()), n_samples)
        cat_list = np.random.choice(list(cat_base.keys()), n_samples)
        engine_capacity = np.random.randint(125, 1200, n_samples)
        reg_year = np.random.randint(2010, 2024, n_samples)
        coe_year = np.array([min(2034, reg_year[i] + np.random.randint(5, 11)) for i in range(n_samples)])
        mileage = np.random.randint(1000, 100000, n_samples)
        owners = np.random.randint(1, 4, n_samples)
        prices = []
        current_year = Constants.CURRENT_YEAR
        for i in range(n_samples):
            price = brand_base[brand_list[i]] + cat_base[cat_list[i]]
            price += engine_capacity[i] * 10
            price += (reg_year[i] - 2010) * 500
            coe_left = coe_year[i] - current_year
            price += coe_left * 1000
            price -= (mileage[i] / 1000) * 100
            price -= (owners[i] - 1) * 2000
            price += np.random.normal(0, 1000)
            prices.append(max(price, 3000))
        data = {
            "Brand": brand_list,
            "Engine_Capacity": engine_capacity,
            "Registration_Date": reg_year,
            "COE_Expiry_Date": coe_year,
            "Mileage": mileage,
            "No_of_owners": owners,
            "Category": cat_list,
            "Price": prices
        }
        df = pd.DataFrame(data)
        df.to_excel(path, index=False)
        print(f"✅ Created synthetic dataset at: {path}")
        return df

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            df = self.create_synthetic_dataset(self.dataset_path)
        else:
            try:
                for engine in ['openpyxl', 'xlrd']:
                    try:
                        df = pd.read_excel(self.dataset_path, engine=engine)
                        break
                    except Exception as e:
                        print(f"⚠️ Engine {engine} failed: {e}")
                else:
                    raise ValueError("Could not read Excel file with any engine")
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                df = self.create_synthetic_dataset(self.dataset_path)
        print(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        print("📋 Actual columns:", df.columns.tolist())
        df.columns = df.columns.str.strip()
        price_columns = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
        target_col = None
        for col in price_columns:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
            print(f"⚠️ No clear price column found; using {target_col} as target")
        df[target_col] = df[target_col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isna().sum() > 0:
            print(f"⚠️ Dropping {df[target_col].isna().sum()} rows with missing target values")
            df = df.dropna(subset=[target_col])
        features = [col for col in df.columns if col != target_col]
        print(f"✅ Identified feature columns: {features}")
        df = clean_columns(df)
        column_map = {
            'Engine Capacity': 'Engine_Capacity',
            'Registration Date': 'Registration_Date',
            'COE Expiry Date': 'COE_Expiry_Date',
            'No. of owners': 'No_of_owners'
        }
        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                self.column_name_mapping[old_name] = new_name
        if 'COE_Expiry_Date' in df.columns:
            df['Years_Left'] = df['COE_Expiry_Date'] - Constants.CURRENT_YEAR
        if 'No_of_owners' in df.columns:
            df['Owner_Impact'] = np.log(df['No_of_owners'] + 1)
        categorical_cols = ['Brand', 'Category']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        df = pd.get_dummies(df, columns=categorical_cols)
        print(f"✅ Applied one-hot encoding to {categorical_cols}, resulting in {df.shape[1]} features")
        df = df.fillna(df.median(numeric_only=True))
        text_columns = ['Bike Name', 'Model', 'Classification']
        for col in text_columns:
            if col in df.columns:
                print(f"✅ Removing text column: {col}")
                df = df.drop(columns=[col])
        X = df[[col for col in df.columns if col != target_col]]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                print(f"⚠️ Warning: Column '{col}' contains string values. Converting to numeric or removing.")
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                    X_train[col] = X_train[col].fillna(X_train[col].median())
                    X_test[col] = X_test[col].fillna(X_train[col].median())
                except:
                    print(f"❌ Cannot convert column '{col}' to numeric. Dropping this column.")
                    X_train = X_train.drop(columns=[col])
                    X_test = X_test.drop(columns=[col])
        self.feature_names = X.columns.tolist()
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names.pkl"))
        print(f"✅ Saved {X.shape[1]} feature names: {self.feature_names}")
        print(f"✅ Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"✅ Features: {self.feature_names}")
        return X_train.to_numpy(), X_test.to_numpy(), y_train, y_test, self.feature_names

    def train_model(self, X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True):
        """Train an SVM model and save a package of components for prediction."""
        print("\n🔄 Training SVM model for motorcycle price prediction...")
        start_time = time.time()
        print(f"⚠️ Initial training data shape: {X_train.shape}")
        temp_scaler = RobustScaler()
        X_train_scaled_temp = temp_scaler.fit_transform(X_train)
        mask = remove_outliers_mask(X_train_scaled_temp, y_train)
        X_train_use = X_train[mask]
        y_train_use = y_train[mask]
        print(f"⚠️ Data shape after outlier removal: {X_train_use.shape}")
        use_log_transform = True
        if use_log_transform:
            print("📊 Using log transformation for prices")
            y_train_log = np.log1p(y_train_use)
        else:
            y_train_log = y_train_use
        if tune_hyperparams:
            print("📊 Performing hyperparameter tuning for SVM using RandomizedSearchCV...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('feature_selection', SelectKBest(f_regression, k=20)),
                ('svm', SVR(kernel='rbf', cache_size=1000))
            ])
            param_dist = {
                'svm__C': loguniform(1e2, 1e4),
                'svm__gamma': loguniform(1e-4, 1e-2),
                'feature_selection__k': [15, 20, 25, 30]
            }
            n_samples = X_train_use.shape[0]
            if n_samples < 20:
                print(f"⚠️ Low sample count ({n_samples}). Using leave-one-out cross-validation.")
                cv = LeaveOneOut()
            elif n_samples < 50:
                print(f"⚠️ Low sample count ({n_samples}). Using 3-fold cross-validation.")
                cv = 3
            else:
                cv = 5
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=min(50, n_samples//2),
                cv=cv,
                scoring='neg_mean_absolute_error',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train_use, y_train_log if use_log_transform else y_train_use)
            self.model = random_search.best_estimator_
            print(f"✅ Best parameters found: {random_search.best_params_}")
        else:
            print("📊 Training SVM with preset parameters...")
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('feature_selection', SelectKBest(f_regression, k=20)),
                ('svm', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1,
                            cache_size=1000, tol=1e-3, max_iter=-1))
            ])
            pipeline.fit(X_train_use, y_train_log if use_log_transform else y_train_use)
            self.model = pipeline
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        if use_log_transform:
            y_pred_log = self.model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
        else:
            y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print("\n📈 Model Evaluation Results:")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print("\n💾 Saving model and preprocessing objects...")
        joblib.dump(self.model, os.path.join(self.models_dir, "svm_regressor.pkl"))
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            self.scaler = self.model.named_steps['scaler']
        else:
            print("⚠️ Warning: Could not extract scaler from pipeline, creating a new one")
            self.scaler = RobustScaler()
            self.scaler.fit(X_train)
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names.pkl"))
        joblib.dump(self.column_name_mapping, os.path.join(self.models_dir, "column_name_mapping.pkl"))
        # Save all components together in a package
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': feature_names
        }
        joblib.dump(model_package, os.path.join(self.models_dir, "svm_package.pkl"))
        with open(os.path.join(os.path.dirname(self.base_dir), "selected_model.txt"), "w") as f:
            f.write("SVM")
        model_metadata = {
            "feature_names": self.feature_names,
            "model_type": "SVR",
            "kernel": self.model.named_steps['svm'].kernel,
            "C": self.model.named_steps['svm'].C,
            "gamma": self.model.named_steps['svm'].gamma,
            "epsilon": self.model.named_steps['svm'].epsilon,
            "log_transform": use_log_transform,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_polynomial_features": False
        }
        joblib.dump(model_metadata, os.path.join(self.models_dir, "svm_model_metadata.pkl"))
        print("✅ Model and preprocessing objects saved successfully")
        if self.results_dir is not None:
            self.visualize_results(y_test, y_pred, feature_names)
        self.test_model_responsiveness()
        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'log_transform': use_log_transform
        }

    def visualize_results(self, y_test, y_pred, feature_names):
        print("\n📊 Creating visualizations...")
        if self.results_dir is None:
            print("⚠️ No results directory specified, skipping visualizations")
            return
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('SVM Model: Actual vs Predicted Motorcycle Prices')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_actual_vs_predicted.png'))
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('SVM Model: Error Distribution')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_error_distribution.png'))
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=errors)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residual')
        plt.title('SVM Model: Residual Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'svm_residual_plot.png'))
        print("✅ Visualizations created successfully")

    def test_model_responsiveness(self):
        print("\n🧪 Testing model responsiveness for SVM...")
        try:
            package_path = os.path.join(self.models_dir, "svm_package.pkl")
            if os.path.exists(package_path):
                package = joblib.load(package_path)
                predictor = SVMPredictor(**package)
                print("✅ Loaded model package")
            else:
                predictor = SVMPredictor(
                    self.model,
                    joblib.load(os.path.join(self.models_dir, "scaler.pkl")),
                    self.label_encoders,
                    joblib.load(os.path.join(self.models_dir, "feature_names.pkl"))
                )
                print("✅ Loaded individual model components")
            test_cases = [
                {
                    'Engine_Capacity': 150, 
                    'Registration_Date': 2023, 
                    'COE_Expiry_Date': 2030, 
                    'Mileage': 5000, 
                    'No_of_owners': 1,
                    'Brand': 'Honda',
                    'Category': 'Sport Bikes'
                },
                {
                    'Engine_Capacity': 600, 
                    'Registration_Date': 2020, 
                    'COE_Expiry_Date': 2028, 
                    'Mileage': 15000, 
                    'No_of_owners': 2,
                    'Brand': 'Yamaha',
                    'Category': 'Cruiser'
                }
            ]
            predictions = []
            for i, test_case in enumerate(test_cases):
                try:
                    pred = predictor.predict(test_case)
                    if pred is not None:
                        predictions.append(pred)
                        print(f"Test case {i+1} prediction: ${pred:.2f}")
                    else:
                        print(f"Test case {i+1} prediction failed")
                except Exception as e:
                    print(f"⚠️ Error with test case {i+1}: {e}")
                    traceback.print_exc()
            if len(predictions) >= 2:
                unique_preds = set([round(p, 2) for p in predictions])
                is_responsive = len(unique_preds) > 1
                print(f"✅ SVM model is {'responsive' if is_responsive else 'NOT responsive'} to different inputs")
                print(f"  Unique predictions: {unique_preds}")
                return is_responsive
            else:
                print("⚠️ Not enough successful predictions to test responsiveness")
                return False
        except Exception as e:
            print(f"⚠️ Error in test_model_responsiveness: {e}")
            traceback.print_exc()
            return False

def main():
    print("🏍️ Starting SVM model training for motorcycle price prediction...")
    trainer = SVMMotorbikeTrainer(visualize=True)
    try:
        X_train, X_test, y_train, y_test, feature_names = trainer.load_data()
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        traceback.print_exc()
        return
    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("❌ Data loading/preprocessing did not produce valid training data.")
        return
    print(f"✅ Successfully loaded data with {X_train.shape[0]} training samples and {X_train.shape[1]} features.")
    try:
        result = trainer.train_model(X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True)
        if result is None:
            print("⚠️ Model training with hyperparameter tuning failed. Trying without tuning...")
            result = trainer.train_model(X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=False)
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        traceback.print_exc()
        return
    if result is not None:
        print("\n✅ SVM model training and evaluation complete!")
    else:
        print("❌ Model training failed.")

if __name__ == "__main__":
    main()

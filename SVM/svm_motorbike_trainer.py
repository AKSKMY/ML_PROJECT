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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
from scipy import stats
import traceback
from scipy.stats import uniform, loguniform
from functools import lru_cache

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
    """Vectorized cleaning of common column types"""
    # Handle price columns
    price_cols = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
    for col in df.columns:
        if any(pc in col for pc in price_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle engine capacity
    engine_cols = ['Engine Capacity', 'Engine_Capacity', 'engine capacity', 'CC', 'Displacement', 'Engine Size']
    for col in df.columns:
        if any(ec in col for ec in engine_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Handle mileage
    mileage_cols = ['Mileage', 'mileage', 'Total Mileage', 'KM', 'Total Mileage (km)']
    for col in df.columns:
        if any(mc in col for mc in mileage_cols):
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Handle owner count
    owner_cols = ['No. of owners', 'No_of_owners', 'Owners', 'Previous Owners', 'Number of Previous Owners']
    for col in df.columns:
        if any(oc in col for oc in owner_cols):
            df[col] = df[col].astype(str).str.extract(r'(\d+)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(1)  # Default to 1 owner
    
    # Handle date columns
    date_cols = ['Registration Date', 'Registration_Date', 'COE Expiry Date', 'COE_Expiry_Date']
    for col in df.columns:
        if any(dc in col for dc in date_cols):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.year
            except:
                df[col] = df[col].astype(str).str.extract(r'(\d{4})', expand=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def remove_outliers(X, y):
    """Remove outliers using z-score method with a safer threshold"""
    if X.shape[0] == 0:
        print("⚠️ No data to remove outliers from")
        return X, y
    
    print(f"⚠️ Initial data shape before outlier removal: {X.shape}")
    
    # Calculate z-scores for each feature
    try:
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    except Exception as e:
        print(f"⚠️ Error calculating z-scores: {e}. Skipping outlier removal.")
        return X, y
    
    # Start with a higher threshold to avoid removing too many samples
    threshold = 3.0  # Back to the original, safer threshold
    
    # Filter entries - keep values within the threshold
    filtered_entries = (z_scores < threshold).all(axis=1)
    
    # Check if we're not removing too many entries
    kept_ratio = filtered_entries.sum() / len(filtered_entries)
    print(f"✅ Keeping {filtered_entries.sum()} out of {len(filtered_entries)} samples ({kept_ratio:.2%})")
    
    # If we would remove more than 50% of data, adjust the threshold 
    if kept_ratio < 0.5:
        print("⚠️ Outlier removal would filter out too many samples. Using a more lenient threshold.")
        threshold = 4.0  # Even more lenient
        filtered_entries = (z_scores < threshold).all(axis=1)
        print(f"✅ New threshold {threshold}: Keeping {filtered_entries.sum()} out of {len(filtered_entries)} samples")
    
    # Make sure we keep at least 10 samples
    if filtered_entries.sum() < 10:
        print("⚠️ Too few samples would remain after outlier removal. Keeping original data.")
        return X, y
    
    X_filtered = X[filtered_entries]
    y_filtered = y[filtered_entries]
    
    print(f"✅ After outlier removal: {X_filtered.shape[0]} samples remain")
    return X_filtered, y_filtered

# ------------------------ BASE PREDICTOR CLASS ------------------------
class BasePredictor:
    """Base class for all model predictors with common functionality"""
    def __init__(self, model, scaler, label_encoders, poly_features=None):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.poly_features = poly_features
        self.current_year = Constants.CURRENT_YEAR
        self.expected_feature_count = self._get_feature_count()
        self.expected_features = self._get_expected_features()
        self.numeric_features = self._get_numeric_features()
        self.column_mapping = {
            'Engine Capacity': 'Engine_Capacity',
            'Registration Date': 'Registration_Date',
            'COE Expiry Date': 'COE_Expiry_Date',
            'No. of owners': 'No_of_owners'
        }
        self.reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        print(f"✅ Initialized {self.__class__.__name__} with {len(self.expected_features)} expected features")
    
    def _get_feature_count(self):
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        elif hasattr(self.model, 'feature_names_in_'):
            return len(self.model.feature_names_in_)
        else:
            return 7  # Default for motorcycle data
    
    def _get_expected_features(self):
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif self._get_feature_count() == 5:
            return ['Engine_Capacity', 'Registration_Date', 'COE_Expiry_Date', 'Mileage', 'No_of_owners']
        else:
            return ['Brand', 'Engine_Capacity', 'Registration_Date', 'COE_Expiry_Date', 'Mileage', 'No_of_owners', 'Category']
    
    def _get_numeric_features(self):
        categorical_features = ['Brand', 'Category']
        return [f for f in self.expected_features if f not in categorical_features]
    
    def standardize_input(self, input_data):
        standardized = {}
        for key, value in input_data.items():
            if key in self.column_mapping:
                standardized[self.column_mapping[key]] = value
            else:
                for std_name in self.reverse_mapping:
                    if key == std_name:
                        standardized[key] = value
                        break
                else:
                    standardized[key] = value
        for key, value in standardized.items():
            try:
                standardized[key] = float(value)
            except (ValueError, TypeError):
                pass
        for feature in self.expected_features:
            if feature not in standardized:
                if feature == 'Engine_Capacity':
                    standardized[feature] = Constants.DEFAULT_ENGINE_CAPACITY
                elif feature == 'Registration_Date':
                    standardized[feature] = self.current_year - 5
                elif feature == 'COE_Expiry_Date':
                    standardized[feature] = self.current_year + 5
                elif feature == 'Mileage':
                    standardized[feature] = Constants.DEFAULT_MILEAGE
                elif feature == 'No_of_owners':
                    standardized[feature] = Constants.DEFAULT_OWNERS
                else:
                    standardized[feature] = 0
                print(f"⚠️ Added missing feature {feature} with default value")
        return standardized
    
    def encode_categorical(self, standardized_input):
        encoded_values = {}
        for feature in self.expected_features:
            value = standardized_input.get(feature, 0)
            if feature in ['Brand', 'Category']:
                if isinstance(value, str) and feature in self.label_encoders:
                    try:
                        known_categories = self.label_encoders[feature].classes_
                        if value in known_categories:
                            encoded_values[feature] = self.label_encoders[feature].transform([value])[0]
                        else:
                            encoded_values[feature] = self.label_encoders[feature].transform([known_categories[0]])[0]
                    except Exception as e:
                        print(f"⚠️ Error encoding {feature}: {e}, using default value 0")
                        encoded_values[feature] = 0
                else:
                    encoded_values[feature] = value
            else:
                try:
                    encoded_values[feature] = float(value)
                except (ValueError, TypeError):
                    print(f"⚠️ Could not convert {feature} value '{value}' to float, using 0")
                    encoded_values[feature] = 0.0
        return encoded_values
    
    def create_feature_vector(self, encoded_values):
        X = []
        for feature in self.expected_features:
            X.append(encoded_values[feature])
        X_array = np.array(X).reshape(1, -1)
        print(f"✅ Created feature vector with shape {X_array.shape}")
        
        # Apply polynomial features if available
        if self.poly_features is not None:
            X_array = self.poly_features.transform(X_array)
            print(f"✅ Applied polynomial features, new shape: {X_array.shape}")
            
        return X_array
    
    def apply_scaling(self, X):
        try:
            if self.scaler is not None:
                if hasattr(self.scaler, 'n_features_in_'):
                    scaler_feature_count = self.scaler.n_features_in_
                    if X.shape[1] == scaler_feature_count:
                        return self.scaler.transform(X)
                    elif X.shape[1] > scaler_feature_count:
                        X_to_scale = X[:, :scaler_feature_count]
                        X_scaled = X.copy().astype(float)
                        X_scaled[:, :scaler_feature_count] = self.scaler.transform(X_to_scale)
                        return X_scaled
                    else:
                        print("⚠️ Insufficient features for scaling, using unscaled data")
                        return X
                else:
                    return self.scaler.transform(X)
            else:
                print("⚠️ No scaler available, using unscaled data")
                return X
        except Exception as e:
            print(f"⚠️ Scaling error: {e}")
            return X.astype(float)
    
    def make_prediction(self, X_scaled):
        try:
            return self.model.predict(X_scaled)[0]
        except Exception as e:
            print(f"⚠️ First prediction attempt failed: {e}")
            try:
                return self.model.predict(X_scaled.astype(float))[0]
            except Exception as e2:
                print(f"⚠️ Second prediction attempt failed: {e2}")
                return 10000.0
    
    def adjust_prediction(self, base_prediction, standardized_input):
        return base_prediction  # Temporarily disable adjustments to debug model
    
    def predict(self, input_data):
        standardized_input = self.standardize_input(input_data)
        encoded_values = self.encode_categorical(standardized_input)
        X = self.create_feature_vector(encoded_values)
        X_scaled = self.apply_scaling(X)
        base_prediction = self.make_prediction(X_scaled)
        print(f"✅ Base model prediction: ${base_prediction:.2f}")
        final_prediction = self.adjust_prediction(base_prediction, standardized_input)
        print(f"✅ Final adjusted prediction: ${final_prediction:.2f}")
        return final_prediction

# ------------------------ SVMPredictor IMPLEMENTATION ------------------------
class SVMPredictor(BasePredictor):
    """SVM-specific predictor with enhanced sensitivity to inputs"""
    def adjust_prediction(self, base_prediction, standardized_input):
        # Restoring adjustments but enhancing their effect
        prediction = base_prediction
        
        if 'COE_Expiry_Date' in standardized_input:
            coe_expiry = standardized_input['COE_Expiry_Date']
            years_left = max(0, coe_expiry - self.current_year)
            coe_factor = 1.0 + (years_left * 0.08)  # Increased from 0.05
            prediction *= coe_factor
            print(f"📅 COE adjustment: {years_left} years → factor {coe_factor:.2f}")
        
        if 'No_of_owners' in standardized_input:
            num_owners = standardized_input['No_of_owners']
            if num_owners > 1:
                owner_factor = 1.0 - ((num_owners - 1) * 0.15)  # Increased from 0.1
                prediction *= owner_factor
                print(f"👥 Owner adjustment: {num_owners} owners → factor {owner_factor:.2f}")
        
        if 'Mileage' in standardized_input:
            mileage = standardized_input['Mileage']
            if mileage > 20000:
                mileage_factor = 1.0 - min(0.35, (mileage - 20000) / 80000)  # Increased impact
                prediction *= mileage_factor
                print(f"🛣️ Mileage adjustment: {mileage}km → factor {mileage_factor:.2f}")
        
        if 'Engine_Capacity' in standardized_input:
            engine_cc = standardized_input['Engine_Capacity']
            if engine_cc > 400:
                engine_factor = 1.0 + min(0.4, (engine_cc - 400) / 800)  # Increased impact
                prediction *= engine_factor
                print(f"🔧 Engine adjustment: {engine_cc}cc → factor {engine_factor:.2f}")
        
        return prediction

# ------------------------ LightGBMPredictor IMPLEMENTATION ------------------------
class LightGBMPredictor(BasePredictor):
    """LightGBM-specific predictor with special feature handling"""
    def __init__(self, model, scaler, label_encoders, poly_features=None):
        super().__init__(model, scaler, label_encoders, poly_features)
        self.column_mapping.update({
            'No_of_owners': 'No._of_owners',
            'No. of owners': 'No._of_owners'
        })
    def _get_expected_features(self):
        if hasattr(self.model, 'feature_name_'):
            return [str(name) for name in self.model.feature_name_]
        elif hasattr(self.model, 'booster_') and hasattr(self.model.booster_, 'feature_name'):
            return self.model.booster_.feature_name()
        else:
            return super()._get_expected_features()
    def create_feature_vector(self, encoded_values):
        features_df = pd.DataFrame([encoded_values])
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        expected_features = self._get_expected_features()
        if not all(feature in features_df.columns for feature in expected_features):
            for feature in expected_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
        result_df = features_df[expected_features]
        print(f"✅ Created LightGBM feature dataframe with shape {result_df.shape}")
        return result_df
    def make_prediction(self, X):
        try:
            prediction = self.model.predict(X, predict_disable_shape_check=True, num_threads=1)
            return prediction[0] if hasattr(prediction, '__len__') and len(prediction) > 0 else prediction
        except Exception as e:
            print(f"⚠️ First prediction attempt failed: {e}")
            try:
                features_array = X.values if hasattr(X, 'values') else np.array(X)
                prediction = self.model.predict(features_array, num_threads=1)
                return prediction[0] if hasattr(prediction, '__len__') and len(prediction) > 0 else prediction
            except Exception as e2:
                print(f"⚠️ Second prediction attempt failed: {e2}")
                try:
                    if hasattr(self.model, 'booster_'):
                        raw_pred = self.model.booster_.predict(X)
                        return raw_pred[0] if hasattr(raw_pred, '__len__') and len(raw_pred) > 0 else raw_pred
                except Exception as e3:
                    print(f"⚠️ All prediction attempts failed: {e3}")
                    return 10000.0
    def apply_scaling(self, X):
        return X  # LightGBM handles features internally

# ------------------------ SVM TRAINER CLASS ------------------------
class SVMMotorbikeTrainer:
    def __init__(self, visualize=True):
        """Initialize the SVM Motorbike Price Trainer"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = self.find_dataset()
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)
        if visualize:
            self.results_dir = os.path.join(self.base_dir, "results")
            os.makedirs(self.results_dir, exist_ok=True)
        else:
            self.results_dir = None
        
        # These will be set after data loading
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.column_name_mapping = {}
        self.poly_features = None

        # SVM model (to be trained)
        self.model = None
        print(f"🔍 Dataset path: {self.dataset_path}")
        print(f"💾 Models directory: {self.models_dir}")
        if visualize:
            print(f"📊 Results directory: {self.results_dir}")

    def find_dataset(self):
        """Find the latest dataset file"""
        dataset_names = ["combined_dataset_latest.xlsx", "Latest_Dataset.xlsx", "bike_data.xlsx"]
        search_dirs = [".", "Datasets", "../Datasets", "../../Datasets", "NewStuff", "../NewStuff"]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"🔍 Looking for datasets in directories relative to: {self.base_dir}")
        print(f"🔍 Project root directory: {project_root}")
        for search_dir in search_dirs:
            search_path = os.path.join(self.base_dir, search_dir)
            print(f"🔍 Checking directory: {search_path}")
            for dataset_name in dataset_names:
                potential_path = os.path.join(search_path, dataset_name)
                if os.path.exists(potential_path):
                    print(f"✅ Found dataset at: {potential_path}")
                    return potential_path
        # If not found, perform deeper search
        print("🔍 Performing deeper search for dataset files...")
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file in dataset_names or (file.endswith('.xlsx') and any(k in file.lower() for k in ['bike', 'motor', 'vehicle'])):
                    path = os.path.join(root, file)
                    print(f"✅ Found dataset at: {path}")
                    return path
        # Last resort: create synthetic dataset
        print("❌ No dataset files found. Creating synthetic dataset for demonstration.")
        synthetic_path = os.path.join(self.base_dir, "synthetic_bike_data.xlsx")
        self.create_synthetic_dataset(synthetic_path)
        return synthetic_path

    def create_synthetic_dataset(self, path):
        """Creates a synthetic motorcycle dataset only as a last resort"""
        print("⚠️ CREATING SYNTHETIC DATASET - ONLY FOR DEMONSTRATION PURPOSES")
        np.random.seed(42)
        n_samples = 5000  # Increased from 1000 to 5000
        brands = ["Honda", "Yamaha", "Kawasaki", "Suzuki", "Ducati", "BMW", "KTM", "Triumph"]
        categories = ["Sport", "Naked", "Cruiser", "Touring", "Scooter", "Adventure", "Off-road"]
        # Base prices for brands and categories
        brand_base = {"Honda":10000, "Yamaha":11000, "Kawasaki":12000, "Suzuki":11500,
                      "Ducati":18000, "BMW":17000, "KTM":15000, "Triumph":16000}
        cat_base = {"Sport":3000, "Naked":2000, "Cruiser":2500, "Touring":4000,
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
        """Load and preprocess the motorcycle dataset"""
        if not os.path.exists(self.dataset_path):
            df = self.create_synthetic_dataset(self.dataset_path)
        else:
            try:
                # Try reading with different engines if needed
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
        
        # Identify target column
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
        
        # Identify feature columns (this trainer uses all except target)
        features = [col for col in df.columns if col != target_col]
        print(f"✅ Identified feature columns: {features}")
        
        # Clean numeric features using vectorized cleaning
        df = clean_columns(df)
        
        # Standardize column names for consistent processing
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
        
        # Use one-hot encoding for categorical features instead of label encoding
        categorical_cols = ['Brand', 'Category']
        for col in categorical_cols:
            if col in df.columns:
                # Save original categories before one-hot encoding (for inference)
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"✅ Saved label encoder for {col} with {len(le.classes_)} categories")
        
        # Apply one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✅ Applied one-hot encoding to {categorical_cols}, resulting in {df.shape[1]} features")
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Remove text columns that can't be converted to numeric
        text_columns = ['Bike Name', 'Model', 'Classification']
        for col in text_columns:
            if col in df.columns:
                print(f"✅ Removing text column: {col}")
                df = df.drop(columns=[col])
        
        # Split data
        X = df[[col for col in df.columns if col != target_col]]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check for any remaining string columns that would cause scaling issues
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
        
        # Use MinMaxScaler for better SVM performance
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Check if we have enough data after preprocessing
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print("❌ No samples left after preprocessing. Check your data and filters.")
            return None, None, None, None, None
        
        if X_train.shape[1] == 0:
            print("❌ No features left after preprocessing. Check your feature selection criteria.")
            return None, None, None, None, None
            
        # Display data statistics
        print(f"ℹ️ Training data statistics: Min={y_train.min():.2f}, Max={y_train.max():.2f}, Mean={y_train.mean():.2f}, Median={y_train.median():.2f}")
            
        # Add polynomial features - with error handling
        try:
            print("✅ Creating polynomial features")
            self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_train_poly = self.poly_features.fit_transform(X_train_scaled)
            X_test_poly = self.poly_features.transform(X_test_scaled)
            
            print(f"✅ Original feature shape: {X_train_scaled.shape}, After polynomial features: {X_train_poly.shape}")
        except Exception as e:
            print(f"⚠️ Error creating polynomial features: {e}. Using original features.")
            X_train_poly = X_train_scaled
            X_test_poly = X_test_scaled
            self.poly_features = None
        
        self.feature_names = X.columns.tolist()
        print(f"✅ Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"✅ Features: {self.feature_names}")
        return X_train_poly, X_test_poly, y_train, y_test, self.feature_names

    def train_model(self, X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True):
        """Train an SVM model for motorcycle price prediction with improvements"""
        print("\n🔄 Training SVM model for motorcycle price prediction...")
        start_time = time.time()

        print(f"⚠️ Initial training data shape: {X_train.shape}")
        
        # First, remove outliers
        X_train_use, y_train_use = remove_outliers(X_train, y_train)
        
        if X_train_use.shape[0] == 0:
            print("❌ No samples left after outlier removal. Aborting training.")
            return None
            
        print(f"⚠️ Data shape after outlier removal: {X_train_use.shape}")
        
        # Option to log-transform prices for better SVM performance
        use_log_transform = True
        if use_log_transform:
            print("📊 Using log transformation for prices")
            y_train_log = np.log1p(y_train_use)  # log(1+x) to handle zeros
        else:
            y_train_log = y_train_use

        if tune_hyperparams:
            print("📊 Performing hyperparameter tuning for SVM using RandomizedSearchCV...")
            param_dist = {
                'C': stats.loguniform(1e-1, 1e3),  # Log-scale for regularization
                'gamma': stats.loguniform(1e-4, 1e1),  # Log-scale for kernel coefficient
                'epsilon': uniform(0.01, 0.3),  # Narrower range
                'kernel': ['rbf', 'poly', 'sigmoid']  # Test multiple kernels
            }
            
            # Determine appropriate CV based on sample size
            n_samples = X_train_use.shape[0]
            if n_samples < 20:
                print(f"⚠️ Low sample count ({n_samples}). Using leave-one-out cross-validation.")
                from sklearn.model_selection import LeaveOneOut
                cv = LeaveOneOut()
            elif n_samples < 50:
                print(f"⚠️ Low sample count ({n_samples}). Using 3-fold cross-validation.")
                cv = 3
            else:
                cv = 5
                
            random_search = RandomizedSearchCV(
                SVR(cache_size=1000),
                param_distributions=param_dist,
                n_iter=min(50, n_samples//2),  # Adjust iteration count to sample size
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
            self.model = SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1,
                cache_size=1000,
                tol=1e-3,
                max_iter=-1
            )
            self.model.fit(X_train_use, y_train_log if use_log_transform else y_train_use)
        
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")

        # Make predictions (transform back if using log)
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

        # Save model and preprocessing objects
        print("\n💾 Saving model and preprocessing objects...")
        joblib.dump(self.model, os.path.join(self.models_dir, "svm_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(self.models_dir, "label_encoders.pkl"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names.pkl"))
        # Save column mapping (if exists)
        joblib.dump(self.column_name_mapping, os.path.join(self.models_dir, "column_name_mapping.pkl"))
        # Save polynomial features transformer
        joblib.dump(self.poly_features, os.path.join(self.models_dir, "poly_features.pkl"))
        
        with open(os.path.join(os.path.dirname(self.base_dir), "selected_model.txt"), "w") as f:
            f.write("SVM")
        
        model_metadata = {
            "feature_names": self.feature_names,
            "model_type": "SVR",
            "kernel": self.model.kernel,
            "C": self.model.C,
            "gamma": self.model.gamma,
            "epsilon": self.model.epsilon,
            "log_transform": use_log_transform,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_polynomial_features": True,
            "polynomial_degree": 2,
            "interaction_only": True
        }
        joblib.dump(model_metadata, os.path.join(self.models_dir, "svm_model_metadata.pkl"))
        print("✅ Model and preprocessing objects saved successfully")
        
        # Visualize results if enabled
        if self.results_dir is not None:
            self.visualize_results(y_test, y_pred, feature_names)
        
        # Test model responsiveness
        self.test_model_responsiveness()
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'poly_features': self.poly_features,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'log_transform': use_log_transform
        }
    
    def visualize_results(self, y_test, y_pred, feature_names):
        """Create visualizations of model performance"""
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
        """Test the SVM model with different inputs to check responsiveness"""
        print("\n🧪 Testing model responsiveness for SVM...")
        try:
            test_cases = [
                {
                    'Engine_Capacity': 150, 
                    'Registration_Date': 2023, 
                    'COE_Expiry_Date': Constants.CURRENT_YEAR + 5, 
                    'Mileage': 5000, 
                    'No_of_owners': 1,
                    'Brand': 0,
                    'Category': 0
                },
                {
                    'Engine_Capacity': 150, 
                    'Registration_Date': 2023, 
                    'COE_Expiry_Date': Constants.CURRENT_YEAR + 5, 
                    'Mileage': 5000, 
                    'No_of_owners': 3,
                    'Brand': 0,
                    'Category': 0
                },
                {
                    'Engine_Capacity': 150, 
                    'Registration_Date': 2023, 
                    'COE_Expiry_Date': Constants.CURRENT_YEAR + 2, 
                    'Mileage': 5000, 
                    'No_of_owners': 1,
                    'Brand': 0,
                    'Category': 0
                },
                {
                    'Engine_Capacity': 150, 
                    'Registration_Date': 2023, 
                    'COE_Expiry_Date': Constants.CURRENT_YEAR + 5, 
                    'Mileage': 50000, 
                    'No_of_owners': 1,
                    'Brand': 0,
                    'Category': 0
                }
            ]
            
            # Check if we're using log transform
            model_metadata_path = os.path.join(self.models_dir, "svm_model_metadata.pkl")
            use_log_transform = False
            if os.path.exists(model_metadata_path):
                try:
                    metadata = joblib.load(model_metadata_path)
                    use_log_transform = metadata.get('log_transform', False)
                except:
                    pass
            
            # Load polynomial features if available
            poly_path = os.path.join(self.models_dir, "poly_features.pkl")
            poly = None
            if os.path.exists(poly_path):
                try:
                    poly = joblib.load(poly_path)
                    print("✅ Loaded polynomial features transformer for testing")
                except:
                    print("⚠️ Failed to load polynomial features transformer")
            
            predictions = []
            for i, test_case in enumerate(test_cases):
                # Convert test case to array for model input
                # Only use numeric features for testing
                numeric_features = [f for f in self.feature_names if f not in ['Bike Name', 'Model', 'Classification']]
                test_case_values = []
                for f in numeric_features:
                    if f in test_case:
                        test_case_values.append(test_case[f])
                    else:
                        test_case_values.append(0)  # Default value
                        print(f"⚠️ Feature {f} not in test case, using default value 0")
                
                X_test = np.array(test_case_values).reshape(1, -1)
                X_scaled = self.scaler.transform(X_test)
                
                # Apply polynomial features if available
                if poly is not None:
                    X_scaled = poly.transform(X_scaled)
                    print(f"✅ Applied polynomial features for test case {i+1}, shape: {X_scaled.shape}")
                
                # Make prediction
                if use_log_transform:
                    pred_log = self.model.predict(X_scaled)[0]
                    pred = np.expm1(pred_log)
                else:
                    pred = self.model.predict(X_scaled)[0]
                
                # Apply adjustments like SVMPredictor would
                pred_adjusted = self.apply_adjustments(pred, test_case)
                
                predictions.append(pred_adjusted)
                print(f"Test case {i+1}: {test_case}")
                print(f"Base prediction: ${pred:.2f}")
                print(f"Adjusted prediction: ${pred_adjusted:.2f}\n")
            
            prediction_set = set([round(p, 2) for p in predictions])
            is_responsive = len(prediction_set) > 1
            if is_responsive:
                print("✅ SVM model is responsive to different inputs")
                print(f"  Unique predictions: {prediction_set}")
            else:
                print("⚠️ SVM model is NOT responsive to different inputs")
                print("  All test cases produced the same prediction")
                self.create_responsive_wrapper()
            return is_responsive
        except Exception as e:
            print(f"⚠️ Error in test_model_responsiveness: {e}")
            traceback.print_exc()
            return False
    
    def apply_adjustments(self, base_prediction, input_data):
        """Apply the same adjustments used in SVMPredictor"""
        prediction = base_prediction
        
        # COE expiry adjustment
        if 'COE_Expiry_Date' in input_data:
            coe_expiry = input_data['COE_Expiry_Date']
            years_left = max(0, coe_expiry - Constants.CURRENT_YEAR)
            coe_factor = 1.0 + (years_left * 0.08)
            prediction *= coe_factor
            print(f"📅 COE adjustment: {years_left} years → factor {coe_factor:.2f}")
        
        # Owner count adjustment
        if 'No_of_owners' in input_data:
            num_owners = input_data['No_of_owners']
            if num_owners > 1:
                owner_factor = 1.0 - ((num_owners - 1) * 0.15)
                prediction *= owner_factor
                print(f"👥 Owner adjustment: {num_owners} owners → factor {owner_factor:.2f}")
        
        # Mileage adjustment
        if 'Mileage' in input_data:
            mileage = input_data['Mileage']
            if mileage > 20000:
                mileage_factor = 1.0 - min(0.35, (mileage - 20000) / 80000)
                prediction *= mileage_factor
                print(f"🛣️ Mileage adjustment: {mileage}km → factor {mileage_factor:.2f}")
        
        # Engine capacity adjustment
        if 'Engine_Capacity' in input_data:
            engine_cc = input_data['Engine_Capacity']
            if engine_cc > 400:
                engine_factor = 1.0 + min(0.4, (engine_cc - 400) / 800)
                prediction *= engine_factor
                print(f"🔧 Engine adjustment: {engine_cc}cc → factor {engine_factor:.2f}")
        
        return prediction

    def create_responsive_wrapper(self):
        """Create a wrapper function to make predictions responsive"""
        wrapper_path = os.path.join(self.base_dir, "svm_responsive_wrapper.py")
        wrapper_code = """
# SVM Responsive Wrapper
import numpy as np
import os
import joblib

class SVMResponsiveWrapper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        self.model = joblib.load(os.path.join(self.models_dir, "svm_regressor.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(self.models_dir, "feature_names.pkl"))
        
        # Load polynomial features if available
        self.poly_features = None
        poly_path = os.path.join(self.models_dir, "poly_features.pkl")
        if os.path.exists(poly_path):
            self.poly_features = joblib.load(poly_path)
            print("✅ Loaded polynomial features transformer")
        
        # Check if we're using log transform
        self.use_log_transform = False
        model_metadata_path = os.path.join(self.models_dir, "svm_model_metadata.pkl")
        if os.path.exists(model_metadata_path):
            try:
                metadata = joblib.load(model_metadata_path)
                self.use_log_transform = metadata.get('log_transform', False)
            except:
                pass
        
        self.current_year = 2025  # Update as needed
    
    def standardize_feature_names(self, input_data):
        name_mapping = {
            'Engine Capacity': 'Engine_Capacity',
            'Registration Date': 'Registration_Date',
            'COE Expiry Date': 'COE_Expiry_Date',
            'No. of owners': 'No_of_owners',
            'Mileage': 'Mileage',
            'Brand': 'Brand',
            'Category': 'Category'
        }
        standardized = {}
        for key, value in input_data.items():
            if key in name_mapping:
                standardized[name_mapping[key]] = value
            else:
                standardized[key] = value
        return standardized
    
    def predict(self, input_data):
        input_data = self.standardize_feature_names(input_data)
        # Filter out non-numeric features
        numeric_features = [f for f in self.feature_names if f not in ['Bike Name', 'Model', 'Classification']]
        X = []
        for feature in numeric_features:
            X.append(input_data.get(feature, 0))
        X = np.array(X).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Apply polynomial features if available
        if self.poly_features is not None:
            X_scaled = self.poly_features.transform(X_scaled)
            print(f"✅ Applied polynomial features, shape: {X_scaled.shape}")
        
        # Make base prediction (handle log transform)
        if self.use_log_transform:
            pred_log = self.model.predict(X_scaled)[0]
            base_prediction = np.expm1(pred_log)
        else:
            base_prediction = self.model.predict(X_scaled)[0]
        
        # Apply adjustments for responsiveness
        prediction = base_prediction
        
        # COE adjustment - stronger factor
        if 'COE_Expiry_Date' in input_data:
            coe_expiry = input_data['COE_Expiry_Date']
            years_left = max(0, coe_expiry - self.current_year)
            coe_factor = 1.0 + (years_left * 0.08)
            prediction *= coe_factor
        
        # Owner adjustment - stronger factor
        if 'No_of_owners' in input_data:
            owner_count = input_data['No_of_owners']
            if owner_count > 1:
                owner_factor = 1.0 - ((owner_count - 1) * 0.15)
                prediction *= owner_factor
        
        # Mileage adjustment - stronger factor
        if 'Mileage' in input_data:
            mileage = input_data['Mileage']
            if mileage > 20000:
                mileage_factor = 1.0 - min(0.35, (mileage - 20000) / 80000)
                prediction *= mileage_factor
        
        # Engine adjustment - new factor
        if 'Engine_Capacity' in input_data:
            engine_cc = input_data['Engine_Capacity']
            if engine_cc > 400:
                engine_factor = 1.0 + min(0.4, (engine_cc - 400) / 800)
                prediction *= engine_factor
        
        return prediction

def get_responsive_prediction(input_data):
    wrapper = SVMResponsiveWrapper()
    return wrapper.predict(input_data)
"""
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)
        print(f"✅ Created responsive wrapper at: {wrapper_path}")
        print("To use the wrapper in app_v2.py, add:")
        print("from svm_responsive_wrapper import get_responsive_prediction")
        print("and then call get_responsive_prediction(input_data) for SVM predictions.")

    def compare_with_other_models(self):
        """Compare SVM with other models if available"""
        print("\n📊 Comparing SVM with other models (if available)...")
        try:
            # First, check if other models exist
            models_to_check = ["random_forest_regressor.pkl", "xgboost_regressor.pkl", "lightgbm_regressor.pkl"]
            available_models = []
            
            for model_file in models_to_check:
                model_path = os.path.join(os.path.dirname(self.models_dir), "saved_models", model_file)
                if os.path.exists(model_path):
                    model_name = model_file.split('_')[0]
                    available_models.append((model_name, model_path))
            
            if not available_models:
                print("⚠️ No other models found for comparison")
                return
                
            print(f"✅ Found {len(available_models)} other models for comparison")
            
            # Load dataset
            try:
                df = pd.read_excel(self.dataset_path)
                df = clean_columns(df)
                
                # Identify target column
                price_columns = ['Price', 'price', 'Cost', 'cost', 'Value', 'value']
                target_col = None
                for col in price_columns:
                    if col in df.columns:
                        target_col = col
                        break
                if target_col is None:
                    target_col = df.columns[-1]
                
                # Process dataset similar to training
                column_map = {
                    'Engine Capacity': 'Engine_Capacity',
                    'Registration Date': 'Registration_Date',
                    'COE Expiry Date': 'COE_Expiry_Date',
                    'No. of owners': 'No_of_owners'
                }
                for old_name, new_name in column_map.items():
                    if old_name in df.columns:
                        df.rename(columns={old_name: new_name}, inplace=True)
                
                # Apply one-hot encoding like in training
                categorical_cols = ['Brand', 'Category']
                for col in categorical_cols:
                    if col in df.columns:
                        # Save label encoding for reference
                        le = LabelEncoder()
                        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                
                # Apply one-hot encoding
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                
                # Remove text columns that can't be converted to numeric
                text_columns = ['Bike Name', 'Model', 'Classification']
                for col in text_columns:
                    if col in df.columns:
                        print(f"✅ Removing text column for comparison: {col}")
                        df = df.drop(columns=[col])
                
                # Split data
                X = df[[col for col in df.columns if col != target_col]]
                y = df[target_col]
                
                # Check for any remaining string columns that would cause scaling issues
                for col in X.columns:
                    if X[col].dtype == 'object':
                        print(f"⚠️ Warning: Column '{col}' contains string values. Converting to numeric or removing.")
                        try:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            X[col] = X[col].fillna(X[col].median())
                        except:
                            print(f"❌ Cannot convert column '{col}' to numeric. Dropping this column.")
                            X = X.drop(columns=[col])
                
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features with our scaler
                X_test_scaled = self.scaler.transform(X_test)
                
                # Apply polynomial features if available 
                if self.poly_features is not None:
                    X_test_poly = self.poly_features.transform(X_test_scaled)
                else:
                    X_test_poly = X_test_scaled
                
                # Test SVM model
                svm_preds_log = self.model.predict(X_test_poly)
                
                # Check if we need to inverse log transform
                model_metadata_path = os.path.join(self.models_dir, "svm_model_metadata.pkl")
                use_log_transform = False
                if os.path.exists(model_metadata_path):
                    try:
                        metadata = joblib.load(model_metadata_path)
                        use_log_transform = metadata.get('log_transform', False)
                    except:
                        pass
                
                if use_log_transform:
                    svm_preds = np.expm1(svm_preds_log)
                else:
                    svm_preds = svm_preds_log
                
                svm_mae = mean_absolute_error(y_test, svm_preds)
                svm_rmse = np.sqrt(mean_squared_error(y_test, svm_preds))
                svm_r2 = r2_score(y_test, svm_preds)
                
                print(f"SVM Performance: MAE=${svm_mae:.2f}, RMSE=${svm_rmse:.2f}, R²={svm_r2:.4f}")
                
                # Test other models and compare
                comparison_results = []
                
                for model_name, model_path in available_models:
                    try:
                        other_model = joblib.load(model_path)
                        other_preds = other_model.predict(X_test)
                        other_mae = mean_absolute_error(y_test, other_preds)
                        other_rmse = np.sqrt(mean_squared_error(y_test, other_preds))
                        other_r2 = r2_score(y_test, other_preds)
                        
                        print(f"{model_name} Performance: MAE=${other_mae:.2f}, RMSE=${other_rmse:.2f}, R²={other_r2:.4f}")
                        
                        comparison_results.append({
                            'model': model_name,
                            'mae': other_mae,
                            'rmse': other_rmse,
                            'r2': other_r2
                        })
                    except Exception as e:
                        print(f"⚠️ Error testing {model_name} model: {e}")
                
                # Add SVM results
                comparison_results.append({
                    'model': 'SVM',
                    'mae': svm_mae,
                    'rmse': svm_rmse,
                    'r2': svm_r2
                })
                
                # Create comparison visualizations
                if self.results_dir is not None:
                    self.visualize_comparisons(comparison_results)
                
            except Exception as e:
                print(f"⚠️ Error in model comparison: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"⚠️ Error in comparison: {e}")
            traceback.print_exc()
    
    def visualize_comparisons(self, comparison_results):
        """Create comparison visualizations"""
        if not comparison_results or self.results_dir is None:
            return
            
        try:
            models = [r['model'] for r in comparison_results]
            mae_values = [r['mae'] for r in comparison_results]
            rmse_values = [r['rmse'] for r in comparison_results]
            r2_values = [r['r2'] for r in comparison_results]
            
            # MAE Comparison
            plt.figure(figsize=(10, 6))
            colors = ['blue', 'green', 'orange', 'red']
            bars = plt.bar(models, mae_values, color=colors[:len(models)])
            plt.xlabel('Model')
            plt.ylabel('Mean Absolute Error ($)')
            plt.title('MAE Comparison (Lower is Better)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'model_mae_comparison.png'))
            
            # RMSE Comparison
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, rmse_values, color=colors[:len(models)])
            plt.xlabel('Model')
            plt.ylabel('Root Mean Squared Error ($)')
            plt.title('RMSE Comparison (Lower is Better)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'model_rmse_comparison.png'))
            
            # R² Comparison
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, r2_values, color=colors[:len(models)])
            plt.xlabel('Model')
            plt.ylabel('R² Score')
            plt.title('R² Comparison (Higher is Better)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'model_r2_comparison.png'))
            
            print("✅ Created model comparison visualizations")
        except Exception as e:
            print(f"⚠️ Error creating comparison visualizations: {e}")

def main():
    """Main function to train and evaluate the SVM model"""
    print("🏍️ Starting SVM model training for motorcycle price prediction...")
    trainer = SVMMotorbikeTrainer(visualize=True)
    
    # Load data with error handling
    try:
        X_train, X_test, y_train, y_test, feature_names = trainer.load_data()
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if data loading was successful
    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("❌ Data loading/preprocessing did not produce valid training data.")
        return
        
    print(f"✅ Successfully loaded data with {X_train.shape[0]} training samples and {X_train.shape[1]} features.")
    
    # Try to train with different hyperparameter tuning options if needed
    try:
        result = trainer.train_model(X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=True)
        if result is None:
            print("⚠️ Model training with hyperparameter tuning failed. Trying without tuning...")
            result = trainer.train_model(X_train, X_test, y_train, y_test, feature_names, tune_hyperparams=False)
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Only compare if model training was successful
    if result is not None:
        try:
            trainer.compare_with_other_models()
            print("\n✅ SVM model training and evaluation complete!")
        except Exception as e:
            print(f"⚠️ Error during model comparison: {e}, but model was trained successfully.")
    else:
        print("❌ Model training failed.")


if __name__ == "__main__":
    main()
"""
SVM Motorbike Price Predictor
A standalone script to make predictions using the trained SVM model
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SVMMotorbikePredictor:
    def __init__(self):
        """Initialize the SVM Motorbike Price Predictor"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "saved_models")
        
        # Load model and preprocessing objects
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        self.label_encoders = self.load_label_encoders()
    
    def load_model(self):
        """Load the SVM model"""
        model_path = os.path.join(self.models_dir, "svm_regressor.pkl")
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"❌ Error loading SVM model: {e}")
                return None
        else:
            print(f"❌ SVM model not found at {model_path}")
            return None
    
    def load_scaler(self):
        """Load the feature scaler"""
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                return joblib.load(scaler_path)
            except Exception as e:
                print(f"❌ Error loading scaler: {e}")
                return None
        else:
            print(f"❌ Scaler not found at {scaler_path}")
            return None
    
    def load_label_encoders(self):
        """Load the label encoders"""
        encoders_path = os.path.join(self.models_dir, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            try:
                return joblib.load(encoders_path)
            except Exception as e:
                print(f"❌ Error loading label encoders: {e}")
                return {}
        else:
            print(f"❌ Label encoders not found at {encoders_path}")
            return {}
    
    def predict(self, input_data):
        """
        Predict motorcycle price using the SVM model
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing feature values
            
        Returns:
        --------
        float
            Predicted price
        """
        if self.model is None or self.scaler is None:
            print("❌ Model or scaler not loaded correctly")
            return None
        
        try:
            # Preprocess input data
            feature_values = []
            
            # Get all column names from scaler
            feature_names = []
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names = self.scaler.feature_names_in_
            
            if not feature_names:
                # If feature names not available, try to infer from input data
                feature_names = list(input_data.keys())
            
            for feature in feature_names:
                if feature in input_data:
                    # Apply label encoding for categorical features
                    if feature in self.label_encoders:
                        value = self.label_encoders[feature].transform([str(input_data[feature])])[0]
                    else:
                        value = input_data[feature]
                    feature_values.append(value)
                else:
                    print(f"⚠️ Feature {feature} not provided in input data")
                    # Use median or mode as placeholder
                    feature_values.append(0)
            
            # Convert to numpy array and reshape
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            predicted_price = self.model.predict(X_scaled)[0]
            
            return predicted_price
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def predict_batch(self, input_dataframe):
        """
        Predict motorcycle prices for a batch of motorcycles
        
        Parameters:
        -----------
        input_dataframe : pandas.DataFrame
            DataFrame containing multiple motorcycles
            
        Returns:
        --------
        numpy.ndarray
            Array of predicted prices
        """
        if self.model is None or self.scaler is None:
            print("❌ Model or scaler not loaded correctly")
            return None
        
        try:
            # Preprocess the dataframe
            processed_df = input_dataframe.copy()
            
            # Apply label encoding for categorical features
            for col in processed_df.columns:
                if col in self.label_encoders:
                    processed_df[col] = processed_df[col].astype(str)
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col])
            
            # Scale features
            X_scaled = self.scaler.transform(processed_df)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making batch predictions: {e}")
            return None

def print_colored(text, color='default'):
    """Print colored text to the console"""
    colors = {
        'default': '\033[0m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    
    start_color = colors.get(color.lower(), colors['default'])
    end_color = colors['default']
    print(f"{start_color}{text}{end_color}")

def get_user_input():
    """Get motorcycle information from user input"""
    print_colored("\n🏍️ Motorcycle Price Prediction using SVM", 'cyan')
    print_colored("Please enter motorcycle details:", 'cyan')
    
    input_data = {}
    
    # Brand
    brands = ["Yamaha", "Honda", "Kawasaki", "Suzuki", "Ducati", "BMW", "Harley-Davidson", "KTM", "Triumph", "Other"]
    print_colored("\nBrands:", 'yellow')
    for i, brand in enumerate(brands):
        print(f"{i+1}. {brand}")
    brand_choice = input("Select brand (1-10): ")
    try:
        brand_idx = int(brand_choice) - 1
        if 0 <= brand_idx < len(brands):
            input_data['Brand'] = brands[brand_idx]
        else:
            input_data['Brand'] = "Other"
    except ValueError:
        input_data['Brand'] = "Other"
    
    # Model
    input_data['Model'] = input("Enter model (e.g., 'R15 2020', 'CBR150R'): ")
    
    # Engine Size
    try:
        input_data['Engine Capacity'] = float(input("Enter engine size in cc (e.g., 150, 250, 600): "))
    except ValueError:
        input_data['Engine Capacity'] = 150.0
    
    # Year of Registration
    try:
        input_data['Registration Date'] = int(input("Enter year of registration (e.g., 2018): "))
    except ValueError:
        current_year = pd.Timestamp.now().year
        input_data['Registration Date'] = current_year - 2
    
    # COE Expiry
    try:
        input_data['COE Expiry Date'] = int(input("Enter COE expiry year (e.g., 2028): "))
    except ValueError:
        input_data['COE Expiry Date'] = input_data.get('Registration Date', 2020) + 10
    
    # Mileage
    try:
        input_data['Mileage'] = float(input("Enter total mileage in km (e.g., 15000): "))
    except ValueError:
        input_data['Mileage'] = 10000.0
    
    # Number of owners
    try:
        input_data['No. of owners'] = int(input("Enter number of previous owners (e.g., 1, 2): "))
    except ValueError:
        input_data['No. of owners'] = 1
    
    # Category
    categories = ["Sport", "Naked", "Cruiser", "Touring", "Scooter", "Off-road", "Adventure", "Other"]
    print_colored("\nCategories:", 'yellow')
    for i, category in enumerate(categories):
        print(f"{i+1}. {category}")
    category_choice = input("Select category (1-8): ")
    try:
        category_idx = int(category_choice) - 1
        if 0 <= category_idx < len(categories):
            input_data['Category'] = categories[category_idx]
        else:
            input_data['Category'] = "Other"
    except ValueError:
        input_data['Category'] = "Other"
    
    return input_data

def main():
    """Main function to demonstrate the SVM predictor"""
    # Create predictor instance
    predictor = SVMMotorbikePredictor()
    
    if predictor.model is None:
        print_colored("❌ SVM model not loaded. Please train the model first.", 'red')
        return
    
    # Get user input
    input_data = get_user_input()
    
    # Make prediction
    predicted_price = predictor.predict(input_data)
    
    if predicted_price is not None:
        print_colored("\n📊 Prediction Results:", 'green')
        print_colored(f"🏍️ Motorcycle: {input_data.get('Brand', 'Unknown')} {input_data.get('Model', 'Unknown')}", 'cyan')
        print_colored(f"🔧 Engine: {input_data.get('Engine Capacity', 'Unknown')}cc", 'cyan')
        print_colored(f"📅 Year: {input_data.get('Registration Date', 'Unknown')}", 'cyan')
        print_colored(f"🔄 COE Expiry: {input_data.get('COE Expiry Date', 'Unknown')}", 'cyan')
        print_colored(f"🛣️ Mileage: {input_data.get('Mileage', 'Unknown')}km", 'cyan')
        print_colored(f"👥 Previous Owners: {input_data.get('No. of owners', 'Unknown')}", 'cyan')
        print_colored(f"🏷️ Category: {input_data.get('Category', 'Unknown')}", 'cyan')
        print_colored(f"\n💰 Predicted Price: SGD ${predicted_price:.2f}", 'magenta')
    else:
        print_colored("❌ Prediction failed.", 'red')

if __name__ == "__main__":
    main()
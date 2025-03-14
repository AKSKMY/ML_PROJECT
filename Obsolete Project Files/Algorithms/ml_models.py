import numpy as np
import joblib
import os
import sys
import importlib

class TrainPredictor:
    def __init__(self):
        # Get the absolute path to the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for required dependencies
        self.check_dependencies()
        
        # Load vectorizer & label encoder using absolute paths
        self.vectorizer = joblib.load(os.path.join(current_dir, "tfidf_vectorizer.pkl"))
        self.label_encoder = joblib.load(os.path.join(current_dir, "label_encoder.pkl"))

        # Load all models with error handling
        self.models = {}
        self.available_models = []
        
        # Try to load Random Forest models (always available in scikit-learn)
        try:
            self.models["Random Forest"] = {
                "reg": joblib.load(os.path.join(current_dir, "best_random_forest_regressor.pkl")),
                "cls": joblib.load(os.path.join(current_dir, "best_random_forest_classifier.pkl"))
            }
            self.available_models.append("Random Forest")
            print("✅ Random Forest models loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading Random Forest models: {e}")
        
        # Try to load XGBoost models
        if self.is_module_available("xgboost"):
            try:
                self.models["XGBoost"] = {
                    "reg": joblib.load(os.path.join(current_dir, "best_xgboost_regressor.pkl")),
                    "cls": joblib.load(os.path.join(current_dir, "best_xgboost_classifier.pkl"))
                }
                if os.system("nvidia-smi") == 0:
                    print("GPU is available!")
                else:
                    print("No GPU detected.")
                self.available_models.append("XGBoost")
                print("✅ XGBoost models loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading XGBoost models: {e}")
        else:
            print("⚠️ XGBoost module not available. Install with: pip install xgboost")
        
        # Try to load LightGBM models
        if self.is_module_available("lightgbm"):
            try:
                self.models["LightGBM"] = {
                    "reg": joblib.load(os.path.join(current_dir, "best_lightgbm_regressor.pkl")),
                    "cls": joblib.load(os.path.join(current_dir, "best_lightgbm_classifier.pkl"))
                }
                self.available_models.append("LightGBM")
                print("✅ LightGBM models loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading LightGBM models: {e}")
        else:
            print("⚠️ LightGBM module not available. Install with: pip install lightgbm")
        
        # Try to load SVM models
        try:
            self.models["SVM"] = {
                "reg": joblib.load(os.path.join(current_dir, "best_svm_regressor.pkl")),
                "cls": joblib.load(os.path.join(current_dir, "best_svm_classifier.pkl"))
            }
            self.available_models.append("SVM")
            print("✅ SVM models loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading SVM models: {e}")

        # Default model selection - choose first available model
        self.selected_model = self.available_models[0] if self.available_models else None
        
        # Check if any models were loaded
        if not self.available_models:
            print("❌ No models could be loaded. Please check dependencies and model files.")
            return
            
        # Use absolute path for model file
        self.model_file = os.path.join(current_dir, "selected_model.txt")
        self.load_selected_model()

    def check_dependencies(self):
        """Check if required dependencies are available"""
        dependencies = {
            "numpy": "numpy",
            "scikit-learn": "sklearn",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm"
        }
        
        missing = []
        for name, module in dependencies.items():
            if not self.is_module_available(module):
                missing.append(name)
                
        if missing:
            print("⚠️ Missing dependencies: " + ", ".join(missing))
            print("Please install them using:")
            for pkg in missing:
                print(f"pip install {pkg}")
        else:
            print("✅ All dependencies available")

    def is_module_available(self, module_name):
        """Check if a module is available for import"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def load_selected_model(self):
        """ Load the selected model from file"""
        if self.selected_model is None:
            print("❌ No models available to load.")
            return
            
        # Try to load selected model from file
        if os.path.exists(self.model_file):
            with open(self.model_file, "r") as file:
                model_name = file.read().strip()
                if model_name in self.models:
                    self.selected_model = model_name
                else:
                    print(f"⚠️ Model '{model_name}' not available. Defaulting to {self.selected_model}")
        else:
            print(f"⚠️ Model selection file not found. Defaulting to {self.selected_model}")

        # Load the correct models
        self.best_reg = self.models[self.selected_model]["reg"]
        self.best_cls = self.models[self.selected_model]["cls"]

        print(f"✅ Using Selected Model: {self.selected_model}")
        print(f"📊 Available models: {', '.join(self.available_models)}")

    def predict(self, train_alert, feature1, feature2):
        """ Predict train delay and severity based on user input """
        if self.selected_model is None:
            return 0, "No models available"
            
        X_text = self.vectorizer.transform([train_alert]).toarray()
        X_numeric = np.array([[feature1, feature2]])
        X_input = np.hstack((X_text, X_numeric))

        predicted_delay = self.best_reg.predict(X_input)[0]
        predicted_severity = self.best_cls.predict(X_input)[0]
        severity_label = "High Delay" if predicted_severity == 1 else "Low Delay"

        return predicted_delay, severity_label
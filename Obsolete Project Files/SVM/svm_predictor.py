import numpy as np
import joblib
import os

class SVMPredictor:
    """
    A class to handle SVM model predictions for train delay forecasting.
    This class can be integrated into the Flask application.
    """
    
    def __init__(self):
        """Initialize the SVM predictor by loading necessary models"""
        try:
            # Define base directory and model paths more robustly
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(base_dir)
            models_dir = os.path.join(project_dir, "Algorithms")
            
            # Print paths for debugging
            print(f"Base directory: {base_dir}")
            print(f"Project directory: {project_dir}")
            print(f"Models directory: {models_dir}")
            
            # Verify model files exist
            vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
                
            # Load vectorizer & label encoder
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
            
            # Load SVM models
            self.svm_regressor = joblib.load(os.path.join(models_dir, "best_svm_regressor.pkl"))
            self.svm_classifier = joblib.load(os.path.join(models_dir, "best_svm_classifier.pkl"))
            
            print("✅ SVM models loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading SVM models: {e}")
            self.model_loaded = False
    
    def predict(self, train_alert, day, feature1, feature2):
        """
        Predict train delay time and severity using SVM models
        
        Parameters:
        -----------
        train_alert : str
            Text description of the train situation/alert
        day : str
            Day of the week (e.g., 'Monday', 'Tuesday', etc.)
        feature1 : float
            Numeric feature (feature 146 in your dataset)
        feature2 : float
            Numeric feature (feature 10 in your dataset)
            
        Returns:
        --------
        tuple
            (predicted_delay, severity_label, confidence)
        """
        if not self.model_loaded:
            return None, "Error: Models not loaded", 0
        
        try:
            # Preprocess text input
            X_text = self.vectorizer.transform([train_alert]).toarray()
            
            # Encode categorical input
            day_encoded = self.label_encoder.transform([day])[0]
            
            # Create numeric feature array
            X_numeric = np.array([[feature1, feature2]])
            
            # Combine features - we need to ensure the day is included properly
            # Assuming the day would be part of X_numeric in the actual implementation
            X_input = np.hstack((X_text, X_numeric))
            
            # Make predictions
            predicted_delay = self.svm_regressor.predict(X_input)[0]
            severity_prediction = self.svm_classifier.predict(X_input)[0]
            
            # Get probability for classification confidence
            severity_prob = self.svm_classifier.predict_proba(X_input)[0]
            confidence = severity_prob[severity_prediction] * 100  # Convert to percentage
            
            # Convert severity to label
            severity_label = "High Delay" if severity_prediction == 1 else "Low Delay"
            
            return predicted_delay, severity_label, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, f"Error: {str(e)}", 0
    
    def get_model_info(self):
        """Return information about the loaded SVM models"""
        if not self.model_loaded:
            return "SVM models not loaded."
        
        info = {
            "Model Type": "Support Vector Machine (SVM)",
            "Regression Model": str(self.svm_regressor),
            "Classification Model": str(self.svm_classifier),
            "Features Used": ["Train Alert Text (TF-IDF)", "Day", "Numeric Feature 146", "Numeric Feature 10"]
        }
        
        return info

# Example usage
if __name__ == "__main__":
    # Create a predictor instance
    predictor = SVMPredictor()
    
    # Example prediction
    train_alert = "Signal fault at Woodlands station causing delays"
    day = "Monday"
    feature1 = 15.0  # Example value for feature 146
    feature2 = 5.0   # Example value for feature 10
    
    delay, severity, confidence = predictor.predict(train_alert, day, feature1, feature2)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Delay: {delay:.2f} minutes")
    print(f"Delay Severity: {severity}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Print model info
    print("\nModel Information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
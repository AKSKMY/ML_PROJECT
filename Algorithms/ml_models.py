import numpy as np
import joblib
import os

class TrainPredictor:
    def __init__(self):
        # Load vectorizer & label encoder
        self.vectorizer = joblib.load("../Algorithms/tfidf_vectorizer.pkl")
        self.label_encoder = joblib.load("../Algorithms/label_encoder.pkl")

        # Load all models with correct filenames
        self.models = {
            "Random Forest": {
                "reg": joblib.load("../Algorithms/best_random_forest_regressor.pkl"),
                "cls": joblib.load("../Algorithms/best_random_forest_classifier.pkl")
            },
            "XGBoost": {
                "reg": joblib.load("../Algorithms/best_xgboost_regressor.pkl"),
                "cls": joblib.load("../Algorithms/best_xgboost_classifier.pkl")
            },
            "LightGBM": {
                "reg": joblib.load("../Algorithms/best_lightgbm_regressor.pkl"),
                "cls": joblib.load("../Algorithms/best_lightgbm_classifier.pkl")
            }
        }

        # Default model selection
        self.selected_model = "Random Forest"
        self.load_selected_model()

    def load_selected_model(self):
        """ Load the selected model from app.py """
        model_file = "../Algorithms/selected_model.txt"

        if os.path.exists(model_file):
            with open(model_file, "r") as file:
                model_name = file.read().strip()
                if model_name in self.models:
                    self.selected_model = model_name

        # Load the correct models
        self.best_reg = self.models[self.selected_model]["reg"]
        self.best_cls = self.models[self.selected_model]["cls"]

        print(f"✅ Using Selected Model: {self.selected_model}")

    def predict(self, train_alert, feature1, feature2):
        """ Predict train delay and severity based on user input """
        X_text = self.vectorizer.transform([train_alert]).toarray()
        X_numeric = np.array([[feature1, feature2]])
        X_input = np.hstack((X_text, X_numeric))

        predicted_delay = self.best_reg.predict(X_input)[0]
        predicted_severity = self.best_cls.predict(X_input)[0]
        severity_label = "High Delay" if predicted_severity == 1 else "Low Delay"

        return predicted_delay, severity_label

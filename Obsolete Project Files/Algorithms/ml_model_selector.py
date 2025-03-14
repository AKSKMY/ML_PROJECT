import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVR, SVC  # Import SVM models
import joblib

# Get absolute path dynamically
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets", "bike_data.xlsx"))

# Verify if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")

class ModelSelector:
    def __init__(self, dataset_path=file_path):
        self.dataset_path = dataset_path
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.label_encoder = LabelEncoder()

        # Ensure absolute path to Algorithms directory
        self.models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Algorithms"))

        # Ensure directory exists
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def train_and_save_models(self):
        """ Train multiple models and save them to the correct directory. """
        print("Loading dataset...")
        xls = pd.ExcelFile(self.dataset_path)
        df = pd.read_excel(xls, sheet_name="Sheet1")

        # Process text features
        X_text = self.vectorizer.fit_transform(df.iloc[:, 1].astype(str)).toarray()

        # Encode categorical columns
        df['Day'] = self.label_encoder.fit_transform(df['Day'].astype(str))

        # Identify numeric features
        numeric_features = [col for col in df.columns if str(col).strip() in ['146', '10']]
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

        # Combine text features with numeric features
        X_numeric = df[numeric_features].values
        X = np.hstack((X_text, X_numeric))

        # Define targets
        y_regression = df[numeric_features[1]].values  # Predicting delay in minutes
        y_classification = (df[numeric_features[1]] > 10).astype(int).values  # Classifying delay severity

        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

        # Define models
        models = {
            "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42),
                              RandomForestClassifier(n_estimators=100, random_state=42)),
            "XGBoost": (XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                        XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, verbosity=0)),
            "LightGBM": (LGBMRegressor(n_estimators=100, random_state=42),
                         LGBMClassifier(n_estimators=100, random_state=42)),
            "SVM": (SVR(C=10, kernel='rbf', gamma='scale'),  # Add SVM models
                   SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42))
        }

        results = {}

        # Train and save models
        for name, (reg, cls) in models.items():
            print(f"Training {name} models...")

            # Train models
            reg.fit(X_train, y_train_reg)
            cls.fit(X_train_cls, y_train_cls)

            # Predict and evaluate
            y_pred_reg = reg.predict(X_test)
            y_pred_cls = cls.predict(X_test_cls)

            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            acc = accuracy_score(y_test_cls, y_pred_cls)

            results[name] = (mae, acc)
            print(f"{name} - Regression MAE: {mae:.4f} | Classification Accuracy: {acc:.4f}")

            # Save models to absolute path
            reg_path = os.path.join(self.models_directory, f"best_{name.lower().replace(' ', '_')}_regressor.pkl")
            cls_path = os.path.join(self.models_directory, f"best_{name.lower().replace(' ', '_')}_classifier.pkl")

            joblib.dump(reg, reg_path)
            joblib.dump(cls, cls_path)

            print(f"✔ {name} models saved: {reg_path}, {cls_path}")

        # Save vectorizer & label encoder
        joblib.dump(self.vectorizer, os.path.join(self.models_directory, "tfidf_vectorizer.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.models_directory, "label_encoder.pkl"))

        print("✅ All models saved successfully!")

if __name__ == "__main__":
    selector = ModelSelector()
    selector.train_and_save_models()
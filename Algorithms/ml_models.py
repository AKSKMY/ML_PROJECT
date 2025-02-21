import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

class TrainPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.label_encoder = LabelEncoder()
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train_model(self, file_path="../Datasets/Latest_Dataset.xlsx"):
        """ Train the models using the dataset """
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name="Sheet1")

        # Text processing (TF-IDF)
        X_text = self.vectorizer.fit_transform(df.iloc[:, 1].astype(str)).toarray()

        # Encode categorical columns
        df['Day'] = self.label_encoder.fit_transform(df['Day'].astype(str))

        # Identify numeric features
        numeric_features = [col for col in df.columns if str(col).strip() in ['146', '10']]
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

        # Combine text features with numeric features
        X_numeric = df[numeric_features].values
        X = np.hstack((X_text, X_numeric))

        # Define target variables
        y_regression = df[numeric_features[1]].values  # Regression target (delay in minutes)
        y_classification = (df[numeric_features[1]] > 10).astype(int).values  # Classification target (>10 mins delay)

        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

        # Train models
        self.rf_reg.fit(X_train, y_train_reg)
        self.rf_cls.fit(X_train_cls, y_train_cls)

        # Evaluate models
        y_pred_rf_reg = self.rf_reg.predict(X_test)
        y_pred_rf_cls = self.rf_cls.predict(X_test_cls)

        mae_rf = mean_absolute_error(y_test_reg, y_pred_rf_reg)
        acc_rf = accuracy_score(y_test_cls, y_pred_rf_cls)

        print(f"Random Forest Regression MAE: {mae_rf:.4f}")
        print(f"Random Forest Classification Accuracy: {acc_rf:.4f}")

    def predict(self, train_alert, feature1, feature2):
        """ Predict train delay and severity based on user input """
        X_text = self.vectorizer.transform([train_alert]).toarray()
        X_numeric = np.array([[feature1, feature2]])
        X_input = np.hstack((X_text, X_numeric))

        predicted_delay = self.rf_reg.predict(X_input)[0]
        predicted_severity = self.rf_cls.predict(X_input)[0]
        severity_label = "High Delay" if predicted_severity == 1 else "Low Delay"

        return predicted_delay, severity_label

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Ensure algorithms directory exists
MODEL_DIR = os.path.join(os.path.dirname(__file__), "algorithms")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.pkl")

# Load Dataset
file_path = "./NaiveBayes/Dataset_Latest.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Extract necessary columns
stations_col = df.columns[8]
day_col = df.columns[9]

# Data preprocessing
df['stations'] = df[stations_col].apply(lambda x: ' '.join(str(x).split('\n')))
df['day'] = df[day_col]
df['features'] = df['stations'] + ' ' + df['day']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['features'], df['day'], test_size=0.2, random_state=42)

# Create pipeline with CountVectorizer and MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Save the trained model
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"Naive Bayes model saved successfully to: {MODEL_PATH}")

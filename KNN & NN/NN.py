import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

file_path = 'Latest_Dataset.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

df[df.columns[8]] = df[df.columns[8]].astype(str).fillna('')
df[df.columns[9]] = df[df.columns[9]].astype(str).fillna('')

stations_col = df.columns[8]
day_col = df.columns[9]

df['stations'] = df[stations_col].apply(lambda x: ' '.join(x.split('\n'))).astype(str).fillna('')
df['day'] = df[day_col].astype(str).fillna('')
df['features'] = df['stations']

X = df['features']
y = df['day']

null_mask = (X == '') | (y == '')
df = df[~null_mask]
X = df['features']
y = df['day']

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = make_pipeline(
    CountVectorizer(),
    MLPClassifier(
        hidden_layer_sizes=(16, 8),
        alpha=0.005,
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42
    )
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

all_stations = set()
for s in df[stations_col]:
    all_stations.update(s.split('\n'))

def predict_disruption_probability(stations, day):
    for station in stations:
        if station not in all_stations:
            return f"Error: Station '{station}' not found."
    input_data = ' '.join(stations)
    probs = model.predict_proba([input_data])[0]
    if day not in label_enc.classes_:
        return f"Error: Unknown day '{day}'."
    numeric_label = label_enc.transform([day])[0]
    prob = probs[numeric_label]
    if prob <= 0.20: cat = "Incredibly Unlikely"
    elif prob <= 0.40: cat = "Very Unlikely"
    elif prob <= 0.60: cat = "Unlikely"
    elif prob <= 0.80: cat = "Slight Chance"
    else: cat = "Decent Chance"
    return prob, cat

start_station = input("Enter your starting station: ").strip()
end_station   = input("Enter your ending station: ").strip()
day_of_travel = input("Enter day of travel: ").strip()

res = predict_disruption_probability([start_station, end_station], day_of_travel)

if isinstance(res, str):
    print(res)
else:
    prob, cat = res
    print(f"Probability '{day_of_travel}': {cat} ({prob * 100:.2f}%)")
    plt.figure(figsize=(4, 4))
    sns.barplot(x=[day_of_travel], y=[prob])
    plt.text(0, prob + 0.02, cat, ha='center')
    plt.ylim(0, 1)
    plt.title(f"Probability for '{day_of_travel}'")
    plt.show()

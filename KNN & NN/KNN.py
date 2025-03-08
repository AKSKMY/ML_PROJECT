import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

file_path = 'Latest_Dataset.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

stations_col = df.columns[8]
day_col = df.columns[9]

df['stations'] = df[stations_col].apply(lambda x: ' '.join(str(x).split('\n')))
df['day'] = df[day_col]
df['features'] = df['stations'] + ' ' + df['day']

X_train, X_test, y_train, y_test = train_test_split(
    df['features'], df['day'], test_size=0.2, random_state=42
)

model = make_pipeline(
    CountVectorizer(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

all_stations = set()
for s in df[stations_col]:
    all_stations.update(str(s).split('\n'))

def predict_disruption_probability(stations, day):
    for station in stations:
        if station not in all_stations:
            return f"Error: Station '{station}' does not exist in the dataset."

    input_data = ' '.join(stations) + ' ' + day
    probs = model.predict_proba([input_data])[0]

    if day not in model.classes_:
        return f"Error: Day '{day}' not recognized. Known days: {list(model.classes_)}"

    idx = list(model.classes_).index(day)
    prob = probs[idx]

    if prob <= 0.20:
        category = "Incredibly Unlikely"
    elif prob <= 0.40:
        category = "Very Unlikely"
    elif prob <= 0.60:
        category = "Unlikely"
    elif prob <= 0.80:
        category = "Slight Chance"
    else:
        category = "Decent Chance"

    return prob, category

start_station = input("Enter your starting station: ").strip()
end_station = input("Enter your ending station: ").strip()
day_of_travel = input("Enter the day of travel: ").strip()

stations_input = [start_station, end_station]
result = predict_disruption_probability(stations_input, day_of_travel)

if isinstance(result, str):
    print(result)
else:
    prob, category = result
    print(f"Disruption Probability for {day_of_travel}: {category} ({prob * 100:.2f}%)")
    plt.figure(figsize=(5, 5))
    sns.barplot(x=[day_of_travel], y=[prob])
    plt.text(0, prob + 0.02, category, ha='center', fontsize=12)
    plt.title(f'Disruption Probability on {day_of_travel}')
    plt.xlabel('Day')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Dataset loaded here
file_path = 'Dataset_Latest.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Extract the columns needed to make predictions. In this case, just list of stations and day of week. Minutes not utilized
stations_col = df.columns[8]
day_col = df.columns[9]

# Some data preprocessing here
df['stations'] = df[stations_col].apply(lambda x: ' '.join(str(x).split('\n')))
df['day'] = df[day_col]

df['features'] = df['stations'] + ' ' + df['day']

# Data is split between training and testing sets here
X_train, X_test, y_train, y_test = train_test_split(df['features'], df['day'], test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Get the list of all unique stations in the dataset
all_stations = set()
for stations in df[stations_col]:
    all_stations.update(str(stations).split('\n'))

def predict_disruption_probability(stations, day):
    # Check if all provided stations exist in the dataset
    for station in stations:
        if station not in all_stations:
            return f"Error: Station '{station}' does not exist in the dataset."
    
    input_data = ' '.join(stations) + ' ' + day
    prediction_prob = model.predict_proba([input_data])
    disruption_prob = prediction_prob[0][model.classes_.tolist().index(day)]
    
    # Convert probability to descriptive word
    if disruption_prob <= 0.20:
        return "Very Low Chance"
    elif disruption_prob <= 0.40:
        return "Low Chance"
    elif disruption_prob <= 0.60:
        return "Medium Chance"
    elif disruption_prob <= 0.80:
        return "High Chance"
    else:
        return "Very High Chance"

# Example usage for each day of the week
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Not too sure yet whether putting starting/ending station, or all stations, makes a difference. Will experiment further.
stations_input = ['Simei', 'Tampines', 'Tanah Merah']

for day in days_of_week:
    disruption_probability = predict_disruption_probability(stations_input, day)
    if isinstance(disruption_probability, str):
        print(f"{day}: {disruption_probability}")
    else:
        print(f"{day}: Probability of encountering a disruption: {disruption_probability:.2f}%")
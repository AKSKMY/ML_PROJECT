import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

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

# Get unique stations
all_stations = set()
for stations in df[stations_col]:
    all_stations.update(str(stations).split('\n'))

# Function to predict disruption probability
def predict_disruption_probability(stations, day):
    for station in stations:
        if station not in all_stations:
            return f"Error: Station '{station}' does not exist in the dataset."

    input_data = ' '.join(stations) + ' ' + day
    prediction_prob = model.predict_proba([input_data])
    disruption_prob = prediction_prob[0][model.classes_.tolist().index(day)]

    # Convert probability to descriptive category
    if disruption_prob <= 0.20:
        category = "Incredibly Unlikely"
    elif disruption_prob <= 0.40:
        category = "Very Unlikely"
    elif disruption_prob <= 0.60:
        category = "Unlikely"
    elif disruption_prob <= 0.80:
        category = "Slight Chance"
    else:
        category = "Decent Chance"

    return disruption_prob, category

# Get user input
start_station = input("Enter your starting station: ").strip()
end_station = input("Enter your ending station: ").strip()
day_of_travel = input("Enter the day of travel (e.g., Monday): ").strip()

stations_input = [start_station, end_station]

# Predict disruption probability
result = predict_disruption_probability(stations_input, day_of_travel)

# Display results
if isinstance(result, str):  # Error handling
    print(result)
else:
    prob, category = result
    print(f"Disruption Probability for {day_of_travel}: {category} ({prob * 100:.2f}%)")

    # Plot using Seaborn
    plt.figure(figsize=(5, 5))
    sns.barplot(x=[day_of_travel], y=[prob], color='red')

    # Annotate the bar with descriptive category
    plt.text(0, prob + 0.02, category, ha='center', fontsize=12)

    plt.title(f'Disruption Probability on {day_of_travel}')
    plt.xlabel('Day')
    plt.ylabel('Disruption Probability')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1 for probability scale
    plt.show()

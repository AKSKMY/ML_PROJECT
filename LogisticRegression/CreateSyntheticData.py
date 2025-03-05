import pandas as pd
import numpy as np

# Load the dataset
disruption_data = pd.read_excel("Dataset_Latest (1).xlsx")

# Standardize column names: convert to lowercase and replace spaces/underscores
disruption_data.columns = [col.strip().lower().replace(" ", "_") for col in disruption_data.columns]

# Ensure 'date_and_time' is in datetime format
if 'date_and_time' not in disruption_data.columns:
    raise KeyError("Column 'date_and_time' is missing from the dataset. Please check the column names.")

disruption_data['date_and_time'] = pd.to_datetime(disruption_data['date_and_time'], errors="coerce")
disruption_data = disruption_data.dropna(subset=['date_and_time'])  # Drop rows with invalid dates

# Handle missing or invalid 'duration' values
if 'duration' not in disruption_data.columns:
    disruption_data['duration'] = 0  # Default duration of 0 minutes
else:
    disruption_data['duration'] = pd.to_numeric(disruption_data['duration'], errors="coerce").fillna(0).astype(int)

# Create synthetic non-disruption data
synthetic_non_disruption_data = []

for _, row in disruption_data.iterrows():
    # Calculate the end time of the disruption
    disruption_end_time = row['date_and_time'] + pd.Timedelta(minutes=row['duration'])
    
    # Create a new row for the non-disruption entry
    non_disruption_entry = row.copy()
    non_disruption_entry['date_and_time'] = disruption_end_time
    non_disruption_entry['message'] = "Train services are running smoothly with no delays."
    non_disruption_entry['duration'] = 0  # No additional travel time
    non_disruption_entry['start_and_end_station_problem'] = "No issues reported"
    non_disruption_entry['label'] = 0  # Label as non-disruption
    
    # Append the synthetic non-disruption entry
    synthetic_non_disruption_data.append(non_disruption_entry)

# Convert synthetic non-disruption data to a DataFrame
synthetic_non_disruption_data = pd.DataFrame(synthetic_non_disruption_data)

# Combine original disruption data and synthetic non-disruption data
combined_data = pd.concat(
    [disruption_data.assign(label=1), synthetic_non_disruption_data], ignore_index=True
)

# Shuffle the combined dataset
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
combined_data.to_excel("Combined_Dataset.xlsx", index=False)

print("Synthetic non-disruption data created and combined with original dataset.")
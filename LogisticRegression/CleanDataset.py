import pandas as pd
import re

# Load the dataset
file_path = "Combined_Dataset.xlsx"
data = pd.read_excel(file_path)

# Remove unnecessary columns
data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors="ignore")

# Standardize column names
data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

# Handle missing values
data = data.dropna(subset=["line", "message", "date_and_time"])
data["duration"] = data["duration"].fillna(0)

# Clean text fields
def clean_text(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"- smrt|sbs transit", "", text)
        text = re.sub(r"we apologize for the inconvenience caused", "", text)
        text = re.sub(r"please refer to fig.*", "", text)
        text = re.sub(r"https?://\S+", "", text)
    return text

data["message"] = data["message"].apply(clean_text)
data["start_and_end_station_problem"] = data["start_and_end_station_problem"].apply(clean_text)

# Format date and time
data["date_and_time"] = pd.to_datetime(data["date_and_time"], errors="coerce")
data = data.dropna(subset=["date_and_time"])

# Remove duplicates
data = data.drop_duplicates()
data = data.reset_index(drop=True)

# Standardize labels
if "label" not in data.columns:
    data["label"] = 1  # Assume all rows are disruptions unless specified otherwise
data["label"] = data["label"].astype(int)

# Sort by date and time
data = data.sort_values(by="date_and_time", ascending=True)

# Save the cleaned dataset
cleaned_file_path = "Cleaned_Dataset.xlsx"
data.to_excel(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
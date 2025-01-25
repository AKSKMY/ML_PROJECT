from datetime import datetime
import pytz
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# Path to your ChromeDriver
chromedriver_path = r"C:\Users\sel12\git\GITHUB DESKTOP\ML_PROJECT\chromedriver-win64\chromedriver.exe"

# Setup ChromeDriver
service = Service(chromedriver_path)
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--incognito")  # Open in incognito mode

# Initialize the driver
driver = webdriver.Chrome(service=service, options=options)

# Twitter login credentials
twitter_username = "@Learning98123"  # Provided Twitter username
twitter_password = "MachineLearning987@"  # Provided Twitter password

#! Keywords for trigger/processing tweets
keywords = [
    "North East Line", "NEL", "Downtown Line", "DTL",
    "Sengkang LRT line", "Punggol LRT line", "SPLRT", "delay", "MRT", "breakdown"
]
#! EXCLUDED KEYWORDS
exclude_keywords = ["extended", "Services"]  # Add more as needed

# Retry mechanism for logging into Twitter
for attempt in range(3):
    try:
        driver.get("https://twitter.com/login")
        time.sleep(5)  # Wait for the login page to load

        # Enter username/email
        username_field = driver.find_element(By.NAME, "text")
        username_field.send_keys(twitter_username)
        username_field.send_keys(Keys.RETURN)
        time.sleep(3)

        # Enter password
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys(twitter_password)
        password_field.send_keys(Keys.RETURN)
        time.sleep(5)  # Wait for login to complete
        print("Login successful.")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1}: Failed to log in. Retrying...")
        if attempt == 2:
            raise Exception("Failed to log into Twitter after multiple retries.")

# Retry mechanism for loading the Twitter profile
url = "https://x.com/SBSTransit_Ltd"
for attempt in range(3):
    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article[data-testid='tweet']"))
        )
        print("Tweets loaded successfully.")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1}: Failed to load tweets. Retrying...")
        if attempt == 2:
            raise Exception("Failed to load tweets after multiple retries.")

#! Scroll to load more tweets
scroll_pause_time = 10  # Longer delay to reduce rate limiting
max_scroll_attempts = 100  # Maximum number of scroll attempts
target_tweet_count = 100  # Number of relevant tweets to scrape
tweet_data = []  # List to store tweet data
unique_tweets = set()  # Set to track unique tweet texts
previous_height = 0  # Track scroll height to detect page load completion

# Define patterns for dates and times in tweets
date_pattern = r"\b(\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\d{1,2}/\d{1,2})\b"  # Includes 15 Oct 2023 and 15/10
time_pattern = r"\b(\d{1,2}:\d{2}(?:AM|PM)|\d{1,2}\.\d{2}(?:am|pm)|\d{4}hrs)\b"  # Includes 10:30AM, 4.53pm, and 1030hrs

# Function to check if a tweet contains excluded keywords
def contains_excluded_keywords(text, exclude_keywords):
    return any(keyword.lower() in text.lower() for keyword in exclude_keywords)

# Start scrolling
while len(tweet_data) < target_tweet_count:
    # Scroll down
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)  # Allow time for tweets to load

    # Extract tweets on the page
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    tweets = soup.find_all("article", attrs={"data-testid": "tweet"})
    print(f"Total tweets found on page: {len(tweets)}")  # Log the number of tweets

    #TODO Extract data from tweets
    for tweet in tweets:
        if len(tweet_data) >= target_tweet_count:
            print(f"Collected {target_tweet_count} relevant tweets, stopping.")
            break  # Stop collecting once the target count is reached

        try:
            # Extract tweet text
            text_element = tweet.find("div", {"data-testid": "tweetText"})
            text = text_element.get_text(strip=True) if text_element else "N/A"
            # print(f"Processing tweet: {text}")

            # Skip duplicate tweets
            if text in unique_tweets:
                # print(f"Skipped tweet (duplicate): {text}")
                continue

            # Check if the tweet contains any of the keywords
            if not any(keyword in text for keyword in keywords):
                # print(f"Skipped tweet (no matching keywords): {text}")
                continue

            # Exclude tweets containing specific keywords
            if contains_excluded_keywords(text, exclude_keywords):
                print(f"Skipped tweet (excluded keyword): {text}")
                continue

            # Extract username
            username_element = tweet.find("div", {"dir": "ltr"})
            username = username_element.get_text(strip=True) if username_element else "N/A"

            # Extract posted date and time from metadata
            metadata_element = tweet.find("time")
            if metadata_element:
                datetime_metadata = metadata_element["datetime"]  # Format: YYYY-MM-DDTHH:MM:SSZ
                # Convert from UTC to local time
                utc = pytz.utc
                local_tz = pytz.timezone("Asia/Singapore")
                utc_dt = datetime.strptime(datetime_metadata, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=utc)
                local_dt = utc_dt.astimezone(local_tz)
                posted_date = local_dt.strftime("%Y-%m-%d")
                posted_time = local_dt.strftime("%H:%M")
            else:
                posted_date = "N/A"
                posted_time = "N/A"

            # Scan tweet text for mentioned dates and times
            mentioned_dates = re.findall(date_pattern, text, re.IGNORECASE)
            mentioned_times = re.findall(time_pattern, text, re.IGNORECASE)

            # Combine matches into strings (if any found)
            tweet_mentioned_dates = "; ".join(mentioned_dates) if mentioned_dates else "N/A"
            tweet_mentioned_times = "; ".join(mentioned_times) if mentioned_times else "N/A"

            print(f"Tweet added to Excel data: {text} | Username: {username} | Date: {posted_date} | Time: {posted_time}")

            # Append to tweet_data
            tweet_data.append({
                "Username": username,
                "Tweet": text,
                "Posted Date": posted_date,
                "Posted Time": posted_time,
                "Mentioned Dates": tweet_mentioned_dates,  # Column for dates mentioned in tweets
                "Mentioned Times": tweet_mentioned_times   # Column for times mentioned in tweets
            })

            # Add to the set of unique tweets
            unique_tweets.add(text)

        except Exception as e:
            print(f"Error extracting tweet: {e}")

    # Check if scrolling reached the bottom of the page
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == previous_height:
        print("No more tweets to load, stopping.")
        break  # Stop scrolling if no new tweets load
    previous_height = new_height

print(f"Collected {len(tweet_data)} relevant tweets.")

# Close the driver
driver.quit()

# Save the tweets to an Excel file
df = pd.DataFrame(tweet_data)
output_path = r"tweets_scraped.xlsx"
df.to_excel(output_path, index=False)
print(f"Tweets saved to {output_path}")

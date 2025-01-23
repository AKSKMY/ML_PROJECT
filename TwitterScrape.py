import re  # For regular expressions to extract time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# Path to your ChromeDriver
chromedriver_path = r"C:\Users\azzi\Desktop\Scrape\chromedriver-win64\chromedriver.exe"

# Setup ChromeDriver
service = Service(chromedriver_path)
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

# Initialize the driver
driver = webdriver.Chrome(service=service, options=options)

# Open the specified Twitter (X) profile
url = "https://x.com/SBSTransit_Ltd"
driver.get(url)

# Wait for tweets to load
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article[data-testid='tweet']"))
    )
    print("Tweets loaded successfully.")
except Exception as e:
    print(f"Error waiting for tweets: {e}")

# Scroll to load more tweets (adjust range for more tweets)
for _ in range(5):  # Number of scrolls
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # Wait for content to load after scrolling

# Get the page source after scrolling
page_source = driver.page_source

# Parse the page source with BeautifulSoup
soup = BeautifulSoup(page_source, "html.parser")

# Extract tweet containers
tweets = soup.find_all("article", attrs={"data-testid": "tweet"})

# Prepare a list to store tweet data
tweet_data = []

# Regular expression for time
time_pattern = r'\b(\d{1,2}[.:]\d{2}\s?[apAP][mM])\b'  # Matches formats like 6.20pm or 08:02 AM

# Loop through tweets and extract data
for tweet in tweets:
    try:
        # Extract tweet text
        text_element = tweet.find("div", {"data-testid": "tweetText"})
        text = text_element.get_text(strip=True) if text_element else "N/A"
        
        # Extract username
        username_element = tweet.find("div", {"dir": "ltr"})
        username = username_element.get_text(strip=True) if username_element else "N/A"
        
        # Extract time from tweet text
        time_match = re.search(time_pattern, text)
        tweet_time = time_match.group(1) if time_match else "N/A"
        
        # Extract posted date from metadata
        metadata_element = tweet.find("time")
        posted_date = metadata_element["datetime"][:10] if metadata_element else "N/A"  # Format: YYYY-MM-DD

        # Append data to the list
        tweet_data.append({
            "Username": username,
            "Tweet": text,
            "Time of Tweet": tweet_time,
            "Posted Date": posted_date
        })
    except Exception as e:
        print(f"Error extracting tweet: {e}")

# Close the driver
driver.quit()

# Create a DataFrame and save to Excel
df = pd.DataFrame(tweet_data)
output_path = r"tweets_scraped.xlsx"  # Adjust file path as needed
df.to_excel(output_path, index=False)
print(f"Tweets saved to {output_path}")

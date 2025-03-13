import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape data from a single bike listing page
def scrape_bike_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the bike name
        bike_name = soup.find('h2', class_='card-title text-center').text.strip()
        
        # Initialize a dictionary to store the scraped data
        bike_data = {
            'Bike Name': bike_name,
            'Listing Type': None,
            'Brand': None,
            'Model': None,
            'Engine Capacity': None,
            'Classification': None,
            'Registration Date': None,
            'COE Expiry Date': None,
            'Mileage': None,
            'No. of owners': None,
            'Type of Vehicle': None,
            'Price': None
        }
        
        # Extract data from the table
        table = soup.find('table', class_='table mb-0')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) == 3:
                    key = cells[0].text.strip()
                    value = cells[2].text.strip()
                    
                    # Map the key to the corresponding field in the dictionary
                    if key == 'Listing Type':
                        bike_data['Listing Type'] = value
                    elif key == 'Brand':
                        bike_data['Brand'] = value
                    elif key == 'Model':
                        bike_data['Model'] = value
                    elif key == 'Engine Capacity':
                        bike_data['Engine Capacity'] = value
                    elif key == 'Classification':
                        bike_data['Classification'] = value
                    elif key == 'Registration Date':
                        bike_data['Registration Date'] = value
                    elif key == 'COE Expiry Date':
                        bike_data['COE Expiry Date'] = value.split()[0]  # Extract only the date
                    elif key == 'Mileage':
                        bike_data['Mileage'] = value
                    elif key == 'No. of owners':
                        bike_data['No. of owners'] = value
                    elif key == 'Type of Vehicle':
                        bike_data['Type of Vehicle'] = value
        
        # Extract the price
        price = soup.find('h2', class_='text-center strong').text.strip()
        bike_data['Price'] = price
        
        return bike_data
    else:
        print(f"Failed to retrieve the page: {url}. Status code: {response.status_code}")
        return None

# Function to extract URLs of all bike listings from a list page
def extract_listing_urls(list_url):
    response = requests.get(list_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Use a set to store unique URLs
        listing_urls = set()
        
        # Find all links to individual bike listings
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Ensure the link is a bike listing and not a duplicate
            if '/listing/usedbike/' in href and '/model/' not in href:
                full_url = "https://sgbikemart.com.sg" + href
                listing_urls.add(full_url)
        
        return list(listing_urls)  # Convert the set back to a list
    else:
        print(f"Failed to retrieve the list page: {list_url}. Status code: {response.status_code}")
        return []

# Main script
if __name__ == "__main__":
    # Base URL of the list page
    base_url = "https://sgbikemart.com.sg/listing/usedbike/model/motorcycle-for-sale/scrambler-dirt-bike/"
    
    # Number of pages to scrape
    total_pages = 10
    
    # Scrape data from each page
    all_bike_data = []
    for page in range(1, total_pages + 1):
        # Construct the URL for the current page
        list_url = f"{base_url}?page={page}&"
        print(f"Scraping page {page}: {list_url}")
        
        # Extract URLs of all bike listings from the current page
        listing_urls = extract_listing_urls(list_url)
        
        # Scrape data from each listing
        for url in listing_urls:
            print(f"Scraping data from: {url}")
            bike_data = scrape_bike_data(url)
            if bike_data:
                all_bike_data.append(bike_data)
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_bike_data)
    
    # Save the DataFrame to an Excel file
    excel_file = "scrambler-dirt-bike.xlsx"
    df.to_excel(excel_file, index=False)
    
    print(f"Data saved to {excel_file}")
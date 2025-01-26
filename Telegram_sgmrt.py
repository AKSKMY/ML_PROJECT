from telethon import TelegramClient, errors
import pandas as pd
import logging
import re

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace these with your Telegram API credentials from https://telegram.org/
api_id = 26398976  # Replace with your API ID
api_hash = "02927ad4f026886f0a6b1234918b2504"  # Replace with your API Hash
phone_number = "+6596567940"  # Replace with your phone number

# Telegram Client initialization
client = TelegramClient('infmrt_scraper', api_id, api_hash)

# Define the target channel
target_channel = "sgmrt"  # The username or link of the target channel

# List to store scraped messages
messages_data = []

# Base mapping for MRT and LRT lines
base_mapping = {
    "East-West Line": "EWL",
    "North-South Line": "NSL",
    "Downtown Line": "DTL",
    "North-East Line": "NEL",
    "Circle Line": "CCL",
    "Thomson-East Coast Line": "TEL",
    "Bukit Panjang LRT": "BPL",
    "Punggol East LRT": "SPL",
    "Punggol West LRT": "SPL",
    "Sengkang-Punggol LRT": "SPL",
    "Sengkang LRT": "SPL",
    "Punggol LRT": "SPL",
}

# Generate dynamic mapping with variations (e.g., with/without spaces or hyphens)
mrt_lrt_mapping = {}
for full_name, short_form in base_mapping.items():
    # Add base name
    mrt_lrt_mapping[full_name] = short_form
    # Add variations: no spaces, no hyphens, no spaces and no hyphens
    mrt_lrt_mapping[full_name.replace(" ", "")] = short_form
    mrt_lrt_mapping[full_name.replace("-", "")] = short_form
    mrt_lrt_mapping[full_name.replace(" ", "").replace("-", "")] = short_form

# Add abbreviations directly (e.g., "EWL")
for abbreviation in set(base_mapping.values()):
    mrt_lrt_mapping[abbreviation] = abbreviation

def get_mrt_lrt_lines(message):
    """Find and collect all MRT or LRT lines mentioned in the message."""
    mentioned_lines = []
    for full_name, short_form in mrt_lrt_mapping.items():
        if full_name in message:
            mentioned_lines.append(short_form)
    return ", ".join(set(mentioned_lines)) if mentioned_lines else None  # Remove duplicates and return as a string

def get_sender(message):
    """Extract the sender (e.g., '- SBS Transit', '- SMRT') from the message."""
    sender_match = re.search(r"-\s*(\w[\w\s]*?)$", message)
    return sender_match.group(1).strip() if sender_match else None

async def scrape_telegram_channel(channel_name):
    try:
        async with client:
            # Get the target channel entity
            logger.info(f"Fetching channel: {channel_name}")
            try:
                channel = await client.get_entity(channel_name)
            except errors.UsernameNotOccupiedError:
                logger.error(f"The channel '{channel_name}' does not exist or you are not a member.")
                return

            # Fetch unlimited messages from the channel
            logger.info(f"Fetching all available messages from {channel_name}...")
            async for message in client.iter_messages(channel, limit=None):  # Set limit=None for unlimited
                if message.message:  # Skip empty messages
                    # Extract MRT or LRT line information
                    mrt_lrt_lines = get_mrt_lrt_lines(message.message)

                    # Extract Date and Time
                    date = message.date.strftime("%Y-%m-%d")
                    time = message.date.strftime("%H:%M:%S")

                    # Extract the sender
                    sender = get_sender(message.message)

                    # Append processed data
                    messages_data.append({
                        "MRT/LRT Line": mrt_lrt_lines,
                        "Message": message.message,
                        "Date": date,
                        "Time": time,
                        "Sender": sender
                    })

            logger.info(f"Collected {len(messages_data)} messages from {channel_name}.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


# Run the scraping function and disconnect after completion
with client:
    client.loop.run_until_complete(scrape_telegram_channel(target_channel))

# Save scraped messages to an Excel file
if messages_data:
    df = pd.DataFrame(messages_data)
    output_file = "sgmrt_telegram2.xlsx"
    df.to_excel(output_file, index=False)
    logger.info(f"Messages saved to {output_file}")
else:
    logger.warning("No messages were collected. Please check the channel or your credentials.")

# Disconnect the client after execution
client.disconnect()
logger.info("Client disconnected. Script finished.")
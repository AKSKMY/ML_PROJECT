from telethon import TelegramClient, errors
import pandas as pd
import logging

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Login to https://telegram.org/ to create your account
api_id = 26398976  # Replace with your API ID
api_hash = "02927ad4f026886f0a6b1234918b2504"  # Replace with your API Hash
phone_number = "+6596567940"  # Replace with your phone number

# Telegram Client initialization
client = TelegramClient('infmrt_scraper', api_id, api_hash)

# Define the target channel
target_channel = "sgmrt"  # The username or link of the target channel

# List to store scraped messages
messages_data = []

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
                messages_data.append({
                    "Date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Sender": message.sender_id,
                    "Message": message.message
                })

            logger.info(f"Collected {len(messages_data)} messages from {channel_name}.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


# Run the scraping function
with client:
    client.loop.run_until_complete(scrape_telegram_channel(target_channel))

# Save scraped messages to an Excel file
if messages_data:
    df = pd.DataFrame(messages_data)
    output_file = "sgmrt_telegram.xlsx"
    df.to_excel(output_file, index=False)
    logger.info(f"Messages saved to {output_file}")
else:
    logger.warning("No messages were collected. Please check the channel or your credentials.")

import os
import kaggle
import schedule
import time
import logging
from datasets import load_dataset
from datetime import datetime

# Ensure logs directory exists
LOGS_PATH = "logs/"
os.makedirs(LOGS_PATH, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(LOGS_PATH, 'data_storage.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set dataset paths
KAGGLE_DATASET = "muhammadshahidazeem/customer-churn-dataset"
HUGGINGFACE_DATASET_NAME = "scikit-learn/churn-prediction"
LOCAL_STORAGE_PATH = "raw_data/"

def get_timestamped_path(source):
    date_str = datetime.now().strftime('%Y-%m-%d')
    path = os.path.join(LOCAL_STORAGE_PATH, source, date_str)
    os.makedirs(path, exist_ok=True)
    return path

def download_kaggle_data():
    try:
        logging.info("Starting Kaggle data ingestion...")
        kaggle_path = get_timestamped_path("kaggle")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=kaggle_path, unzip=True)
        logging.info(f"Kaggle data successfully downloaded and stored in {kaggle_path}")
    except Exception as e:
        logging.error(f"Kaggle data ingestion failed: {e}")

def download_huggingface_data():
    try:
        logging.info("Starting Hugging Face data ingestion...")
        hf_path = get_timestamped_path("huggingface")
        dataset = load_dataset(HUGGINGFACE_DATASET_NAME)
        df = dataset["train"].to_pandas()
        hf_csv_path = os.path.join(hf_path, 'huggingface_churn.csv')
        df.to_csv(hf_csv_path, index=False)
        logging.info(f"Hugging Face data successfully downloaded and stored in {hf_csv_path}")
    except Exception as e:
        logging.error(f"Hugging Face data ingestion failed: {e}")

def download_data():
    download_kaggle_data()
    download_huggingface_data()

# Schedule the ingestion to run daily at 12 AM
schedule.every().day.at("00:00").do(download_data)

if __name__ == "__main__":
    download_data()  # Run once immediately
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait for the next scheduled run

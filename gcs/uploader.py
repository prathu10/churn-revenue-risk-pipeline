import os
import argparse
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from shared.logging_config import setup_logger

logger = setup_logger("gcs.uploader")

# Load environment variables
load_dotenv()

def get_storage_client():
    """Initializes the GCS client using credentials in environment variables."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GCP_PROJECT_ID")
    
    if creds_path and os.path.exists(creds_path):
        logger.info(f"Initializing Storage Client using credentials file: {creds_path}")
        return storage.Client.from_service_account_json(creds_path, project=project_id)
    else:
        logger.info("Initializing Storage Client using default/ambient credentials.")
        return storage.Client(project=project_id)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    logger.info(f"Attempting to upload '{source_file_name}' to bucket '{bucket_name}' as '{destination_blob_name}'...")
    
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        
        # Verify bucket exists (optional, but good for reporting errors)
        if not bucket.exists():
            logger.error(f"Bucket '{bucket_name}' does not exist! Please check configuration.")
            return False
            
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        
        logger.info(f"File '{source_file_name}' successfully uploaded to '{destination_blob_name}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to upload blob: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to Google Cloud Storage")
    parser.add_argument("--source", type=str, required=True, help="Local file path to upload")
    parser.add_argument("--destination", type=str, required=True, help="Target blob name in GCS")
    parser.add_argument("--bucket", type=str, default=None, help="GCS Bucket name (overrides env variable)")
    
    args = parser.parse_args()
    
    bucket_name = args.bucket or os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        logger.error("Bucket name not provided. Set GCS_BUCKET_NAME in .env or use the --bucket flag.")
        exit(1)
        
    success = upload_blob(bucket_name, args.source, args.destination)
    if not success:
        exit(1)

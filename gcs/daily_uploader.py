import os
import re
import argparse
from dotenv import load_dotenv
from gcs.uploader import get_storage_client
from shared.logging_config import setup_logger

logger = setup_logger("gcs.daily_uploader")

# Load environments
load_dotenv()

def parse_date_from_filename(filename):
    """
    Extracts an 8-digit date string from a filename and converts it to YYYY-MM-DD format.
    Example: 'events_20260706.json' -> '2026-07-06'
    """
    match = re.search(r"(\d{8})", filename)
    if not match:
        return None
        
    date_str = match.group(1)
    try:
        # Validate date format (YYYYMMDD)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:]
        # Simple bounds check
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{month}-{day}"
    except ValueError:
        pass
        
    return None

def upload_daily_files(source_dir, bucket_name):
    """
    Scans source_dir for daily event/status files and uploads them to GCS.
    Files are organized as raw/YYYY-MM-DD/filename.
    """
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return False
        
    logger.info(f"Scanning directory '{source_dir}' for daily files to upload...")
    
    files_to_upload = []
    for f in os.listdir(source_dir):
        file_path = os.path.join(source_dir, f)
        if os.path.isfile(file_path) and (f.startswith("events_") or f.startswith("customer_status_")):
            date_folder = parse_date_from_filename(f)
            if date_folder:
                files_to_upload.append((file_path, f, date_folder))
                
    if not files_to_upload:
        logger.info("No matching daily files found for upload.")
        return True
        
    logger.info(f"Found {len(files_to_upload)} files to upload.")
    
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.error(f"Bucket '{bucket_name}' does not exist! Please check configurations.")
            return False
            
        success_count = 0
        for local_path, filename, date_folder in files_to_upload:
            destination_blob = f"raw/{date_folder}/{filename}"
            logger.info(f"Uploading '{filename}' to GCS path '{destination_blob}'...")
            
            try:
                blob = bucket.blob(destination_blob)
                blob.upload_from_filename(local_path)
                logger.info(f"Successfully uploaded: {filename} -> {destination_blob}")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to upload '{filename}': {str(e)}")
                
        logger.info(f"Batch upload finished. Successfully uploaded {success_count}/{len(files_to_upload)} files.")
        return success_count == len(files_to_upload)
        
    except Exception as e:
        logger.error(f"Failed to connect or access GCS: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload daily simulated files to GCS")
    parser.add_argument("--dir", type=str, default=None, help="Source directory containing files (defaults to output/daily_streams)")
    parser.add_argument("--bucket", type=str, default=None, help="GCS Bucket name (overrides GCS_BUCKET_NAME in .env)")
    args = parser.parse_args()
    
    # Defaults
    source_directory = args.dir or os.path.join(os.path.dirname(__file__), "../output/daily_streams")
    bucket_name = args.bucket or os.getenv("GCS_BUCKET_NAME")
    
    if not bucket_name:
        logger.error("GCS Bucket name not provided. Set GCS_BUCKET_NAME in .env or use the --bucket flag.")
        exit(1)
        
    success = upload_daily_files(source_directory, bucket_name)
    if not success:
        exit(1)

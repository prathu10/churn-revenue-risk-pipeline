import os
import argparse
from dotenv import load_dotenv
from gcs.uploader import get_storage_client
from shared.logging_config import setup_logger

logger = setup_logger("gcs.verify_uploader")

# Load environment variables
load_dotenv()

def format_size(bytes_size):
    """Formats raw bytes size into human-readable units (KB, MB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def list_bucket_contents(bucket_name, prefix=None, limit=50):
    """Lists blobs inside the GCS bucket and prints their details."""
    logger.info(f"Listing contents of GCS bucket: '{bucket_name}'" + (f" with prefix '{prefix}'" if prefix else ""))
    
    try:
        client = get_storage_client()
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.error(f"Bucket '{bucket_name}' does not exist! Please check configurations.")
            return False
            
        blobs = client.list_blobs(bucket, prefix=prefix)
        
        # Collect and print
        blob_list = list(blobs)
        if not blob_list:
            logger.info("The bucket is currently empty or no items match the prefix.")
            return True
            
        logger.info(f"Found {len(blob_list)} objects in bucket (showing up to {limit}):")
        
        # Display header
        print("\n" + "=" * 90)
        print(f"{'Blob Path':<50} | {'Size':<12} | {'Created Time':<22}")
        print("=" * 90)
        
        count = 0
        for blob in blob_list:
            if count >= limit:
                break
            size_str = format_size(blob.size or 0)
            created_str = blob.time_created.strftime("%Y-%m-%d %H:%M:%S") if blob.time_created else "N/A"
            print(f"{blob.name:<50} | {size_str:<12} | {created_str:<22}")
            count += 1
            
        print("=" * 90 + "\n")
        return True
        
    except Exception as e:
        logger.error(f"Failed to list bucket contents: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GCS Uploads by listing bucket contents")
    parser.add_argument("--bucket", type=str, default=None, help="GCS Bucket name (overrides GCS_BUCKET_NAME in env)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix filter for blobs (e.g. 'raw/')")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of blobs to display")
    args = parser.parse_args()
    
    bucket_name = args.bucket or os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        logger.error("Bucket name not provided. Set GCS_BUCKET_NAME in .env or use the --bucket flag.")
        exit(1)
        
    success = list_bucket_contents(bucket_name, prefix=args.prefix, limit=args.limit)
    if not success:
        exit(1)

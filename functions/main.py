import os
import json
import logging
import sys
from google.cloud import storage
from google.cloud import bigquery
import functions_framework

# Self-contained logger for deployment independence
def get_cf_logger():
    logger = logging.getLogger("cloud_function_transform")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [CF_TRANSFORM] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_cf_logger()

# Clients
storage_client = storage.Client()
bq_client = bigquery.Client()

def transform_event(raw_event):
    """
    Transforms and cleans a single event record.
    Returns the cleaned event dict, or None if validation fails.
    """
    try:
        # Validate required fields
        for field in ["event_id", "customer_id", "event_type", "timestamp"]:
            if field not in raw_event or not raw_event[field]:
                logger.warning(f"Skipping event: missing required field '{field}'. Row: {raw_event}")
                return None
                
        # Clean value field (ensure float)
        val = raw_event.get("value", 0.0)
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0
            
        # Clean details (ensure JSON string)
        details = raw_event.get("details", "{}")
        if isinstance(details, dict):
            details = json.dumps(details)
        elif not isinstance(details, str):
            details = str(details)
            
        # Return structured row
        return {
            "event_id": str(raw_event["event_id"]),
            "customer_id": str(raw_event["customer_id"]),
            "event_type": str(raw_event["event_type"]),
            "timestamp": str(raw_event["timestamp"]),
            "value": val,
            "device": str(raw_event.get("device", "unknown")),
            "details": details
        }
    except Exception as e:
        logger.error(f"Error transforming event: {str(e)}")
        return None

@functions_framework.cloud_event
def gcs_transform_trigger(cloud_event):
    """
    Triggered by a change to a GCS bucket. 
    Downloads the file, transforms the JSON lines, and inserts into BigQuery.
    """
    data = cloud_event.data
    bucket_name = data.get("bucket")
    file_name = data.get("name")
    
    if not bucket_name or not file_name:
        logger.error("Cloud event data missing bucket or name.")
        return
        
    logger.info(f"Processing file '{file_name}' from bucket '{bucket_name}'...")
    
    # Check if we should process this file (e.g. ignore non-json files or outputs)
    if not file_name.endswith(".json") and not file_name.endswith(".jsonl"):
        logger.info(f"File '{file_name}' is not JSON. Skipping transform.")
        return
        
    try:
        # Download data from GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        content = blob.download_as_text()
        
        # Parse JSON lines
        raw_rows = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError as je:
                    logger.warning(f"Skipping malformed JSON line: {line}. Error: {str(je)}")
                    
        if not raw_rows:
            logger.info(f"No events found in file '{file_name}'.")
            return
            
        # Transform rows
        transformed_rows = []
        for raw_row in raw_rows:
            cleaned = transform_event(raw_row)
            if cleaned:
                transformed_rows.append(cleaned)
                
        if not transformed_rows:
            logger.info("No rows passed validation. No data to load.")
            return
            
        # Write to BigQuery
        dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
        table_name = os.getenv("BIGQUERY_TABLE_EVENTS", "events")
        table_ref = bq_client.dataset(dataset_id).table(table_name)
        
        logger.info(f"Inserting {len(transformed_rows)} rows into BQ table '{dataset_id}.{table_name}'...")
        errors = bq_client.insert_rows_json(table_ref, transformed_rows)
        
        if errors == []:
            logger.info("Database ingestion successfully completed.")
        else:
            logger.error(f"Failed to insert rows. BQ errors: {errors}")
            
    except Exception as e:
        logger.error(f"Error during GCS event processing: {str(e)}")

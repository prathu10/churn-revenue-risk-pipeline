import os
import io
import re
import time
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import storage
from bigquery.loader import get_bq_client
from shared.logging_config import setup_logger

logger = setup_logger("bigquery.load_features")

# Load environment variables
load_dotenv()

def parse_date(date_str):
    """Parses date string into YYYYMMDD and YYYY-MM-DD formats."""
    cleaned = re.sub(r"\D", "", date_str)
    if len(cleaned) != 8:
        raise ValueError(f"Invalid date format: {date_str}. Must be YYYYMMDD or YYYY-MM-DD.")
    year = cleaned[:4]
    month = cleaned[4:6]
    day = cleaned[6:]
    return cleaned, f"{year}-{month}-{day}"

def execute_ddl_setup():
    """Initializes the dataset and tables using bigquery/ddl.sql."""
    client = get_bq_client()
    project_id = client.project
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    
    # 1. Create dataset if not exists
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset '{project_id}.{dataset_id}' verified/created.")
    except Exception as e:
        logger.error(f"Failed to create dataset '{dataset_id}': {str(e)}")
        return False
        
    # 2. Read and run DDL SQL
    ddl_path = os.path.join(os.path.dirname(__file__), "ddl.sql")
    if not os.path.exists(ddl_path):
        logger.error(f"DDL SQL file not found at: {ddl_path}")
        return False
        
    with open(ddl_path, "r") as f:
        sql_content = f.read()
        
    # Replace GCP_PROJECT_ID placeholder with runtime projectId
    sql_queries = sql_content.replace("GCP_PROJECT_ID", project_id)
    
    # Split queries by semicolon and filter empty statements
    queries = [q.strip() for q in sql_queries.split(";") if q.strip()]
    
    logger.info(f"Executing {len(queries)} DDL statements to verify/create tables...")
    for query in queries:
        try:
            query_job = client.query(query)
            query_job.result()  # Wait for query to complete
        except Exception as e:
            logger.error(f"Failed to execute DDL statement:\n{query}\nError: {str(e)}")
            return False
            
    logger.info("Database schema setup completed successfully.")
    return True

def parse_input_counts_from_log(log_content):
    """Parses raw input event/profile counts from a metrics report log file."""
    profile_cnt = 0
    event_cnt = 0
    try:
        for line in log_content.splitlines():
            if "Input Profile Count:" in line:
                profile_cnt = int(re.findall(r"\d+", line)[0])
            elif "Input Event Count:" in line:
                event_cnt = int(re.findall(r"\d+", line)[0])
    except Exception as e:
        logger.warning(f"Error parsing log file counts: {str(e)}")
        
    return profile_cnt + event_cnt

def load_data_source(date_yyyy_mm_dd, date_yyyymmdd, local_only=True, bucket_name=None):
    """Loads processed features CSV and metrics log from GCS or local disk."""
    if local_only:
        processed_dir = os.path.join(os.path.dirname(__file__), "../output/processed")
        metrics_dir = os.path.join(os.path.dirname(__file__), "../output/metrics")
        
        csv_path = os.path.join(processed_dir, f"processed_features_{date_yyyymmdd}.csv")
        log_path = os.path.join(metrics_dir, f"metrics_{date_yyyy_mm_dd}.log")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Local processed features CSV file not found at: {csv_path}")
            
        logger.info(f"Loading local features from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        records_in = len(df) # fallback default
        if os.path.exists(log_path):
            logger.info(f"Reading local metrics log: {log_path}")
            with open(log_path, "r") as f:
                records_in = parse_input_counts_from_log(f.read())
                
        return df, records_in
    else:
        # GCS cloud mode
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            storage_client = storage.Client.from_service_account_json(creds_path)
        else:
            storage_client = storage.Client()
            
        bucket = storage_client.bucket(bucket_name)
        csv_blob_name = f"processed/{date_yyyy_mm_dd}/processed_features.csv"
        log_blob_name = f"processed/{date_yyyy_mm_dd}/metrics.log"
        
        csv_blob = bucket.blob(csv_blob_name)
        if not csv_blob.exists():
            raise FileNotFoundError(f"GCS Blob not found: gs://{bucket_name}/{csv_blob_name}")
            
        logger.info(f"Downloading GCS features blob: {csv_blob_name}")
        csv_data = csv_blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
        
        records_in = len(df) # fallback default
        log_blob = bucket.blob(log_blob_name)
        if log_blob.exists():
            logger.info(f"Downloading GCS metrics log blob: {log_blob_name}")
            log_data = log_blob.download_as_text()
            records_in = parse_input_counts_from_log(log_data)
            
        return df, records_in

def load_features_pipeline(date_str, local_only=True, bucket_name=None):
    """
    Core feature ingestion pipeline task.
    Reads processed CSV, loads into customer_features, and registers pipeline run metrics.
    """
    start_time = time.time()
    date_yyyymmdd, date_yyyy_mm_dd = parse_date(date_str)
    
    # 1. Load data
    try:
        df, records_in = load_data_source(date_yyyy_mm_dd, date_yyyymmdd, local_only, bucket_name)
    except Exception as e:
        logger.error(f"Failed to load processed daily source: {str(e)}")
        return False
        
    records_out = len(df)
    if records_out == 0:
        logger.warning(f"No records found in processed dataset for date: {date_yyyy_mm_dd}. Ingestion aborted.")
        return False
        
    # 2. Ingest processed features into customer_features table
    client = get_bq_client()
    project_id = client.project
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    features_table_ref = f"{project_id}.{dataset_id}.customer_features"
    metrics_table_ref = f"{project_id}.{dataset_id}.pipeline_metrics"
    
    # Check if load_date already exists for duplicate prevention
    try:
        query = f"SELECT COUNT(1) FROM `{features_table_ref}` WHERE load_date = '{date_yyyy_mm_dd}'"
        query_job = client.query(query)
        results = list(query_job.result())
        count = results[0][0] if results else 0
        if count > 0:
            logger.warning(f"Data for date '{date_yyyy_mm_dd}' has already been loaded into '{features_table_ref}' ({count} rows). Skipping ingestion to prevent duplicates.")
            return True
    except Exception as e:
        logger.info(f"Could not check existing load dates (table might not exist or be empty): {str(e)}")
        
    # Inject load_date column matching updated table schema
    df["load_date"] = date_yyyy_mm_dd
    
    # Use load job config
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    
    logger.info(f"Ingesting {records_out} rows into BigQuery table '{features_table_ref}'...")
    
    # Convert dataframe to CSV byte buffer for BQ load
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    
    try:
        load_job = client.load_table_from_file(csv_buffer, features_table_ref, job_config=job_config)
        load_job.result() # Wait for job completion
        logger.info("Successfully loaded processed feature rows into customer_features.")
    except Exception as e:
        logger.error(f"Failed to load features to BigQuery: {str(e)}")
        return False
        
    # 3. Calculate statistics & log metrics
    run_duration = round(time.time() - start_time, 4)
    # Calculate null rates
    null_rate = df.isnull().mean().mean() * 100
    
    logger.info(f"ETL completed in {run_duration}s. Null rate: {null_rate:.2f}%. Output: {records_out} rows.")
    
    # Assemble pipeline metrics row
    metrics_row = {
        "run_date": date_yyyy_mm_dd, # BigQuery DATE column accepts YYYY-MM-DD string
        "records_in": int(records_in),
        "records_out": int(records_out),
        "null_rate": float(null_rate),
        "run_duration": float(run_duration)
    }
    
    logger.info(f"Inserting execution log row into pipeline_metrics table '{metrics_table_ref}'...")
    try:
        errors = client.insert_rows_json(metrics_table_ref, [metrics_row])
        if errors == []:
            logger.info("Successfully logged metrics to BigQuery.")
        else:
            logger.error(f"Failed to log pipeline metrics. Errors: {errors}")
            return False
    except Exception as e:
        logger.error(f"Error inserting metrics row: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Daily Processed Churn Features to BigQuery")
    parser.add_argument("--date", type=str, help="Target date YYYYMMDD or YYYY-MM-DD (required unless --setup is active)")
    parser.add_argument("--local", action="store_true", help="Load local CSV files instead of GCS")
    parser.add_argument("--setup", action="store_true", help="Verify/create the BigQuery schema first")
    parser.add_argument("--bucket", type=str, default=None, help="GCS Bucket name (overrides env variable)")
    args = parser.parse_args()
    
    bucket_name = args.bucket or os.getenv("GCS_BUCKET_NAME")
    
    if args.setup:
        success = execute_ddl_setup()
        if not success:
            exit(1)
            
    if args.date:
        # If not local and bucket name missing, raise
        if not args.local and not bucket_name:
            logger.error("Bucket name not provided. Set GCS_BUCKET_NAME in .env or use the --bucket flag, or use --local.")
            exit(1)
            
        success = load_features_pipeline(args.date, local_only=args.local, bucket_name=bucket_name)
        if not success:
            exit(1)
    elif not args.setup:
        logger.error("--date is required unless --setup is specified.")
        exit(1)

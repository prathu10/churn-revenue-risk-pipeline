import os
import json
import argparse
from dotenv import load_dotenv
from google.cloud import bigquery
from shared.logging_config import setup_logger

logger = setup_logger("bigquery.loader")

# Load environment variables
load_dotenv()

def get_bq_client():
    """Initializes the BigQuery client using environment variables."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GCP_PROJECT_ID")
    
    if creds_path and os.path.exists(creds_path):
        logger.info(f"Initializing BigQuery Client using credentials file: {creds_path}")
        return bigquery.Client.from_service_account_json(creds_path, project=project_id)
    else:
        logger.info("Initializing BigQuery Client using default/ambient credentials.")
        return bigquery.Client(project=project_id)

def create_dataset_and_tables():
    """Creates the dataset and tables if they don't exist based on schema.json."""
    client = get_bq_client()
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    project = client.project
    
    dataset_ref = bigquery.DatasetReference(project, dataset_id)
    
    # Create dataset
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset '{project}.{dataset_id}' verified/created.")
    except Exception as e:
        logger.error(f"Failed to create dataset '{dataset_id}': {str(e)}")
        return False

    # Load schemas from schema.json
    schema_file_path = os.path.join(os.path.dirname(__file__), "schema.json")
    if not os.path.exists(schema_file_path):
        logger.error(f"Schema file not found at {schema_file_path}")
        return False
        
    with open(schema_file_path, "r") as f:
        schemas = json.load(f)
        
    for table_name, schema_fields in schemas.items():
        table_ref = dataset_ref.table(table_name)
        
        # Build BigQuery SchemaField objects
        bq_schema = []
        for field in schema_fields:
            bq_schema.append(bigquery.SchemaField(
                name=field["name"],
                field_type=field["type"],
                mode=field["mode"]
            ))
            
        table = bigquery.Table(table_ref, schema=bq_schema)
        try:
            table = client.create_table(table, exists_ok=True)
            logger.info(f"Table '{project}.{dataset_id}.{table_name}' verified/created.")
        except Exception as e:
            logger.error(f"Failed to create table '{table_name}': {str(e)}")
            return False
            
    return True

def load_file(table_name, file_path):
    """Loads a JSON Lines or CSV file into the specified table using a Load Job."""
    client = get_bq_client()
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    table_ref = client.dataset(dataset_id).table(table_name)
    
    logger.info(f"Loading '{file_path}' into BigQuery table '{dataset_id}.{table_name}'...")
    
    # Auto-detect format by file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        logger.info("Auto-detected CSV file format.")
        source_format = bigquery.SourceFormat.CSV
        skip_leading_rows = 1
    elif ext in [".json", ".jsonl"]:
        logger.info("Auto-detected JSON Lines file format.")
        source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        skip_leading_rows = 0
    else:
        logger.error(f"Unsupported file format extension '{ext}'. Only .csv, .json, or .jsonl files are accepted.")
        return False
        
    job_config = bigquery.LoadJobConfig(
        source_format=source_format,
        skip_leading_rows=skip_leading_rows,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    
    try:
        with open(file_path, "rb") as source_file:
            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
            
        # Wait for the load job to complete
        job.result()
        logger.info(f"Loaded {job.output_rows} rows into '{dataset_id}.{table_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error loading file to BigQuery: {str(e)}")
        return False

def insert_rows(table_name, rows):
    """Streams rows directly into BigQuery using streaming insert API."""
    client = get_bq_client()
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    table_ref = client.dataset(dataset_id).table(table_name)
    
    logger.info(f"Streaming {len(rows)} rows to BigQuery table '{dataset_id}.{table_name}'...")
    
    try:
        errors = client.insert_rows_json(table_ref, rows)
        if errors == []:
            logger.info(f"Successfully inserted {len(rows)} rows.")
            return True
        else:
            logger.error(f"Failed to stream rows. Errors: {errors}")
            return False
    except Exception as e:
        logger.error(f"Error during streaming insert: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigQuery Schema Manager and Loader")
    parser.add_argument("--setup", action="store_true", help="Create the BigQuery dataset and tables")
    parser.add_argument("--load-file", type=str, help="Path to local file (CSV/JSONLines) to load")
    parser.add_argument("--table", type=str, choices=["events", "churn_risk", "customer_features"], help="Table to load data into")
    
    args = parser.parse_args()
    
    if args.setup:
        success = create_dataset_and_tables()
        if not success:
            exit(1)
            
    if args.load_file:
        if not args.table:
            logger.error("--table is required when loading a file.")
            exit(1)
        success = load_file(args.table, args.load_file)
        if not success:
            exit(1)


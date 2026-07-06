import os
import io
import re
import csv
import json
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from shared.logging_config import setup_logger

logger = setup_logger("transform.transformer")

# Load environments
load_dotenv()

# Attempt to load GCS client
try:
    from gcs.uploader import get_storage_client
except ImportError:
    get_storage_client = None
    logger.warning("Could not import GCS client. Cloud operations will be unavailable.")

# Attempt to load BigQuery client for metrics persistence
try:
    from bigquery.loader import get_bq_client
except ImportError:
    get_bq_client = None
    logger.warning("Could not import BigQuery client. Metrics will only be saved locally.")

def parse_date(date_str):
    """Parses date string (YYYYMMDD or YYYY-MM-DD) into YYYYMMDD and YYYY-MM-DD formats."""
    # Strip non-digits to support both YYYYMMDD and YYYY-MM-DD
    cleaned = re.sub(r"\D", "", date_str)
    if len(cleaned) != 8:
        raise ValueError(f"Invalid date format: {date_str}. Must be YYYYMMDD or YYYY-MM-DD.")
        
    year = cleaned[:4]
    month = cleaned[4:6]
    day = cleaned[6:]
    return cleaned, f"{year}-{month}-{day}"

def load_data_local(date_yyyymmdd):
    """Loads events JSON and customer status CSV from local daily_streams directory."""
    streams_dir = os.path.join(os.path.dirname(__file__), "../output/daily_streams")
    
    events_path = os.path.join(streams_dir, f"events_{date_yyyymmdd}.json")
    status_path = os.path.join(streams_dir, f"customer_status_{date_yyyymmdd}.csv")
    
    if not os.path.exists(events_path) or not os.path.exists(status_path):
        raise FileNotFoundError(f"Local daily stream files missing for date {date_yyyymmdd} inside {streams_dir}.")
        
    logger.info(f"Loading local events from: {events_path}")
    events_df = pd.read_json(events_path, lines=True)
    
    logger.info(f"Loading local customer profiles from: {status_path}")
    profiles_df = pd.read_csv(status_path)
    
    return events_df, profiles_df

def load_data_gcs(date_yyyy_mm_dd, date_yyyymmdd, bucket_name):
    """Downloads events JSON and customer status CSV from GCS raw directory."""
    if not get_storage_client:
        raise ImportError("GCS Client is not loaded. Cannot execute cloud read.")
        
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    
    events_blob_name = f"raw/{date_yyyy_mm_dd}/events_{date_yyyymmdd}.json"
    status_blob_name = f"raw/{date_yyyy_mm_dd}/customer_status_{date_yyyymmdd}.csv"
    
    logger.info(f"Downloading raw events from GCS blob: {events_blob_name}")
    events_blob = bucket.blob(events_blob_name)
    if not events_blob.exists():
        raise FileNotFoundError(f"GCS Blob not found: gs://{bucket_name}/{events_blob_name}")
    events_data = events_blob.download_as_text()
    
    logger.info(f"Downloading raw profiles from GCS blob: {status_blob_name}")
    status_blob = bucket.blob(status_blob_name)
    if not status_blob.exists():
        raise FileNotFoundError(f"GCS Blob not found: gs://{bucket_name}/{status_blob_name}")
    status_data = status_blob.download_as_text()
    
    # Parse into DataFrames
    events_df = pd.read_json(io.StringIO(events_data), lines=True)
    profiles_df = pd.read_csv(io.StringIO(status_data))
    
    return events_df, profiles_df

def engineer_features(events_df, profiles_df):
    """
    Cleans raw inputs and engineers churn-relevant features:
    tenure, monthly_charges, contract_type, usage_trends, support_ticket_frequency,
    payment_method, and customer_lifetime_value.
    """
    logger.info("Starting feature engineering...")
    
    # 1. Input cleanup
    # Remove records without valid customer_id
    profiles_clean = profiles_df.dropna(subset=["customer_id"]).copy()
    if events_df.empty:
        events_clean = pd.DataFrame(columns=["customer_id", "event_type", "details"])
    else:
        events_clean = events_df.dropna(subset=["customer_id"]).copy()
        
    # 2. Extract daily logins (usage trends)
    logins = events_clean[events_clean["event_type"] == "login"].groupby("customer_id").size().rename("usage_trends")
    
    # 3. Extract daily support tickets count
    tickets = events_clean[events_clean["event_type"] == "support_ticket"].groupby("customer_id").size().rename("support_ticket_frequency")
    
    # 4. Extract payment method from signup details
    payment_methods = {}
    signups = events_clean[events_clean["event_type"] == "signup"]
    for _, row in signups.iterrows():
        try:
            details = json.loads(row["details"])
            if "payment_method" in details:
                payment_methods[row["customer_id"]] = details["payment_method"]
        except Exception:
            pass
            
    # 5. Build final dataset using customer profiles
    profiles_clean = profiles_clean.set_index("customer_id")
    
    # Append engineered features
    profiles_clean["usage_trends"] = logins
    profiles_clean["support_ticket_frequency"] = tickets
    profiles_clean["payment_method"] = profiles_clean.index.map(payment_methods)
    
    # Reset index and fill missing values
    output_df = profiles_clean.reset_index()
    output_df["usage_trends"] = output_df["usage_trends"].fillna(0).astype(int)
    output_df["support_ticket_frequency"] = output_df["support_ticket_frequency"].fillna(0).astype(int)
    output_df["payment_method"] = output_df["payment_method"].fillna("Mailed check") # Default fallback
    
    # Rename matching user requested structure
    output_df = output_df.rename(columns={
        "contract": "contract_type"
    })
    
    # Final column ordering check
    desired_cols = [
        "customer_id", "contract_type", "tenure", "monthly_charges", 
        "customer_lifetime_value", "usage_trends", "support_ticket_frequency", 
        "payment_method", "churn_status"
    ]
    
    # Match columns that actually exist
    final_cols = [col for col in desired_cols if col in output_df.columns]
    
    return output_df[final_cols]

def calculate_null_rates(df):
    """Calculates percentage of null rates per field in a DataFrame."""
    null_counts = df.isnull().sum()
    null_rates = (null_counts / len(df)) * 100
    return null_rates.to_dict()

def write_metrics_to_bq(date_str: str, records_in: int, records_out: int, avg_null_rate: float, run_duration: float):
    """
    Persists a single pipeline run row to the BigQuery `pipeline_metrics` table.
    Uses a delete-then-load-job upsert to prevent duplicate date entries.
    """
    if not get_bq_client:
        logger.warning("BigQuery client unavailable. Skipping BQ metrics write.")
        return

    try:
        from google.cloud import bigquery as bq

        project  = os.getenv("GCP_PROJECT_ID")
        dataset  = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
        table_id = "pipeline_metrics"
        full_ref  = f"{project}.{dataset}.{table_id}"

        client = get_bq_client()

        # Upsert: delete existing row for this date, then insert fresh
        client.query(f"DELETE FROM `{full_ref}` WHERE run_date = '{date_str}'").result()

        row = {
            "run_date":    date_str,
            "records_in":  records_in,
            "records_out": records_out,
            "null_rate":   round(avg_null_rate, 4),
            "run_duration": round(run_duration, 4) if run_duration is not None else None,
        }

        csv_buf   = io.StringIO()
        writer    = csv.DictWriter(csv_buf, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        csv_buf.seek(0)
        csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))

        job_config = bq.LoadJobConfig(
            source_format=bq.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=bq.WriteDisposition.WRITE_APPEND,
            schema=[
                bq.SchemaField("run_date",    "DATE",    mode="REQUIRED"),
                bq.SchemaField("records_in",  "INTEGER", mode="NULLABLE"),
                bq.SchemaField("records_out", "INTEGER", mode="NULLABLE"),
                bq.SchemaField("null_rate",   "FLOAT",   mode="NULLABLE"),
                bq.SchemaField("run_duration","FLOAT",   mode="NULLABLE"),
            ],
        )
        table_ref = client.dataset(dataset).table(table_id)
        client.load_table_from_file(csv_bytes, table_ref, job_config=job_config).result()
        logger.info(f"Persisted pipeline metrics for {date_str} to BigQuery table `{full_ref}`.")
    except Exception as e:
        logger.error(f"Failed to write metrics to BigQuery: {e}")


def write_metrics_log(date_str, raw_profile_cnt, raw_events_cnt, output_cnt, null_rates, local_only=True, bucket_name=None, run_duration=None):
    """Logs transform metadata, outputs a local metrics report file, and persists to BigQuery."""
    report = (
        f"===================================================\n"
        f"CHURN PIPELINE TRANSFORMATION METRICS REPORT\n"
        f"Date Simulated: {date_str}\n"
        f"Execution Time: {datetime.utcnow().isoformat()}Z\n"
        f"---------------------------------------------------\n"
        f"Input Profile Count: {raw_profile_cnt}\n"
        f"Input Event Count:   {raw_events_cnt}\n"
        f"Output Feature Row Count: {output_cnt}\n"
        f"---------------------------------------------------\n"
        f"Null Rate per Field (%):\n"
    )
    for col, pct in null_rates.items():
        report += f"  - {col}: {pct:.2f}%\n"
    report += f"===================================================\n"
    
    # Print metrics directly to terminal
    print(report)
    
    # Save locally
    metrics_dir = os.path.join(os.path.dirname(__file__), "../output/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    local_path = os.path.join(metrics_dir, f"metrics_{date_str}.log")
    with open(local_path, "w") as f:
        f.write(report)
    logger.info(f"Saved local metrics report at: {local_path}")

    # Always persist to BigQuery (regardless of local_only flag)
    avg_null = sum(null_rates.values()) / len(null_rates) if null_rates else 0.0
    write_metrics_to_bq(
        date_str=date_str,
        records_in=raw_profile_cnt + raw_events_cnt,
        records_out=output_cnt,
        avg_null_rate=avg_null,
        run_duration=run_duration or 0.0,
    )
    
    # Save to GCS if in cloud mode
    if not local_only and bucket_name and get_storage_client:
        try:
            client = get_storage_client()
            bucket = client.bucket(bucket_name)
            destination_blob = f"processed/{date_str}/metrics.log"
            blob = bucket.blob(destination_blob)
            blob.upload_from_string(report, content_type="text/plain")
            logger.info(f"Uploaded metrics report to GCS blob: {destination_blob}")
        except Exception as e:
            logger.error(f"Failed to upload metrics report to GCS: {str(e)}")

def process_daily_features(date_str, local_only=True, bucket_name=None):
    """
    Main coordinate function.
    Reads inputs, transforms features, logs metrics, and outputs clean dataset.
    """
    # Parse dates
    date_yyyymmdd, date_yyyy_mm_dd = parse_date(date_str)
    
    # 1. Load raw inputs
    try:
        if local_only:
            events_df, profiles_df = load_data_local(date_yyyymmdd)
        else:
            if not bucket_name:
                raise ValueError("Bucket name must be provided for GCS cloud execution.")
            events_df, profiles_df = load_data_gcs(date_yyyy_mm_dd, date_yyyymmdd, bucket_name)
    except Exception as e:
        logger.error(f"Failed to load raw source datasets: {str(e)}")
        return False
        
    raw_profile_cnt = len(profiles_df)
    raw_events_cnt = len(events_df)
    
    # 2. Engineer features
    output_df = engineer_features(events_df, profiles_df)
    
    # 3. Calculate null metrics
    null_rates = calculate_null_rates(output_df)
    output_cnt = len(output_df)
    
    # 4. Save metrics log
    write_metrics_log(
        date_yyyy_mm_dd, raw_profile_cnt, raw_events_cnt, 
        output_cnt, null_rates, local_only, bucket_name
    )
    
    # 5. Output processed features (CSV format)
    if local_only:
        processed_dir = os.path.join(os.path.dirname(__file__), "../output/processed")
        os.makedirs(processed_dir, exist_ok=True)
        local_output_path = os.path.join(processed_dir, f"processed_features_{date_yyyymmdd}.csv")
        output_df.to_csv(local_output_path, index=False)
        logger.info(f"Successfully saved clean processed features locally: {local_output_path}")
    else:
        try:
            client = get_storage_client()
            bucket = client.bucket(bucket_name)
            destination_csv = f"processed/{date_yyyy_mm_dd}/processed_features.csv"
            
            # Write DF to string buffer
            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer, index=False)
            
            blob = bucket.blob(destination_csv)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
            logger.info(f"Successfully uploaded processed features to GCS blob: {destination_csv}")
        except Exception as e:
            logger.error(f"Failed to upload processed dataset to GCS: {str(e)}")
            return False
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Churn Feature Engineering Transformer")
    parser.add_argument("--date", type=str, required=True, help="Target date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--local", action="store_true", help="Execute locally (reading/writing to disk instead of GCS)")
    parser.add_argument("--bucket", type=str, default=None, help="GCS Bucket name (overrides GCS_BUCKET_NAME in env)")
    args = parser.parse_args()
    
    bucket_name = args.bucket or os.getenv("GCS_BUCKET_NAME")
    
    # If not local and bucket name missing, raise
    if not args.local and not bucket_name:
        logger.error("Bucket name not provided. Set GCS_BUCKET_NAME in .env or use the --bucket flag, or use --local.")
        exit(1)
        
    success = process_daily_features(args.date, local_only=args.local, bucket_name=bucket_name)
    if not success:
        exit(1)

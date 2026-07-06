"""
backfill_pipeline_metrics.py
----------------------------
One-shot backfill script. Parses every metrics_YYYY-MM-DD.log file in
output/metrics/ and upserts (delete-then-insert) those rows into the
BigQuery `pipeline_metrics` table. Safe to re-run multiple times.

Usage:
    .venv/Scripts/python -m bigquery.backfill_pipeline_metrics
"""

import os
import re
import io
import csv
from datetime import datetime, date
from dotenv import load_dotenv
from shared.logging_config import setup_logger
from bigquery.loader import get_bq_client

load_dotenv()

logger = setup_logger("bigquery.backfill_pipeline_metrics")

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
DATASET     = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
TABLE       = "pipeline_metrics"
FULL_TABLE  = f"{PROJECT_ID}.{DATASET}.{TABLE}"

METRICS_DIR = os.path.join(os.path.dirname(__file__), "../output/metrics")


def parse_log_file(path: str) -> dict | None:
    """
    Parses a transformer metrics .log file and returns a dict matching
    the pipeline_metrics BigQuery schema:
        run_date, records_in, records_out, null_rate, run_duration
    """
    try:
        with open(path, "r") as f:
            content = f.read()

        run_date_m   = re.search(r"Date Simulated:\s+(\d{4}-\d{2}-\d{2})", content)
        exec_time_m  = re.search(r"Execution Time:\s+(\S+)", content)  # noqa: W605
        records_in_m = re.search(r"Input Profile Count:\s+(\d+)", content)
        records_out_m= re.search(r"Output Feature Row Count:\s+(\d+)", content)

        if not all([run_date_m, records_in_m, records_out_m]):
            logger.warning(f"Skipping {path}: could not parse required fields.")
            return None

        # Compute average null rate across all reported fields
        null_pcts = [float(x) for x in re.findall(r":\s+([\d.]+)%", content)]
        avg_null  = round(sum(null_pcts) / len(null_pcts), 4) if null_pcts else 0.0

        return {
            "run_date":    run_date_m.group(1),          # YYYY-MM-DD string → BQ DATE
            "records_in":  int(records_in_m.group(1)),
            "records_out": int(records_out_m.group(1)),
            "null_rate":   avg_null,
            "run_duration": None,                         # Not recorded in log files
        }
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
        return None


def backfill():
    client = get_bq_client()

    # 1. Discover all local log files
    if not os.path.isdir(METRICS_DIR):
        logger.error(f"Metrics directory not found: {METRICS_DIR}")
        return

    log_files = sorted(
        [f for f in os.listdir(METRICS_DIR) if f.startswith("metrics_") and f.endswith(".log")]
    )
    logger.info(f"Found {len(log_files)} local metrics log file(s): {log_files}")

    # 2. Parse each file into a row dict
    rows = []
    for fname in log_files:
        row = parse_log_file(os.path.join(METRICS_DIR, fname))
        if row:
            rows.append(row)
            logger.info(f"Parsed: {row}")

    if not rows:
        logger.warning("No valid rows to backfill. Exiting.")
        return

    # 3. For each parsed date, delete existing BQ row (upsert safety) then insert
    dates_to_backfill = [r["run_date"] for r in rows]
    dates_str = ", ".join(f"'{d}'" for d in dates_to_backfill)

    delete_sql = f"DELETE FROM `{FULL_TABLE}` WHERE run_date IN ({dates_str})"
    logger.info(f"Deleting any existing rows for dates: {dates_str}")
    client.query(delete_sql).result()
    logger.info("Delete complete.")

    # 4. Build CSV in-memory and load via Load Job (avoids streaming buffer lock)
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=["run_date", "records_in", "records_out", "null_rate", "run_duration"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    csv_buf.seek(0)
    csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))

    from google.cloud import bigquery as bq
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

    table_ref = client.dataset(DATASET).table(TABLE)
    job = client.load_table_from_file(csv_bytes, table_ref, job_config=job_config)
    job.result()
    logger.info(f"Backfill complete. Inserted {len(rows)} row(s) into `{FULL_TABLE}`.")

    # 5. Verify final state
    result = list(client.query(f"SELECT run_date, records_in, records_out, null_rate FROM `{FULL_TABLE}` ORDER BY run_date"))
    print(f"\n{'='*55}")
    print(f"pipeline_metrics — Final State ({len(result)} row(s))")
    print(f"{'='*55}")
    print(f"{'run_date':<14} {'records_in':>12} {'records_out':>12} {'null_rate':>10}")
    print("-"*55)
    for r in result:
        print(f"{str(r['run_date']):<14} {r['records_in']:>12} {r['records_out']:>12} {r['null_rate']:>10.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    backfill()

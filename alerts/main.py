import os
import re
import sys
import logging
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv
import functions_framework

# Self-contained logging config for Cloud Function deployment
def get_alerts_logger():
    logger = logging.getLogger("alerts_function")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [ALERTS] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_alerts_logger()

# Load environment variables
load_dotenv()

def get_bq_client():
    """Initializes the BigQuery client (self-contained for CF packaging limits)."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("GCP_PROJECT_ID")
    from google.cloud import bigquery
    if creds_path and os.path.exists(creds_path):
        return bigquery.Client.from_service_account_json(creds_path, project=project_id)
    return bigquery.Client(project=project_id)

def check_risk_and_alert(threshold=500.0, local_only=False):
    """
    Queries churn predictions exceeding threshold from latest run
    and dispatches aggregated Slack webhook or console alerts.
    """
    # 1. Fetch predictions
    if local_only:
        csv_path = os.path.join(os.path.dirname(__file__), "../output/churn_predictions.csv")
        if not os.path.exists(csv_path):
            logger.error(f"Local predictions file not found at: {csv_path}. Please run scoring first.")
            return False
        logger.info(f"Reading local predictions from: {csv_path}")
        df = pd.read_csv(csv_path)
        critical_df = df[df["revenue_at_risk"] >= threshold]
    else:
        # BQ cloud mode
        try:
            client = get_bq_client()
            dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
            table_ref = f"{client.project}.{dataset_id}.churn_predictions"
            logger.info(f"Querying predictions exceeding threshold ${threshold} from BQ table: {table_ref}")
            
            # Fetch predictions matching threshold from the latest predicted_date run
            query = f"""
                SELECT customer_id, churn_probability, revenue_at_risk, predicted_date
                FROM `{table_ref}`
                WHERE revenue_at_risk >= {threshold}
                  AND predicted_date = (SELECT MAX(predicted_date) FROM `{table_ref}`)
            """
            query_job = client.query(query)
            rows = [dict(row) for row in query_job]
            critical_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["customer_id", "churn_probability", "revenue_at_risk", "predicted_date"])
        except Exception as e:
            logger.error(f"Failed to query predictions from BigQuery: {str(e)}")
            return False
            
    # 2. Check if records found
    if critical_df.empty:
        logger.info(f"No customers exceeded the revenue-at-risk threshold of ${threshold:.2f}.")
        return True
        
    logger.warning(f"CRITICAL RISK: Found {len(critical_df)} customers exceeding threshold of ${threshold:.2f}!")
    
    # 3. Log details to stdout
    for _, row in critical_df.iterrows():
        logger.warning(
            f"ALERT: Customer {row['customer_id']} is high risk! "
            f"Probability: {row['churn_probability']:.1%}, "
            f"Revenue-at-Risk: ${row['revenue_at_risk']:.2f}"
        )
        
    # 4. Dispatch Alert (Slack webhook summary if configured, else console summary)
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("Slack Webhook URL not configured. Formatting console report summary:")
        summary = (
            f"\n===========================================================\n"
            f"!!! CRITICAL REVENUE RISK BATCH REPORT !!!\n"
            f"Threshold Filter: >= ${threshold:.2f}\n"
            f"Exceeded Accounts: {len(critical_df)}\n"
            f"-----------------------------------------------------------\n"
        )
        for _, row in critical_df.iterrows():
            summary += f"- Cust ID: {row['customer_id']} | Churn Prob: {row['churn_probability']:.1%} | Risk: ${row['revenue_at_risk']:.2f}\n"
        summary += (
            f"-----------------------------------------------------------\n"
            f"Intervention outreach recommended immediately.\n"
            f"==========================================================="
        )
        print(summary)
        return True
        
    # Dispatch slack summary block payload
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 Critical Revenue Risk Batch Report 🚨",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Found *{len(critical_df)}* customer(s) exceeding the revenue-at-risk threshold of *${threshold:.2f}* in the latest run."
            }
        },
        {"type": "divider"}
    ]
    
    # Append individual customer summaries
    for _, row in critical_df.head(8).iterrows():
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"• *Customer ID*: `{row['customer_id']}`\n  *Probability*: `{row['churn_probability']:.1%}` | *Revenue-at-Risk*: `${row['revenue_at_risk']:.2f}`"
            }
        })
        
    if len(critical_df) > 8:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_And {len(critical_df) - 8} more critical accounts._"
                }
            ]
        })
        
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Open BigQuery Console",
                    "emoji": True
                },
                "url": "https://console.cloud.google.com/bigquery"
            }
        ]
    })
    
    payload = {"blocks": blocks}
    
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            logger.info("Slack summary webhook dispatched successfully.")
            return True
        else:
            logger.error(f"Slack webhook returned status {response.status_code}. Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to post summary alert to Slack: {str(e)}")
        return False

# Legacy process_alert for backward compatibility imports
def process_alert(data):
    """Business logic for single-record alert ingestion."""
    customer_id = data.get("customer_id")
    churn_prob = data.get("churn_probability", 0.0)
    contract_val = data.get("contract_value", 0.0)
    rev_at_risk = data.get("revenue_at_risk", 0.0)
    
    if not customer_id:
        return False
        
    logger.warning(
        f"!!! INDIVIDUAL RISK ALERT !!!\n"
        f"Customer ID: {customer_id} | Churn Prob: {churn_prob:.1%} | Risk: ${rev_at_risk:.2f}"
    )
    return True

@functions_framework.http
def alerts_http_trigger(request):
    """
    HTTP entry point for GCP Cloud Function.
    Accepts GET/POST. If POST with JSON threshold key, parses it, else uses default.
    """
    threshold = 500.0
    
    if request.method == "POST":
        request_json = request.get_json(silent=True)
        if request_json and "threshold" in request_json:
            try:
                threshold = float(request_json["threshold"])
            except ValueError:
                pass
                
    success = check_risk_and_alert(threshold=threshold, local_only=False)
    if success:
        return ({"status": "success", "message": "Alert processing complete"}, 200)
    else:
        return ({"status": "error", "message": "Failed to check alerts"}, 500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Batch Alerts Checker Trigger")
    parser.add_argument("--threshold", type=float, default=500.0, help="Revenue-at-risk alert threshold (default: $500)")
    parser.add_argument("--local", action="store_true", help="Scan local predictions CSV file")
    args = parser.parse_args()
    
    success = check_risk_and_alert(threshold=args.threshold, local_only=args.local)
    if not success:
        sys.exit(1)

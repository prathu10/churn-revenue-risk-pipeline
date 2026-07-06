import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from shared.logging_config import setup_logger

logger = setup_logger("ml.predict")

# Load environment variables
load_dotenv()

# We import BigQuery loader function to save predictions
try:
    from bigquery.loader import insert_rows
except ImportError:
    logger.warning("Could not import bigquery.loader. Data ingestion will fall back to local outputs.")
    insert_rows = None

def get_active_customers(n_samples=50):
    """
    Simulates getting active customer features for scoring.
    In a real system, this would query BigQuery events.
    """
    logger.info("Fetching active customer features...")
    np.random.seed(10)
    
    customer_ids = [f"CUST_{1000 + i}" for i in range(n_samples)]
    support_tickets = np.random.poisson(lam=0.8, size=n_samples)
    login_frequency = np.random.randint(2, 11, size=n_samples)
    contract_value = np.random.uniform(30.0, 450.0, size=n_samples)
    days_since_last_login = np.random.poisson(lam=2.5, size=n_samples)
    
    # Introduce a few outlier high-risk customers manually
    support_tickets[0] = 5
    login_frequency[0] = 1
    days_since_last_login[0] = 14
    
    support_tickets[1] = 4
    login_frequency[1] = 2
    days_since_last_login[1] = 8
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "support_tickets_count": support_tickets,
        "login_frequency": login_frequency,
        "contract_value": contract_value,
        "days_since_last_login": days_since_last_login
    })
    return df

def generate_risk_scores():
    """Loads model, scores active customers, calculates revenue-at-risk, and writes output."""
    model_path = os.path.join(os.path.dirname(__file__), "churn_model.joblib")
    
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please run train.py first!")
        sys.exit(1)
        
    # Load model
    clf = joblib.load(model_path)
    logger.info("Successfully loaded churn classification model.")
    
    # Get features to score
    df = get_active_customers()
    
    features = ["support_tickets_count", "login_frequency", "contract_value", "days_since_last_login"]
    X = df[features]
    
    # Run predictions
    probs = clf.predict_proba(X)[:, 1]
    
    # Assemble results
    df["churn_probability"] = np.round(probs, 4)
    df["revenue_at_risk"] = np.round(df["churn_probability"] * df["contract_value"], 2)
    df["contract_value"] = np.round(df["contract_value"], 2)
    
    # Risk segmentation
    # High: prob >= 0.65, Medium: 0.3 <= prob < 0.65, Low: prob < 0.3
    df["risk_segment"] = pd.cut(
        df["churn_probability"],
        bins=[-0.01, 0.30, 0.65, 1.01],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    
    df["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # Output file
    predictions_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(predictions_dir, exist_ok=True)
    csv_path = os.path.join(predictions_dir, "churn_risk_scores.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved locally to {csv_path}")
    
    # Ingest to BigQuery if enabled
    bq_enabled = os.getenv("BIGQUERY_DATASET") is not None
    if bq_enabled and insert_rows:
        # Convert df to list of dicts matching schema
        bq_rows = df.to_dict(orient="records")
        # Ensure correct column types
        for row in bq_rows:
            row["churn_probability"] = float(row["churn_probability"])
            row["contract_value"] = float(row["contract_value"])
            row["revenue_at_risk"] = float(row["revenue_at_risk"])
            row["support_tickets_count"] = int(row["support_tickets_count"])
            
        table_name = os.getenv("BIGQUERY_TABLE_CHURN_RISK", "churn_risk")
        success = insert_rows(table_name, bq_rows)
        if success:
            logger.info("Successfully ingested churn predictions into BigQuery.")
        else:
            logger.warning("Failed to ingest churn predictions into BigQuery.")
            
    # Trigger local alerting check for High Risk
    trigger_alerts(df)
    
    return df

def trigger_alerts(df):
    """Checks for severe revenue-at-risk breaches and notifies via alerting module."""
    # Find customers: High risk and contract_value > 150
    alert_condition = (df["risk_segment"] == "High") & (df["revenue_at_risk"] > 100.0)
    critical_customers = df[alert_condition]
    
    if not critical_customers.empty:
        logger.warning(f"CRITICAL RISK: Found {len(critical_customers)} customers with high churn risk and high revenue-at-risk!")
        
        # Import alerts dynamically
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "../alerts"))
            from main import process_alert
            
            for _, customer in critical_customers.iterrows():
                alert_payload = {
                    "customer_id": customer["customer_id"],
                    "churn_probability": customer["churn_probability"],
                    "revenue_at_risk": customer["revenue_at_risk"],
                    "contract_value": customer["contract_value"]
                }
                process_alert(alert_payload)
        except Exception as e:
            logger.error(f"Failed to trigger alert notification: {str(e)}")
    else:
        logger.info("No critical churn alert thresholds breached.")

if __name__ == "__main__":
    generate_risk_scores()

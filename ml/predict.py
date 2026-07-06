import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from bigquery.loader import get_bq_client, insert_rows
from shared.logging_config import setup_logger

logger = setup_logger("ml.predict")

# Load environment variables
load_dotenv()

def pull_features_from_bq():
    """Queries BigQuery customer_features table and loads it into a Pandas DataFrame."""
    client = get_bq_client()
    project_id = client.project
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    table_ref = f"{project_id}.{dataset_id}.customer_features"
    
    logger.info(f"Querying features to score from BigQuery table '{table_ref}'...")
    # Fetch the latest features for each customer to score
    # (or simply score all rows currently in the features table)
    query = f"SELECT * FROM `{table_ref}`"
    query_job = client.query(query)
    
    # Iterate and construct DataFrame to avoid db-dtypes package dependency
    rows = [dict(row) for row in query_job]
    if not rows:
        raise ValueError(f"No records found in BigQuery table {table_ref}.")
        
    df = pd.DataFrame(rows)
    logger.info(f"Successfully loaded {len(df)} records from BigQuery for scoring.")
    return df

def pull_features_local():
    """Loads and merges all local processed CSV files from the output directory for scoring."""
    processed_dir = os.path.join(os.path.dirname(__file__), "../output/processed")
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Local processed directory not found at: {processed_dir}")
        
    files = [
        os.path.join(processed_dir, f) 
        for f in os.listdir(processed_dir) 
        if f.startswith("processed_features_") and f.endswith(".csv")
    ]
    
    if not files:
        raise FileNotFoundError("No processed feature files found inside output/processed/.")
        
    logger.info(f"Merging and loading {len(files)} local feature files for scoring...")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Successfully loaded {len(df)} records from local files for scoring.")
    return df

def run_predictions_pipeline(local_only=False):
    """Loads fitted model, scores customers, calculates revenue risk, ranks and writes predictions."""
    model_path = os.path.join(os.path.dirname(__file__), "churn_model.joblib")
    
    # 1. Load trained model
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please run train.py first!")
        sys.exit(1)
        
    pipeline = joblib.load(model_path)
    logger.info("Successfully loaded churn model pipeline.")
    
    # 2. Get customer features
    if local_only:
        df = pull_features_local()
    else:
        df = pull_features_from_bq()
        
    # Check if features are empty
    if len(df) == 0:
        logger.warning("No customers found for scoring. Aborting pipeline.")
        return False
        
    # 3. Predict churn probabilities
    # The pipeline automatically handles One-Hot Encoding via its preprocessor step
    features = [
        "contract_type", "tenure", "monthly_charges", 
        "customer_lifetime_value", "usage_trends", "support_ticket_frequency", "payment_method"
    ]
    
    # Check that features match columns in X
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.error(f"Missing expected columns in features dataset: {missing}")
        return False
        
    X = df[features]
    
    # Get probability of the positive class (Churned)
    probs = pipeline.predict_proba(X)[:, 1]
    
    # 4. Calculate revenue-at-risk
    df["churn_probability"] = np.round(probs, 4)
    df["revenue_at_risk"] = np.round(df["churn_probability"] * df["customer_lifetime_value"], 2)
    
    # 5. Rank by revenue_at_risk descending
    df_ranked = df.sort_values(by="revenue_at_risk", ascending=False).copy()
    
    # 6. Save results
    predicted_date = datetime.utcnow().isoformat() + "Z"
    
    # Output schema: customer_id, churn_probability, revenue_at_risk, predicted_date
    results_df = pd.DataFrame({
        "customer_id": df_ranked["customer_id"],
        "churn_probability": df_ranked["churn_probability"],
        "revenue_at_risk": df_ranked["revenue_at_risk"],
        "predicted_date": predicted_date
    })
    
    if local_only:
        predictions_dir = os.path.join(os.path.dirname(__file__), "../output")
        os.makedirs(predictions_dir, exist_ok=True)
        local_output_path = os.path.join(predictions_dir, "churn_predictions.csv")
        results_df.to_csv(local_output_path, index=False)
        logger.info(f"Successfully saved ranked churn predictions locally: {local_output_path}")
        print("\n" + "="*50)
        print("TOP 10 LOCAL CUSTOMERS BY REVENUE RISK RANKING")
        print(results_df.head(10).to_string(index=False))
        print("="*50 + "\n")
    else:
        # Ingest into BigQuery churn_predictions table
        bq_rows = results_df.to_dict(orient="records")
        # Ensure correct column typing
        for row in bq_rows:
            row["churn_probability"] = float(row["churn_probability"])
            row["revenue_at_risk"] = float(row["revenue_at_risk"])
            
        logger.info(f"Streaming {len(bq_rows)} scored churn predictions into BigQuery...")
        success = insert_rows("churn_predictions", bq_rows)
        if success:
            logger.info("Successfully ingested predictions into 'churn_predictions' BigQuery table.")
            print("\n" + "="*50)
            print("TOP 10 SCORDED CUSTOMERS STREAMED TO BIGQUERY")
            print(results_df.head(10).to_string(index=False))
            print("="*50 + "\n")
        else:
            logger.error("Failed to stream predictions into BigQuery.")
            return False
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Ingestion & Scoring Engine")
    parser.add_argument("--local", action="store_true", help="Score using local processed feature CSVs")
    args = parser.parse_args()
    
    run_predictions_pipeline(local_only=args.local)

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from dotenv import load_dotenv
from bigquery.loader import get_bq_client
from shared.logging_config import setup_logger

logger = setup_logger("ml.train")

# Load environment variables
load_dotenv()

"""
Baseline Model Choice Explanation:
Given the small dataset size (~108 rows), a shallow Random Forest Classifier (max_depth=4) is an excellent baseline choice:
1. Low Risk of Overfitting: Restricted trees (shallow depth) limit complexity and prevent high-variance fitting on small datasets.
2. Handling Categorical Variables: Integrated with scikit-learn's ColumnTransformer and OneHotEncoder, it handles different categorical values and missing fallbacks smoothly.
3. No Feature Scaling Required: Unlike SVMs, KNN, or Neural Networks, Tree-based models are scale-invariant, allowing tenure, CLV, and monthly charges to be processed natively.

Recommendations to Improve with More Data:
1. Feature Engineering: Incorporate historical rolling averages (slopes/trends of activity) instead of single-day count snapshots.
2. Imbalance Management: If churn rates are low, use class-weight balancing (class_weight="balanced") or resampling techniques (SMOTE).
3. Advanced Models: Transition to gradient-boosted trees (e.g., XGBoost, LightGBM, CatBoost) once the dataset exceeds 1,000+ records.
4. Hyperparameter Tuning: Use cross-validated grid search (GridSearchCV) to tune model parameters.
"""

def pull_data_from_bq():
    """Queries BigQuery customer_features table and loads it into a Pandas DataFrame."""
    client = get_bq_client()
    project_id = client.project
    dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")
    table_ref = f"{project_id}.{dataset_id}.customer_features"
    
    logger.info(f"Querying features from BigQuery table '{table_ref}'...")
    query = f"SELECT * FROM `{table_ref}`"
    query_job = client.query(query)
    
    # Iterate and construct DataFrame to avoid db-dtypes package dependency
    rows = [dict(row) for row in query_job]
    if not rows:
        raise ValueError(f"No records found in BigQuery table {table_ref}.")
        
    df = pd.DataFrame(rows)
    logger.info(f"Successfully loaded {len(df)} records from BigQuery.")
    return df

def pull_data_local():
    """Loads and merges all local processed CSV files from the output directory."""
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
        
    logger.info(f"Merging and loading {len(files)} local feature files...")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Successfully loaded {len(df)} records from local files.")
    return df

def train_model(local_only=False):
    """Fits baseline model, prints validation metrics, and stores pipeline artifact."""
    # 1. Pull data
    if local_only:
        df = pull_data_local()
    else:
        df = pull_data_from_bq()
        
    # 2. Check for minimum sample size
    if len(df) < 5:
        raise ValueError(f"Insufficient data for training. Found only {len(df)} samples.")
        
    # 3. Separate features and target
    # Map target churn_status (Active vs Churned) to binary
    if "churn_status" not in df.columns:
        raise KeyError("Target column 'churn_status' is missing from features dataset.")
        
    df["target"] = (df["churn_status"] == "Churned").astype(int)
    
    features = [
        "contract_type", "tenure", "monthly_charges", 
        "customer_lifetime_value", "usage_trends", "support_ticket_frequency", "payment_method"
    ]
    
    # Keep only columns that exist
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df["target"]
    
    logger.info(f"Training features selected: {features}")
    
    # 4. Build preprocessing pipeline
    categorical_features = ["contract_type", "payment_method"]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    numeric_features = [f for f in X.columns if f not in categorical_features]
    
    # Pipeline transformation configuration
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough" # Leave numeric columns untouched
    )
    
    # Restricted tree depth with balanced class weight configuration to prevent overfitting on small samples
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, class_weight='balanced')
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    # 5. Split train/test (stratified if multiple classes exist)
    unique_classes = np.unique(y)
    stratify_target = y if len(unique_classes) > 1 and (y.value_counts() > 1).all() else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_target
    )
    
    # Log the exact count of churned vs active customers in both splits
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")
    logger.info(f"Train set class distribution: {train_counts} (0=Active, 1=Churned)")
    logger.info(f"Test set class distribution:  {test_counts} (0=Active, 1=Churned)")
    
    # 6. Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluate Model
    probs = pipeline.predict_proba(X_test)[:, 1] if len(unique_classes) > 1 else np.zeros_like(y_test)
    
    # Default 0.5 threshold evaluation
    preds_05 = (probs >= 0.5).astype(int)
    precision_05, recall_05, f1_05, _ = precision_recall_fscore_support(y_test, preds_05, average="binary", zero_division=0)
    cm_05 = confusion_matrix(y_test, preds_05)
    
    # Custom 0.3 threshold evaluation
    preds_03 = (probs >= 0.3).astype(int)
    precision_03, recall_03, f1_03, _ = precision_recall_fscore_support(y_test, preds_03, average="binary", zero_division=0)
    cm_03 = confusion_matrix(y_test, preds_03)
    
    # ROC AUC requires both classes in the test partition
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = 1.0 # default fallback
        
    print("\n" + "="*50)
    print("BASELINE CHURN MODEL EVALUATION RESULTS")
    print(f"Test Partition Size: {len(y_test)}")
    print("-"*50)
    print("Default Threshold (0.5) Metrics:")
    print(f"  Precision (Churned Class): {precision_05:.4f}")
    print(f"  Recall (Churned Class):    {recall_05:.4f}")
    print(f"  F1 Score (Churned Class):  {f1_05:.4f}")
    print("  Confusion Matrix:")
    print(f"    {cm_05[0].tolist()}\n    {cm_05[1].tolist()}")
    print("-"*50)
    print("Custom Threshold (0.3) Metrics (RECOMMENDED):")
    print(f"  Precision (Churned Class): {precision_03:.4f}")
    print(f"  Recall (Churned Class):    {recall_03:.4f}")
    print(f"  F1 Score (Churned Class):  {f1_03:.4f}")
    print("  Confusion Matrix:")
    print(f"    {cm_03[0].tolist()}\n    {cm_03[1].tolist()}")
    print("-"*50)
    print(f"ROC-AUC Score (Threshold Independent): {auc:.4f}")
    print("="*50 + "\n")
    
    logger.info("Classification Report (Default 0.5 Threshold):\n" + classification_report(y_test, preds_05, zero_division=0))
    logger.info("Classification Report (Custom 0.3 Threshold):\n" + classification_report(y_test, preds_03, zero_division=0))
    
    # 8. Save Pipeline Model
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "churn_model.joblib")
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved complete processing & classifier pipeline to: {model_path}")
    
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Churn Model Ingestion & Training")
    parser.add_argument("--local", action="store_true", help="Execute locally merging processed output files")
    args = parser.parse_args()
    
    train_model(local_only=args.local)

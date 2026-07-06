-- DDL definitions for churn_pipeline BigQuery Dataset

-- 1. raw_events table storing raw event streams
CREATE TABLE IF NOT EXISTS `GCP_PROJECT_ID.churn_pipeline.raw_events` (
    event_id STRING NOT NULL,
    customer_id STRING NOT NULL,
    event_type STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    value FLOAT64,
    device STRING,
    details JSON
);

-- 2. customer_features table storing engineered features
CREATE TABLE IF NOT EXISTS `GCP_PROJECT_ID.churn_pipeline.customer_features` (
    customer_id STRING NOT NULL,
    contract_type STRING,
    tenure INT64,
    monthly_charges FLOAT64,
    customer_lifetime_value FLOAT64,
    usage_trends INT64,
    support_ticket_frequency INT64,
    payment_method STRING,
    churn_status STRING
);

-- 3. churn_predictions table storing model runs and scoring logs
CREATE TABLE IF NOT EXISTS `GCP_PROJECT_ID.churn_pipeline.churn_predictions` (
    customer_id STRING NOT NULL,
    churn_probability FLOAT64 NOT NULL,
    revenue_at_risk FLOAT64 NOT NULL,
    predicted_date TIMESTAMP NOT NULL
);

-- 4. pipeline_metrics table logging ETL run-time and statistical metrics
CREATE TABLE IF NOT EXISTS `GCP_PROJECT_ID.churn_pipeline.pipeline_metrics` (
    run_date DATE NOT NULL,
    records_in INT64,
    records_out INT64,
    null_rate FLOAT64,
    run_duration FLOAT64
);

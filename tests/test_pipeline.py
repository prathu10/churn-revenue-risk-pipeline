import os
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock

# Mock Google Cloud clients before they get instantiated at import time
import google.cloud.storage
import google.cloud.bigquery
google.cloud.storage.Client = MagicMock()
google.cloud.bigquery.Client = MagicMock()

# Import system modules
from shared.logging_config import setup_logger
from data_sim.generator import generate_customers, generate_event, CUSTOMER_PERSONAS
from functions.main import transform_event

from ml.train import generate_historical_dataset
from ml.predict import get_active_customers
from alerts.main import process_alert

def test_shared_logging():
    logger = setup_logger("test.logger")
    assert logger is not None
    assert logger.name == "test.logger"

def test_data_sim_generator():
    # Clear and regenerate customers
    CUSTOMER_PERSONAS.clear()
    generate_customers(10)
    assert len(CUSTOMER_PERSONAS) == 10
    
    # Check shape of customer profile
    cust_id = list(CUSTOMER_PERSONAS.keys())[0]
    profile = CUSTOMER_PERSONAS[cust_id]
    assert "persona" in profile
    assert profile["persona"] in ["stable", "high_risk", "new"]
    assert "contract_value" in profile
    assert profile["contract_value"] > 0
    
    # Generate an event
    event = generate_event(cust_id)
    assert event["customer_id"] == cust_id
    assert "event_id" in event
    assert "event_type" in event
    assert "timestamp" in event
    assert "details" in event

def test_cf_transform_event():
    # Valid event
    valid_raw = {
        "event_id": "evt123",
        "customer_id": "cust456",
        "event_type": "login",
        "timestamp": "2026-07-06T12:00:00Z",
        "value": 15.5,
        "device": "desktop",
        "details": {"session_id": "xyz"}
    }
    
    cleaned = transform_event(valid_raw)
    assert cleaned is not None
    assert cleaned["event_id"] == "evt123"
    assert cleaned["value"] == 15.5
    # details dict must be serialized to JSON string for BQ load
    assert isinstance(cleaned["details"], str)
    assert "xyz" in cleaned["details"]
    
    # Invalid event - missing customer_id
    invalid_raw = {
        "event_id": "evt123",
        "event_type": "login",
        "timestamp": "2026-07-06T12:00:00Z"
    }
    assert transform_event(invalid_raw) is None

def test_ml_historical_data_gen():
    df = generate_historical_dataset(100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    expected_cols = ["support_tickets_count", "login_frequency", "contract_value", "days_since_last_login", "churned"]
    for col in expected_cols:
        assert col in df.columns

def test_ml_predict_active_customers():
    df = get_active_customers(20)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    expected_cols = ["customer_id", "support_tickets_count", "login_frequency", "contract_value", "days_since_last_login"]
    for col in expected_cols:
        assert col in df.columns

def test_alerts_process():
    # Test alert logic execution with fallback logging
    alert_data = {
        "customer_id": "CUST_9999",
        "churn_probability": 0.85,
        "revenue_at_risk": 150.0,
        "contract_value": 200.0
    }
    success = process_alert(alert_data)
    assert success is True

from data_sim.daily_event_simulator import calculate_clv, run_simulation

def test_clv_calculation():
    # Month-to-month: tenure = 10, monthly = 50.0. CLV = 10 * 50 = 500.0
    assert calculate_clv("Month-to-month", 10, 50.0) == 500.0
    
    # One year: tenure = 5, monthly = 100.0. CLV = 5 * 100 + 12 * 100 * 0.9 = 500 + 1080 = 1580.0
    assert calculate_clv("One year", 5, 100.0) == 1580.0
    
    # Two year: tenure = 2, monthly = 150.0. CLV = 2 * 150 + 24 * 150 * 0.8 = 300 + 2880 = 3180.0
    assert calculate_clv("Two year", 2, 150.0) == 3180.0

def test_daily_event_simulator(tmp_path):
    output_dir = str(tmp_path)
    
    # Run a small 2-day simulation with 2 signups per day
    run_simulation(days=2, signups_per_day=2, output_dir=output_dir)
    
    files = os.listdir(output_dir)
    assert len(files) > 0
    
    # Verify we have both events and status files
    event_files = [f for f in files if f.startswith("events_")]
    status_files = [f for f in files if f.startswith("customer_status_")]
    
    assert len(event_files) == 2
    assert len(status_files) == 2
    
    # Verify the customer status file structure
    status_path = os.path.join(output_dir, status_files[0])
    status_df = pd.read_csv(status_path)
    assert "customer_lifetime_value" in status_df.columns
    assert "churn_status" in status_df.columns
    assert len(status_df) > 0

from gcs.daily_uploader import parse_date_from_filename, upload_daily_files
from gcs.verify_uploader import format_size, list_bucket_contents

def test_date_parsing_from_filename():
    assert parse_date_from_filename("events_20260706.json") == "2026-07-06"
    assert parse_date_from_filename("customer_status_20260706.csv") == "2026-07-06"
    assert parse_date_from_filename("customer_status_123.csv") is None
    assert parse_date_from_filename("invalid_file.json") is None

def test_daily_uploader_mock_run(tmp_path):
    output_dir = str(tmp_path)
    
    # Create fake daily files
    with open(os.path.join(output_dir, "events_20260706.json"), "w") as f:
        f.write("{}\n")
    with open(os.path.join(output_dir, "customer_status_20260706.csv"), "w") as f:
        f.write("customer_id,churn_status\n")
        
    # Runs using global mock GCS client
    success = upload_daily_files(output_dir, "mock-bucket")
    assert success is True

def test_verify_uploader_format_size():
    assert format_size(500) == "500.00 B"
    assert format_size(1500) == "1.46 KB"
    assert format_size(2000000) == "1.91 MB"

from transform.transformer import parse_date, engineer_features, calculate_null_rates

def test_transform_parse_date():
    assert parse_date("20260706") == ("20260706", "2026-07-06")
    assert parse_date("2026-07-06") == ("20260706", "2026-07-06")
    with pytest.raises(ValueError):
        parse_date("invalid-date")

def test_transform_engineer_features():
    # Setup mock event data
    events_data = [
        {"customer_id": "C_1", "event_type": "login", "details": "{}"},
        {"customer_id": "C_1", "event_type": "login", "details": "{}"},
        {"customer_id": "C_1", "event_type": "support_ticket", "details": '{"topic": "billing"}'},
        {"customer_id": "C_2", "event_type": "signup", "details": '{"payment_method": "Electronic check"}'},
        {"customer_id": "C_2", "event_type": "login", "details": "{}"}
    ]
    events_df = pd.DataFrame(events_data)
    
    # Setup mock profile data
    profiles_data = [
        {"customer_id": "C_1", "contract": "Month-to-month", "tenure": 12, "monthly_charges": 50.0, "customer_lifetime_value": 600.0, "churn_status": "Active"},
        {"customer_id": "C_2", "contract": "Two year", "tenure": 1, "monthly_charges": 100.0, "customer_lifetime_value": 2220.0, "churn_status": "Active"}
    ]
    profiles_df = pd.DataFrame(profiles_data)
    
    out_df = engineer_features(events_df, profiles_df)
    assert len(out_df) == 2
    
    # Check index mapping
    c1 = out_df[out_df["customer_id"] == "C_1"].iloc[0]
    c2 = out_df[out_df["customer_id"] == "C_2"].iloc[0]
    
    assert c1["usage_trends"] == 2
    assert c1["support_ticket_frequency"] == 1
    assert c1["payment_method"] == "Mailed check" # default fallback
    
    assert c2["usage_trends"] == 1
    assert c2["support_ticket_frequency"] == 0
    assert c2["payment_method"] == "Electronic check" # from details
    
    assert c1["contract_type"] == "Month-to-month"
    assert c2["contract_type"] == "Two year"

def test_transform_calculate_null_rates():
    test_df = pd.DataFrame({
        "col1": [1, None, 3, None],
        "col2": ["a", "b", "c", "d"]
    })
    rates = calculate_null_rates(test_df)
    assert rates["col1"] == 50.0
    assert rates["col2"] == 0.0

from bigquery.load_features import parse_input_counts_from_log, load_features_pipeline, execute_ddl_setup

def test_bigquery_parse_log_counts():
    mock_log = (
        "Input Profile Count: 45\n"
        "Input Event Count: 120\n"
        "Output Feature Row Count: 45\n"
    )
    assert parse_input_counts_from_log(mock_log) == 165
    assert parse_input_counts_from_log("invalid report data") == 0

def test_bigquery_load_features_mock_run():
    # Fake processed CSV
    df = pd.DataFrame({
        "customer_id": ["C_10"],
        "contract_type": ["One year"],
        "tenure": [15],
        "monthly_charges": [85.5],
        "customer_lifetime_value": [2200.0],
        "usage_trends": [5],
        "support_ticket_frequency": [0],
        "payment_method": ["Electronic check"],
        "churn_status": ["Active"]
    })
    
    # Configure mock client properties
    mock_client = google.cloud.bigquery.Client.from_service_account_json()
    mock_client.project = "mock-project"
    mock_client.insert_rows_json.return_value = []
    
    # Mock load data source
    import bigquery.load_features
    original_load = bigquery.load_features.load_data_source
    bigquery.load_features.load_data_source = MagicMock(return_value=(df, 35))
    
    success = load_features_pipeline("2026-07-06", local_only=True, bucket_name="mock-bucket")
    assert success is True
    
    # Restore original function
    bigquery.load_features.load_data_source = original_load

def test_bigquery_setup_mock_run():
    # Configure mock client properties
    mock_client = google.cloud.bigquery.Client.from_service_account_json()
    mock_client.project = "mock-project"
    
    # Since BQ client is mocked globally, this should run successfully
    success = execute_ddl_setup()
    assert success is True






import os
import sys
import json
import random
import requests
import argparse
import pandas as pd
from datetime import datetime, timedelta
from shared.logging_config import setup_logger

logger = setup_logger("data_sim.daily_event_simulator")

DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "Telco-Customer-Churn.csv")

def download_dataset():
    """Downloads the IBM Telco Customer Churn dataset if not cached locally."""
    if os.path.exists(LOCAL_CSV_PATH):
        logger.info(f"Dataset already exists locally at: {LOCAL_CSV_PATH}")
        return True
        
    logger.info(f"Downloading Telco Customer Churn dataset from: {DATASET_URL}")
    try:
        response = requests.get(DATASET_URL, timeout=15)
        response.raise_for_status()
        with open(LOCAL_CSV_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Dataset downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return False

def calculate_clv(contract, tenure, monthly_charges):
    """
    Calculates the Customer Lifetime Value (CLV) based on contract, tenure, and monthly charges.
    
    Formula:
      CLV = (tenure * monthly_charges) + contract_multiplier
    """
    try:
        tenure = float(tenure)
        monthly_charges = float(monthly_charges)
    except (ValueError, TypeError):
        return 0.0

    # Historical spend
    historical_spend = tenure * monthly_charges
    
    # Contract commitment bonus (projected CLV based on commitment stability)
    if contract == "One year":
        commitment_bonus = 12 * monthly_charges * 0.9
    elif contract == "Two year":
        commitment_bonus = 24 * monthly_charges * 0.8
    else: # Month-to-month
        commitment_bonus = 0.0
        
    return round(historical_spend + commitment_bonus, 2)

def run_simulation(days=7, signups_per_day=3, output_dir=None):
    """
    Runs the daily event stream simulation.
    Takes static dataset customer profiles and steps day-by-day to simulate events and signups.
    """
    if not download_dataset():
        logger.error("Simulation aborted due to missing dataset.")
        return
        
    # Read the dataset
    df = pd.read_csv(LOCAL_CSV_PATH)
    
    # Ensure TotalCharges is numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.replace(" ", ""), errors="coerce").fillna(0.0)
    
    # Split into active and churned customers pools for characteristics sampling
    active_pool = df[df["Churn"] == "No"].to_dict(orient="records")
    churned_pool = df[df["Churn"] == "Yes"].to_dict(orient="records")
    
    logger.info(f"Loaded {len(df)} templates. Active pool: {len(active_pool)}, Churned pool: {len(churned_pool)}")
    
    # Initialize monitored state with 30 initial active customers
    monitored_customers = {}
    initial_sample = random.sample(active_pool, min(30, len(active_pool)))
    
    for c in initial_sample:
        customer_id = c["customerID"]
        monitored_customers[customer_id] = {
            "customer_id": customer_id,
            "contract": c["Contract"],
            "tenure": int(c["tenure"]),
            "monthly_charges": float(c["MonthlyCharges"]),
            "churn_status": "Active",
            "days_monitored": 0,
            "support_tickets": 0,
            "logins": 0
        }
        
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "../output/daily_streams")
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime.now() - timedelta(days=days)
    logger.info(f"Simulating {days} days of data starting from {start_date.strftime('%Y-%m-%d')}...")
    
    for d in range(days):
        current_day = start_date + timedelta(days=d)
        date_str = current_day.strftime("%Y%m%d")
        logger.info(f"--- Simulating Day {d + 1}/{days} ({current_day.strftime('%Y-%m-%d')}) ---")
        
        day_events = []
        
        # 1. Process New Customer Signups
        for _ in range(signups_per_day):
            # Sample demographics from either pool to simulate new registration
            template = random.choice(active_pool)
            new_id = f"NEW_CUST_{random.randint(100000, 999999)}"
            
            monitored_customers[new_id] = {
                "customer_id": new_id,
                "contract": template["Contract"],
                "tenure": 1, # starting fresh
                "monthly_charges": float(template["MonthlyCharges"]),
                "churn_status": "Active",
                "days_monitored": 1,
                "support_tickets": 0,
                "logins": 0
            }
            
            day_events.append({
                "event_id": str(random.randint(1000000, 9999999)),
                "customer_id": new_id,
                "event_type": "signup",
                "timestamp": current_day.isoformat() + "Z",
                "value": float(template["MonthlyCharges"]),
                "device": random.choice(["desktop", "mobile", "tablet"]),
                "details": json.dumps({"contract": template["Contract"], "payment_method": template["PaymentMethod"]})
            })
            
        # 2. Update existing active customer usage and trigger events
        active_ids = [cid for cid, c in monitored_customers.items() if c["churn_status"] == "Active"]
        
        for cid in active_ids:
            cust = monitored_customers[cid]
            cust["days_monitored"] += 1
            
            # Increment tenure month every 30 days
            if cust["days_monitored"] % 30 == 0:
                cust["tenure"] += 1
                day_events.append({
                    "event_id": str(random.randint(1000000, 9999999)),
                    "customer_id": cid,
                    "event_type": "tenure_anniversary",
                    "timestamp": current_day.isoformat() + "Z",
                    "value": 0.0,
                    "device": "system",
                    "details": json.dumps({"new_tenure_months": cust["tenure"]})
                })
            
            # Daily active check: Logins, page views, support tickets
            if random.random() < 0.65: # 65% chance of logging in today
                cust["logins"] += 1
                day_events.append({
                    "event_id": str(random.randint(1000000, 9999999)),
                    "customer_id": cid,
                    "event_type": "login",
                    "timestamp": (current_day + timedelta(hours=random.randint(0, 23))).isoformat() + "Z",
                    "value": 0.0,
                    "device": random.choice(["desktop", "mobile", "tablet"]),
                    "details": "{}"
                })
                
            if random.random() < 0.12: # support tickets
                cust["support_tickets"] += 1
                day_events.append({
                    "event_id": str(random.randint(1000000, 9999999)),
                    "customer_id": cid,
                    "event_type": "support_ticket",
                    "timestamp": (current_day + timedelta(hours=random.randint(0, 23))).isoformat() + "Z",
                    "value": 0.0,
                    "device": random.choice(["desktop", "mobile", "tablet"]),
                    "details": json.dumps({"topic": random.choice(["billing", "speed", "intermittent_connection"])})
                })
                
        # 3. Simulate Churn Event
        # Sample customers at risk of churning
        # Month-to-month and high ticket counts have a higher probability of churning
        for cid in active_ids:
            cust = monitored_customers[cid]
            churn_probability = 0.01 # baseline daily risk
            
            if cust["contract"] == "Month-to-month":
                churn_probability += 0.03
            if cust["support_tickets"] > 2:
                churn_probability += 0.05
                
            # Cap maximum daily probability
            if random.random() < min(0.35, churn_probability):
                cust["churn_status"] = "Churned"
                day_events.append({
                    "event_id": str(random.randint(1000000, 9999999)),
                    "customer_id": cid,
                    "event_type": "churn_intent",
                    "timestamp": current_day.isoformat() + "Z",
                    "value": -cust["monthly_charges"],
                    "device": random.choice(["desktop", "mobile", "tablet"]),
                    "details": json.dumps({"reason": "competitor_offer", "final_tenure": cust["tenure"]})
                })
                logger.info(f"Customer {cid} churned on Day {d + 1}.")
                
        # 4. Write Daily Output Files
        # Daily Events (JSON Lines)
        event_file = os.path.join(output_dir, f"events_{date_str}.json")
        with open(event_file, "w") as ef:
            for ev in day_events:
                ef.write(json.dumps(ev) + "\n")
                
        # Daily Customer Status (CSV) containing calculated Customer Lifetime Value
        status_rows = []
        for cid, cust in monitored_customers.items():
            clv = calculate_clv(cust["contract"], cust["tenure"], cust["monthly_charges"])
            status_rows.append({
                "customer_id": cust["customer_id"],
                "contract": cust["contract"],
                "tenure": cust["tenure"],
                "monthly_charges": cust["monthly_charges"],
                "customer_lifetime_value": clv,
                "churn_status": cust["churn_status"],
                "days_active": cust["days_monitored"]
            })
            
        status_df = pd.DataFrame(status_rows)
        status_file = os.path.join(output_dir, f"customer_status_{date_str}.csv")
        status_df.to_csv(status_file, index=False)
        
        logger.info(f"Day {d + 1} generated: {len(day_events)} events, {len(status_rows)} customer status records.")
        
    logger.info(f"All outputs successfully written to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Event Stream & CLV Simulation")
    parser.add_argument("--days", type=int, default=7, help="Number of simulated days")
    parser.add_argument("--rate", type=int, default=3, help="New signups per day")
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    
    run_simulation(days=args.days, signups_per_day=args.rate, output_dir=args.output)

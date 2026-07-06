import time
import random
import uuid
import json
import argparse
from datetime import datetime
from shared.logging_config import setup_logger

logger = setup_logger("data_sim.generator")

# Personas to make the data interesting
# Stable: high usage, low support tickets, no downgrade/cancel intent
# High Risk: declining usage, high support tickets, views billing/pricing/cancellation
# New: low/medium usage, moderate growth
CUSTOMER_PERSONAS = {}
DEVICES = ["desktop", "mobile", "tablet"]
EVENT_TYPES = [
    "login", "page_view", "pricing_view", "support_ticket", 
    "subscription_upgrade", "subscription_downgrade", "churn_intent"
]

def generate_customers(num_customers=100):
    """Generate static customer profiles with specific risk profiles."""
    global CUSTOMER_PERSONAS
    for i in range(num_customers):
        customer_id = f"CUST_{1000 + i}"
        persona_type = random.choices(["stable", "high_risk", "new"], weights=[0.6, 0.25, 0.15])[0]
        # Monthly contract value (revenue)
        contract_value = round(random.uniform(20.0, 500.0), 2)
        CUSTOMER_PERSONAS[customer_id] = {
            "persona": persona_type,
            "contract_value": contract_value,
            "support_tickets_count": 0,
            "login_frequency": random.randint(1, 10), # scale of 1-10
            "days_since_last_login": 0
        }
    logger.info(f"Generated {num_customers} customer profiles with distinct personas.")

def generate_event(customer_id):
    """Generate a single event based on the customer persona."""
    profile = CUSTOMER_PERSONAS[customer_id]
    persona = profile["persona"]
    
    # Select event type based on persona probability weights
    if persona == "stable":
        event_weights = [0.4, 0.5, 0.05, 0.04, 0.01, 0.0, 0.0]
    elif persona == "high_risk":
        event_weights = [0.2, 0.2, 0.15, 0.35, 0.0, 0.08, 0.02]
    else: # new
        event_weights = [0.45, 0.45, 0.05, 0.03, 0.02, 0.0, 0.0]
        
    event_type = random.choices(EVENT_TYPES, weights=event_weights)[0]
    
    # Event metadata / details
    details = {}
    value = 0.0
    
    if event_type == "support_ticket":
        profile["support_tickets_count"] += 1
        details = {
            "ticket_id": f"TKT_{uuid.uuid4().hex[:6].upper()}",
            "priority": random.choices(["low", "medium", "high"], weights=[0.5, 0.3, 0.2])[0],
            "topic": random.choice(["billing", "technical_issue", "feature_request"])
        }
    elif event_type == "subscription_upgrade":
        old_val = profile["contract_value"]
        upgrade_amount = round(random.uniform(10.0, 100.0), 2)
        profile["contract_value"] += upgrade_amount
        value = upgrade_amount
        details = {"previous_value": old_val, "new_value": profile["contract_value"]}
    elif event_type == "subscription_downgrade":
        old_val = profile["contract_value"]
        downgrade_amount = round(random.uniform(5.0, min(50.0, old_val - 10.0)), 2)
        profile["contract_value"] = max(9.99, round(profile["contract_value"] - downgrade_amount, 2))
        value = -downgrade_amount
        details = {"previous_value": old_val, "new_value": profile["contract_value"]}
    elif event_type == "churn_intent":
        details = {"reason": random.choice(["price", "missing_features", "poor_support"])}
        
    event = {
        "event_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "value": value,
        "device": random.choice(DEVICES),
        "details": json.dumps(details)
    }
    return event

def stream_events(duration=60, rate=1.0, output_file=None):
    """Simulate a streaming event generator."""
    logger.info(f"Starting synthetic streaming event generator for {duration} seconds at {rate} events/sec.")
    start_time = time.time()
    
    file_handle = None
    if output_file:
        file_handle = open(output_file, "w")
        logger.info(f"Writing events to file: {output_file}")
        
    events_generated = 0
    try:
        while time.time() - start_time < duration:
            customer_id = random.choice(list(CUSTOMER_PERSONAS.keys()))
            event = generate_event(customer_id)
            
            event_json = json.dumps(event)
            if file_handle:
                file_handle.write(event_json + "\n")
                file_handle.flush()
            else:
                # Standard console output for streaming
                print(event_json)
                
            events_generated += 1
            time.sleep(1.0 / rate)
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
    finally:
        if file_handle:
            file_handle.close()
        logger.info(f"Simulation ended. Total events generated: {events_generated}")
        
    # Return summary profiles for ML input mock
    return CUSTOMER_PERSONAS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Streaming Customer Event Generator")
    parser.add_argument("--customers", type=int, default=100, help="Number of simulated customers")
    parser.add_argument("--duration", type=int, default=10, help="Duration of stream in seconds")
    parser.add_argument("--rate", type=float, default=2.0, help="Events per second")
    parser.add_argument("--output", type=str, default=None, help="Output JSON line file path")
    args = parser.parse_args()
    
    generate_customers(args.customers)
    stream_events(duration=args.duration, rate=args.rate, output_file=args.output)

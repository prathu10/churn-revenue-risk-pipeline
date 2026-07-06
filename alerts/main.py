import os
import sys
import logging
import requests
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

# Load environments
load_dotenv()

def process_alert(data):
    """
    Core alerting business logic. Can be imported directly or called by HTTP.
    
    Args:
        data (dict): Dictionary with customer_id, churn_probability, 
                     contract_value, and revenue_at_risk.
    """
    customer_id = data.get("customer_id")
    churn_prob = data.get("churn_probability", 0.0)
    contract_val = data.get("contract_value", 0.0)
    rev_at_risk = data.get("revenue_at_risk", 0.0)
    
    if not customer_id:
        logger.error("No customer_id specified in alert data.")
        return False
        
    logger.info(f"Processing risk alert for customer {customer_id} (Revenue-at-risk: ${rev_at_risk:.2f})")
    
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    # Check if we have a Slack Webhook configured.
    # If not, output formatted local log.
    if not webhook_url:
        logger.info(
            f"\n"
            f"===========================================================\n"
            f"🚨 MOCK INTERVENTION PIPELINE ALERT 🚨\n"
            f"Customer ID: {customer_id}\n"
            f"Churn Probability: {churn_prob:.1%}\n"
            f"Contract Value: ${contract_val:.2f}/mo\n"
            f"Revenue-at-Risk: ${rev_at_risk:.2f}/mo\n"
            f"RECOMMENDED INTERVENTION: Initiate high-priority customer support outreach\n"
            f"==========================================================="
        )
        return True
        
    # Standard Slack Webhook Block Kit
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "⚠️ Critical Churn Risk Alert ⚠️",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"A customer has breached the maximum acceptable revenue risk threshold."
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Customer ID:*\n{customer_id}"},
                    {"type": "mrkdwn", "text": f"*Churn Probability:*\n{churn_prob:.1%}"},
                    {"type": "mrkdwn", "text": f"*Contract Value:*\n${contract_val:.2f}/mo"},
                    {"type": "mrkdwn", "text": f"*Revenue-at-Risk:*\n${rev_at_risk:.2f}/mo"}
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Customer History",
                            "emoji": True
                        },
                        "value": customer_id,
                        "url": "https://console.cloud.google.com/bigquery"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            webhook_url, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            logger.info(f"Slack webhook sent successfully for {customer_id}.")
            return True
        else:
            logger.error(f"Slack webhook returned status {response.status_code}. Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to post alert to Slack: {str(e)}")
        return False

@functions_framework.http
def alerts_http_trigger(request):
    """
    HTTP entry point for GCP Cloud Function.
    Accepts JSON body with churn risk data.
    """
    if request.method != "POST":
        return ("Only POST requests are accepted", 405)
        
    request_json = request.get_json(silent=True)
    if not request_json:
        return ("Invalid JSON payload", 400)
        
    success = process_alert(request_json)
    if success:
        return ({"status": "success", "message": "Alert processed"}, 200)
    else:
        return ({"status": "error", "message": "Failed to process alert"}, 500)

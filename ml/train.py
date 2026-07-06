import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from shared.logging_config import setup_logger

logger = setup_logger("ml.train")

def generate_historical_dataset(n_samples=2000):
    """Generates synthetic historical customer churn data to train our model."""
    logger.info(f"Generating {n_samples} synthetic customer histories for training...")
    
    np.random.seed(42)
    
    # Features
    support_tickets = np.random.poisson(lam=1.2, size=n_samples)
    login_frequency = np.random.randint(1, 11, size=n_samples)
    contract_value = np.random.uniform(20.0, 500.0, size=n_samples)
    days_since_last_login = np.random.geometric(p=0.15, size=n_samples) - 1
    
    # Latent churn risk logic (formula + noise)
    # Higher support tickets, lower login frequency, more days inactive -> higher churn chance
    score = (support_tickets * 0.4) - (login_frequency * 0.25) + (days_since_last_login * 0.15)
    # Add random noise
    score += np.random.normal(loc=0.0, scale=0.5, size=n_samples)
    
    # Sigmoid probability
    prob = 1 / (1 + np.exp(-score))
    
    # Target label: Churn (1 = churned, 0 = active)
    # Threshold at 0.5
    churned = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        "support_tickets_count": support_tickets,
        "login_frequency": login_frequency,
        "contract_value": contract_value,
        "days_since_last_login": days_since_last_login,
        "churned": churned
    })
    
    return df

def train_churn_model():
    """Trains a Random Forest Classifier and saves the model artifact."""
    df = generate_historical_dataset()
    
    features = ["support_tickets_count", "login_frequency", "contract_value", "days_since_last_login"]
    X = df[features]
    y = df["churned"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    logger.info(f"Model Training Completed. ROC AUC: {auc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test, preds))
    
    # Save the model
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "churn_model.joblib")
    joblib.dump(clf, model_path)
    logger.info(f"Saved model to: {model_path}")
    
    return clf

if __name__ == "__main__":
    train_churn_model()

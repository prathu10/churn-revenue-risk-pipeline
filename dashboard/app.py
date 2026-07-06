import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# Page configuration for a premium, clean look
st.set_page_config(
    page_title="Churn Revenue Risk Monitor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vanilla CSS) inside Streamlit for enhanced design aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    .metric-card.critical {
        border-left-color: #ef4444;
    }
    .metric-card.warning {
        border-left-color: #f59e0b;
    }
    .metric-card.good {
        border-left-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Load env variables
load_dotenv()

def get_predictions_data():
    """
    Attempts to read data from local csv predictions cache.
    If unavailable, generates mock prediction records dynamically so the app runs out-of-the-box.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../output/churn_risk_scores.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        source = "local_cache"
    else:
        # Generate elegant mock data for the UI
        import numpy as np
        np.random.seed(42)
        n_samples = 80
        customer_ids = [f"CUST_{1000 + i}" for i in range(n_samples)]
        contract_values = np.round(np.random.uniform(30.0, 500.0, size=n_samples), 2)
        
        # Risk factors
        support_tickets = np.random.poisson(lam=0.9, size=n_samples)
        login_freq = np.random.randint(1, 11, size=n_samples)
        days_inactive = np.random.poisson(lam=3.0, size=n_samples)
        
        # Inject outliers
        support_tickets[5] = 5; login_freq[5] = 1; days_inactive[5] = 12
        support_tickets[12] = 4; login_freq[12] = 2; days_inactive[12] = 9
        support_tickets[22] = 6; login_freq[22] = 1; days_inactive[22] = 15
        
        # Compute scoring
        score = (support_tickets * 0.45) - (login_freq * 0.3) + (days_inactive * 0.2)
        probs = 1 / (1 + np.exp(-score))
        probs = np.round(probs, 4)
        rev_at_risk = np.round(probs * contract_values, 2)
        
        # Risk segments
        risk_segments = []
        for p in probs:
            if p >= 0.65:
                risk_segments.append("High")
            elif p >= 0.3:
                risk_segments.append("Medium")
            else:
                risk_segments.append("Low")
                
        df = pd.DataFrame({
            "customer_id": customer_ids,
            "support_tickets_count": support_tickets,
            "login_frequency": login_freq,
            "contract_value": contract_values,
            "days_since_last_login": days_inactive,
            "churn_probability": probs,
            "revenue_at_risk": rev_at_risk,
            "risk_segment": risk_segments,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        })
        source = "generated_mock"
        
    return df, source

# Load predictions
df, data_source = get_predictions_data()

# App Header
st.title("🔮 Churn Revenue Risk & Intervention Monitor")
st.markdown("Monitor high-risk customers, evaluate revenue exposure, and initiate proactive customer interventions.")

if data_source == "generated_mock":
    st.info("💡 **Demo Mode:** Displaying simulated customer data. To display actual pipeline output, run the ML scoring module `ml/predict.py` first.")
else:
    st.success("📈 **Live Mode:** Displaying active risk predictions loaded from local pipeline outputs.")

# KPI Row
total_customers = len(df)
avg_churn_prob = df["churn_probability"].mean()
total_contract_value = df["contract_value"].sum()
total_revenue_at_risk = df["revenue_at_risk"].sum()
overall_risk_pct = total_revenue_at_risk / total_contract_value if total_contract_value > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="metric-card good">'
        f'<h3>Active Customers</h3>'
        f'<h2>{total_customers}</h2>'
        f'<p style="color: #64748b; font-size: 14px;">Total monitored accounts</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="metric-card warning">'
        f'<h3>Avg Churn Risk</h3>'
        f'<h2>{avg_churn_prob:.1%}</h2>'
        f'<p style="color: #64748b; font-size: 14px;">Mean account probability</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="metric-card good">'
        f'<h3>Total Monthly ARR</h3>'
        f'<h2>${total_contract_value:,.2f}</h2>'
        f'<p style="color: #64748b; font-size: 14px;">Combined monthly revenue</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f'<div class="metric-card critical">'
        f'<h3>Revenue At Risk</h3>'
        f'<h2>${total_revenue_at_risk:,.2f}</h2>'
        f'<p style="color: #ef4444; font-size: 14px;">Exposure ({overall_risk_pct:.1%})</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

st.markdown("---")

# Main Content Layout
left_col, right_col = st.columns([7, 5])

with left_col:
    st.subheader("📊 Customer Risk Analysis")
    
    # Plotly Scatter Plot: Contract Value vs. Churn Probability
    fig_scatter = px.scatter(
        df,
        x="churn_probability",
        y="contract_value",
        size="revenue_at_risk",
        color="risk_segment",
        hover_name="customer_id",
        hover_data=["support_tickets_count", "days_since_last_login", "revenue_at_risk"],
        labels={
            "churn_probability": "Churn Probability",
            "contract_value": "Monthly Contract Value ($)",
            "risk_segment": "Risk Level"
        },
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        title="Contract Value vs Churn Probability (Bubble size = Revenue At Risk)"
    )
    
    fig_scatter.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with right_col:
    st.subheader("🍕 Risk Segmentation & Health")
    
    # Pie chart for risk segments
    segment_counts = df["risk_segment"].value_counts().reset_index()
    segment_counts.columns = ["risk_segment", "count"]
    
    fig_pie = px.pie(
        segment_counts,
        names="risk_segment",
        values="count",
        color="risk_segment",
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        hole=0.4,
        title="Customer Base Segmentation"
    )
    fig_pie.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Bottom Section: Customer List & Intervention Actions
st.subheader("🔍 Active Risk Directory & Interventions")

# Filters
col_filt1, col_filt2, col_filt3 = st.columns([3, 3, 6])
with col_filt1:
    search_query = st.text_input("Search Customer ID", placeholder="CUST_1001")
with col_filt2:
    selected_segment = st.multiselect("Filter by Risk Level", options=["High", "Medium", "Low"], default=["High", "Medium"])
with col_filt3:
    min_arr = st.slider("Min Contract Value ($)", min_value=0, max_value=500, value=0)

# Filter Dataframe
filtered_df = df.copy()
if search_query:
    filtered_df = filtered_df[filtered_df["customer_id"].str.contains(search_query, case=False)]
if selected_segment:
    filtered_df = filtered_df[filtered_df["risk_segment"].isin(selected_segment)]
filtered_df = filtered_df[filtered_df["contract_value"] >= min_arr]

# Sort by Revenue At Risk descending
filtered_df = filtered_df.sort_values(by="revenue_at_risk", ascending=False)

# Display table
st.dataframe(
    filtered_df[[
        "customer_id", "risk_segment", "churn_probability", "contract_value", 
        "revenue_at_risk", "support_tickets_count", "days_since_last_login"
    ]].style.format({
        "churn_probability": "{:.1%}",
        "contract_value": "${:,.2f}",
        "revenue_at_risk": "${:,.2f}"
    }).background_gradient(subset=["revenue_at_risk"], cmap="OrRd"),
    use_container_width=True
)

# Sidebar - Interventions & Model Pipeline Run Simulation
with st.sidebar:
    st.header("⚙️ Pipeline Intervention Control")
    st.markdown("Perform quick operations and mock alerts to support systems.")
    
    st.subheader("Manual Alert Trigger")
    high_risk_list = df[df["risk_segment"] == "High"]["customer_id"].tolist()
    
    if high_risk_list:
        selected_cust_id = st.selectbox("Select Customer to Alert", options=high_risk_list)
        cust_row = df[df["customer_id"] == selected_cust_id].iloc[0]
        
        st.write(f"**Customer ID:** {selected_cust_id}")
        st.write(f"**Churn Risk:** {cust_row['churn_probability']:.1%}")
        st.write(f"**Revenue at Risk:** ${cust_row['revenue_at_risk']:.2f}")
        
        if st.button("🚨 Dispatch Alert Notification"):
            # Call alert system
            try:
                import requests
                # Mock calling the local alerts processing
                sys_path_added = False
                alerts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../alerts"))
                import sys
                if alerts_path not in sys.path:
                    sys.path.append(alerts_path)
                    sys_path_added = True
                    
                from main import process_alert
                
                alert_data = {
                    "customer_id": cust_row["customer_id"],
                    "churn_probability": float(cust_row["churn_probability"]),
                    "revenue_at_risk": float(cust_row["revenue_at_risk"]),
                    "contract_value": float(cust_row["contract_value"])
                }
                
                success = process_alert(alert_data)
                if success:
                    st.success(f"Notification triggered for {selected_cust_id}!")
                else:
                    st.error("Failed to send notification. Check logs.")
            except Exception as ex:
                st.error(f"Error triggering alert: {str(ex)}")
    else:
        st.write("No high risk customers detected to trigger alerts.")
        
    st.markdown("---")
    st.subheader("System Info")
    st.write(f"**Loaded records:** {len(df)}")
    st.write(f"**Last Sync:** {df['last_updated'].max()[:19] if 'last_updated' in df.columns else 'N/A'}")

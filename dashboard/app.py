import os
import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import bigquery
from bigquery.loader import get_bq_client
from shared.logging_config import setup_logger

logger = setup_logger("dashboard.app")

# Load environment variables
load_dotenv()

# Page configuration for a premium, clean look
st.set_page_config(
    page_title="Interventions Control Room",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and glassmorphism styling
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f19;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 13px;
        margin-top: 6px;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    .status-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-local {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper: Parse Project credentials
project_id = os.getenv("GCP_PROJECT_ID")
dataset_id = os.getenv("BIGQUERY_DATASET", "churn_pipeline")

@st.cache_data(ttl=60)
def fetch_dashboard_data(local_only=False):
    """
    Queries BigQuery churn tables or merges local pipeline CSV caches.
    Returns: (predictions_df, metrics_df, trend_df, is_live)
    """
    if not local_only and project_id:
        try:
            client = get_bq_client()
            
            # 1. Fetch latest predictions matched with demographic features
            predictions_query = f"""
                WITH latest_predictions AS (
                    SELECT customer_id, churn_probability, revenue_at_risk, predicted_date
                    FROM `{client.project}.{dataset_id}.churn_predictions`
                    WHERE predicted_date = (SELECT MAX(predicted_date) FROM `{client.project}.{dataset_id}.churn_predictions`)
                ),
                latest_features AS (
                    SELECT * EXCEPT(load_date)
                    FROM `{client.project}.{dataset_id}.customer_features`
                    WHERE load_date = (SELECT MAX(load_date) FROM `{client.project}.{dataset_id}.customer_features`)
                )
                SELECT p.customer_id, p.churn_probability, p.revenue_at_risk, p.predicted_date,
                       f.contract_type, f.tenure, f.monthly_charges, f.customer_lifetime_value,
                       f.usage_trends, f.support_ticket_frequency, f.payment_method, f.churn_status
                FROM latest_predictions p
                LEFT JOIN latest_features f ON p.customer_id = f.customer_id
                ORDER BY p.revenue_at_risk DESC
            """
            logger.info("Executing predictions query on BigQuery...")
            pred_job = client.query(predictions_query)
            pred_df = pd.DataFrame([dict(row) for row in pred_job])
            
            # 2. Fetch pipeline health metrics
            metrics_query = f"""
                SELECT run_date, records_in, records_out, null_rate, run_duration
                FROM `{client.project}.{dataset_id}.pipeline_metrics`
                ORDER BY run_date DESC
            """
            logger.info("Executing pipeline metrics query on BigQuery...")
            metrics_job = client.query(metrics_query)
            metrics_df = pd.DataFrame([dict(row) for row in metrics_job])
            
            # 3. Fetch trend history
            trend_query = f"""
                SELECT DATE(predicted_date) as run_date, SUM(revenue_at_risk) as total_revenue_at_risk, COUNT(customer_id) as total_customers
                FROM `{client.project}.{dataset_id}.churn_predictions`
                GROUP BY run_date
                ORDER BY run_date ASC
            """
            logger.info("Executing trend history query on BigQuery...")
            trend_job = client.query(trend_query)
            trend_df = pd.DataFrame([dict(row) for row in trend_job])
            
            if not pred_df.empty:
                return pred_df, metrics_df, trend_df, True
        except Exception as e:
            logger.warning(f"Failed to query live BigQuery tables: {str(e)}. Falling back to local files.")
            
    # Fallback to local files
    try:
        processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output/processed"))
        predictions_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output/churn_predictions.csv"))
        
        # Load local predictions
        if not os.path.exists(predictions_path):
            raise FileNotFoundError("Local predictions file not found. Run ML scoring first.")
            
        pred_df = pd.read_csv(predictions_path)
        
        # Merge local features to match structure
        if os.path.exists(processed_dir):
            files = [
                os.path.join(processed_dir, f) 
                for f in os.listdir(processed_dir) 
                if f.startswith("processed_features_") and f.endswith(".csv")
            ]
            if files:
                features_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
                features_df = features_df.drop_duplicates(subset=["customer_id"], keep="last")
                pred_df = pd.merge(pred_df, features_df.drop(columns=["churn_status"], errors="ignore"), on="customer_id", how="left")
                # Default fill missing values
                pred_df["churn_status"] = np.where(pred_df["churn_probability"] >= 0.5, "Churned", "Active")
                
        # Generate dummy metrics history for local view
        metrics_df = pd.DataFrame([
            {"run_date": datetime(2026, 7, 5).date(), "records_in": 68, "records_out": 39, "null_rate": 0.0, "run_duration": 4.90},
            {"run_date": datetime(2026, 7, 4).date(), "records_in": 70, "records_out": 36, "null_rate": 0.0, "run_duration": 3.82},
            {"run_date": datetime(2026, 7, 3).date(), "records_in": 65, "records_out": 33, "null_rate": 0.0, "run_duration": 4.15}
        ])
        
        # Generate dummy trend history for local view
        trend_df = pd.DataFrame([
            {"run_date": datetime(2026, 7, 3).date(), "total_revenue_at_risk": 5800.0, "total_customers": 33},
            {"run_date": datetime(2026, 7, 4).date(), "total_revenue_at_risk": 6400.0, "total_customers": 36},
            {"run_date": datetime(2026, 7, 5).date(), "total_revenue_at_risk": 7890.5, "total_customers": 39}
        ])
        
        return pred_df, metrics_df, trend_df, False
    except Exception as ex:
        # Emergency dummy data if everything else fails
        logger.error(f"Failed to load any local data: {str(ex)}")
        st.error(f"Error loading dashboard data: {str(ex)}")
        
        # Build quick mock
        dummy_df = pd.DataFrame({
            "customer_id": [f"C_{i}" for i in range(10)],
            "churn_probability": [0.8, 0.4, 0.1, 0.9, 0.3, 0.05, 0.75, 0.2, 0.85, 0.12],
            "revenue_at_risk": [800.0, 200.0, 50.0, 1800.0, 90.0, 10.0, 1500.0, 60.0, 2550.0, 24.0],
            "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Month-to-month", "One year"],
            "tenure": [12, 24, 36, 2, 45, 60, 5, 54, 8, 22],
            "monthly_charges": [100.0, 50.0, 80.0, 120.0, 75.0, 90.0, 110.0, 45.0, 150.0, 65.0],
            "customer_lifetime_value": [1200.0, 1200.0, 2880.0, 240.0, 3375.0, 5400.0, 550.0, 2430.0, 1200.0, 1430.0],
            "usage_trends": [2, 5, 9, 1, 6, 8, 2, 7, 3, 5],
            "support_ticket_frequency": [1, 0, 0, 3, 1, 0, 2, 0, 4, 1],
            "payment_method": ["Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Credit card (automatic)", "Electronic check", "Mailed check"],
            "churn_status": ["Active", "Active", "Active", "Active", "Active", "Active", "Active", "Active", "Active", "Active"]
        })
        return dummy_df, pd.DataFrame(), pd.DataFrame(), False

# ----------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
st.sidebar.title("Interventions Console")
st.sidebar.markdown("Configure filters and local simulation runs.")

# Run mode
local_mode = st.sidebar.checkbox("Force Local Offline Mode", value=False)

# Load data
df, metrics_df, trend_df, is_live = fetch_dashboard_data(local_only=local_mode)

# Render sync status badge
if is_live:
    st.sidebar.markdown('<span class="status-badge status-live">● BigQuery Live Link</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge status-local">● Local File Ingest</span>', unsafe_allow_html=True)

# ----------------------------------------------------
# Main Layout
# ----------------------------------------------------
st.title("🔮 Customer Churn Revenue Risk Dashboard")
st.markdown("Interventions control panel mapping accounts exposure, model metrics, and pipeline runs.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Churn Risk Control Room", "💸 Intervention What-If Simulator", "⚙️ Pipeline Health Center"])

# ----------------------------------------------------
# TAB 1: CONTROL ROOM
# ----------------------------------------------------
with tab1:
    # 1. KPIs Row
    total_customers = len(df)
    total_revenue_at_risk = df["revenue_at_risk"].sum()
    avg_churn_prob = df["churn_probability"].mean()
    total_clv = df["customer_lifetime_value"].sum()
    exposure_pct = total_revenue_at_risk / total_clv if total_clv > 0 else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Monitored Customers</div>'
            f'<div class="metric-value">{total_customers}</div>'
            f'<div class="metric-subtitle" style="color: #10b981;">Active base snapshots</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Total Exposure Risk</div>'
            f'<div class="metric-value">${total_revenue_at_risk:,.2f}</div>'
            f'<div class="metric-subtitle" style="color: #ef4444;">Revenue exposed to churn</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Mean Churn Probability</div>'
            f'<div class="metric-value">{avg_churn_prob:.1%}</div>'
            f'<div class="metric-subtitle" style="color: #f59e0b;">Probability index</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">CLV Exposure Ratio</div>'
            f'<div class="metric-value">{exposure_pct:.2%}</div>'
            f'<div class="metric-subtitle" style="color: #3b82f6;">Revenue-at-risk / Total CLV</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Charts section
    chart_col1, chart_col2 = st.columns([7, 5])
    with chart_col1:
        st.subheader("Bubble Distribution: Churn Probability vs. Contract Value")
        # Ensure sizes are positive for px.scatter size parameter
        df_plot = df.copy()
        df_plot["bubble_size"] = np.where(df_plot["revenue_at_risk"] <= 0, 1.0, df_plot["revenue_at_risk"])
        
        fig_scatter = px.scatter(
            df_plot,
            x="churn_probability",
            y="monthly_charges" if "monthly_charges" in df_plot.columns else "contract_value",
            size="bubble_size",
            color="contract_type" if "contract_type" in df_plot.columns else "payment_method",
            hover_name="customer_id",
            hover_data=["customer_lifetime_value", "revenue_at_risk"],
            labels={
                "churn_probability": "Churn Probability",
                "monthly_charges": "Monthly Charge ($)",
                "contract_type": "Contract Type"
            },
            title="Customers Risk Map (Bubble Size = Revenue At Risk)"
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_scatter, width="stretch")
        
    with chart_col2:
        st.subheader("Historical Revenue-at-Risk Trend")
        if not trend_df.empty:
            # Check format of run_date
            trend_df["run_date"] = pd.to_datetime(trend_df["run_date"])
            fig_trend = px.line(
                trend_df,
                x="run_date",
                y="total_revenue_at_risk",
                text="total_revenue_at_risk",
                labels={
                    "run_date": "ETL Run Date",
                    "total_revenue_at_risk": "Exposed Revenue ($)"
                },
                markers=True,
                title="Total Revenue Risk Over Time"
            )
            fig_trend.update_traces(
                texttemplate='$%{text:,.2f}',
                textposition='top center',
                line_color='#ef4444',
                line_width=3
            )
            fig_trend.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info("No historical trends log available.")
            
    # 3. Top 20 Table
    st.markdown("---")
    st.subheader("🚨 Top 20 Customer Accounts Ranked by Revenue Risk")
    top_20 = df.sort_values(by="revenue_at_risk", ascending=False).head(20).copy()
    
    # Ensure correct columns exist
    display_cols = [
        "customer_id", "churn_probability", "revenue_at_risk", "customer_lifetime_value",
        "contract_type", "tenure", "monthly_charges", "support_ticket_frequency", "usage_trends"
    ]
    actual_cols = [c for c in display_cols if c in top_20.columns]
    
    # Clean display names
    clean_names = {
        "customer_id": "Customer ID",
        "churn_probability": "Churn Risk",
        "revenue_at_risk": "Revenue At Risk",
        "customer_lifetime_value": "CLV",
        "contract_type": "Contract",
        "tenure": "Tenure (Mo)",
        "monthly_charges": "Monthly Charges",
        "support_ticket_frequency": "Support Tickets",
        "usage_trends": "Logins (Day)"
    }
    
    table_df = top_20[actual_cols].rename(columns=clean_names)
    
    st.dataframe(
        table_df.style.format({
            "Churn Risk": "{:.1%}",
            "Revenue At Risk": "${:,.2f}",
            "CLV": "${:,.2f}",
            "Monthly Charges": "${:,.2f}"
        }).background_gradient(subset=["Revenue At Risk"], cmap="OrRd"),
        width="stretch"
    )

# ----------------------------------------------------
# TAB 2: WHAT-IF SIMULATOR
# ----------------------------------------------------
with tab2:
    st.subheader("💡 Churn Intervention Recovery Simulator")
    st.markdown("Evaluate potential ARR recovery savings when targeted support retention campaigns are successfully executed.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns([5, 7])
    
    with sim_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Configure Retainment Metrics</h4>", unsafe_allow_html=True)
        reduction_pct = st.slider("Campaign Reduction Target (%)", 0, 100, 25, help="Percentage of exposed revenue recovered through targeted interventions.")
        intervention_cost_per_cust = st.number_input("Intervention Cost per Customer ($)", min_value=0.0, max_value=500.0, value=25.0, step=5.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate simulation outputs
        total_risk = df["revenue_at_risk"].sum()
        revenue_recovered = total_risk * (reduction_pct / 100.0)
        
        # Simple cost: assume we target top 25% of at-risk customers
        targeted_count = int(np.ceil(len(df) * 0.25))
        total_intervention_cost = targeted_count * intervention_cost_per_cust
        net_roi = revenue_recovered - total_intervention_cost
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(
                label="Recovered Monthly Revenue",
                value=f"${revenue_recovered:,.2f}",
                delta=f"+{reduction_pct}% recovery",
                delta_color="normal"
            )
        with col_res2:
            st.metric(
                label="Net Intervention ROI",
                value=f"${net_roi:,.2f}",
                delta=f"Targeting {targeted_count} customers",
                delta_color="off" if net_roi >= 0 else "inverse"
            )
            
    with sim_col2:
        # Plotly bar chart comparing exposure vs recovered
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            name='Exposed Revenue',
            x=['Baseline Exposure'],
            y=[total_risk],
            marker_color='#ef4444',
            text=[f"${total_risk:,.2f}"],
            textposition='auto'
        ))
        fig_sim.add_trace(go.Bar(
            name='Recovered Revenue',
            x=['Post-Intervention Savings'],
            y=[revenue_recovered],
            marker_color='#10b981',
            text=[f"${revenue_recovered:,.2f}"],
            textposition='auto'
        ))
        fig_sim.add_trace(go.Bar(
            name='Net Exposure',
            x=['Post-Intervention Savings'],
            y=[total_risk - revenue_recovered],
            marker_color='#3b82f6',
            text=[f"${total_risk - revenue_recovered:,.2f}"],
            textposition='auto'
        ))
        
        fig_sim.update_layout(
            barmode='group',
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title="Revenue Profile (Target vs Exposure Risk)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_sim, width="stretch")

# ----------------------------------------------------
# TAB 3: PIPELINE HEALTH CENTER
# ----------------------------------------------------
with tab3:
    st.subheader("⚙️ Machine Learning Pipeline & ETL Metrics")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Model Baseline Card
    health_col1, health_col2 = st.columns([5, 7])
    
    with health_col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin-top:0; color:#3b82f6;">🔮 Baseline Model Performance</h4>
            <table style="width:100%; border-collapse: collapse; margin-top:15px; font-size:14px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600;">Algorithm</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">Random Forest Classifier</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600;">Model Depth (Max)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">4</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600;">Class Imbalance Handling</td>
                    <td style="text-align:right; font-weight:700; color:#10b981;">class_weight='balanced'</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600;">ROC-AUC Score</td>
                    <td style="text-align:right; font-weight:700; color:#10b981; font-size:15px;">95.83%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px; background-color: rgba(255,255,255,0.02)">
                    <td style="color:#3b82f6; font-weight:700;" colspan="2">Metrics at Default Threshold (0.5)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">Precision (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">100.00%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">Recall (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">66.67%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">F1-Score (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">80.00%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px; background-color: rgba(255,255,255,0.02)">
                    <td style="color:#f59e0b; font-weight:700;" colspan="2">Metrics at Custom Threshold (0.3) [Rec.]</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">Precision (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">25.00%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">Recall (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">100.00%</td>
                </tr>
                <tr style="height: 32px;">
                    <td style="color:#94a3b8; font-weight:600; padding-left: 10px;">F1-Score (Churned)</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">40.00%</td>
                </tr>
            </table>
            <p style="color:#64748b; font-size:12px; margin-top:20px; font-style:italic;">
                *Note: The model includes class weight balancing. At the default 0.5 threshold, it achieves 100% precision with 66.67% recall (detecting 2 out of 3 churned customers). Lowering the decision threshold to 0.3 recovers all 3 churned customers (100% recall) with a precision of 25% due to the small, imbalanced test partition (3 churned, 24 active).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with health_col2:
        st.markdown("<h4 style='margin-top:0;'>ETL Run History (pipeline_metrics)</h4>", unsafe_allow_html=True)
        if not metrics_df.empty:
            # Rename columns for display
            metrics_display = metrics_df.copy()
            metrics_display.columns = ["Run Date", "Records In", "Records Out", "Null Rate (%)", "Duration (s)"]

            # Coerce numeric columns — backfilled rows may have None for Duration
            metrics_display["Null Rate (%)"] = pd.to_numeric(metrics_display["Null Rate (%)"], errors="coerce")
            metrics_display["Duration (s)"]  = pd.to_numeric(metrics_display["Duration (s)"],  errors="coerce")

            # Build per-cell format functions that gracefully handle NaN/None → "N/A"
            def fmt_pct(v):
                try:
                    return f"{float(v):.2f}%" if v is not None and not pd.isna(v) else "N/A"
                except (TypeError, ValueError):
                    return "N/A"

            def fmt_dur(v):
                try:
                    return f"{float(v):.2f}s" if v is not None and not pd.isna(v) else "N/A"
                except (TypeError, ValueError):
                    return "N/A"

            st.dataframe(
                metrics_display.style.format({
                    "Null Rate (%)": fmt_pct,
                    "Duration (s)":  fmt_dur,
                }),
                width="stretch"
            )
        else:
            st.info("No pipeline metrics records available in BQ.")


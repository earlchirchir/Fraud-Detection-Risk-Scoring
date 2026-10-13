"""
Finlora Finance Flow - Fraud Detection & Transaction Risk Scoring Command Center
Streamlit Application for Prioritized Analyst Review & Real-Time Scoring
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Finlora | Fraud Risk Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Fintech styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model Artifacts (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_holdout_data():
    holdout_path = os.path.join('data', 'processed', 'holdout_predictions.csv')
    if not os.path.exists(holdout_path):
        holdout_path = os.path.join('..', 'data', 'processed', 'holdout_predictions.csv')
    df = pd.read_csv(holdout_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models():
    model_dir = 'model'
    if not os.path.exists(os.path.join(model_dir, 'random_forest_champion.joblib')):
        model_dir = os.path.join('..', 'model')
    rf_model = joblib.load(os.path.join(model_dir, 'random_forest_champion.joblib'))
    lr_model = joblib.load(os.path.join(model_dir, 'logistic_regression_baseline.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    
    with open(os.path.join(model_dir, 'evaluation_summary.json'), 'r') as f:
        eval_summary = json.load(f)
        
    return rf_model, lr_model, scaler, feature_cols, eval_summary

try:
    df_holdout = load_holdout_data()
    rf_model, lr_model, scaler, feature_cols, eval_summary = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    data_loaded = False

# ---------------------------------------------------------
# Sidebar: Brand, Policy Configuration & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.title("Finlora Risk Ops")
    st.caption("Fintech Transaction Monitoring & Risk Scoring")
    st.divider()
    
    st.subheader("Operating Risk Policy")
    
    # Extract calibrated threshold values from evaluation summary
    t_70 = 0.85
    t_80 = 0.70
    t_90 = 0.50
    if data_loaded and 'tunable_thresholds' in eval_summary:
        for t_info in eval_summary['tunable_thresholds']:
            if '70%' in t_info['Operating Policy']:
                t_70 = t_info['Probability Threshold']
            elif '80%' in t_info['Operating Policy']:
                t_80 = t_info['Probability Threshold']
            elif '90%' in t_info['Operating Policy']:
                t_90 = t_info['Probability Threshold']

    policy_mode = st.radio(
        "Select Operating Calibration:",
        ["80% Recall (Balanced - Recommended)", "70% Recall (Conservative)", "90% Recall (Aggressive)", "Custom Cutoff"],
        index=0
    )
    
    if "70%" in policy_mode:
        selected_threshold = t_70
        target_label = "70% Fraud Caught (Fewer false alarms, lowest review workload)"
    elif "80%" in policy_mode:
        selected_threshold = t_80
        target_label = "80% Fraud Caught (Recommended balance between catching fraud and review workload)"
    elif "90%" in policy_mode:
        selected_threshold = t_90
        target_label = "90% Fraud Caught (Catches the most fraud, but flags more innocent payments for review)"
    else:
        selected_threshold = st.slider("Custom Fraud Probability Cutoff:", 0.05, 0.95, float(t_80), 0.01)
        target_label = f"Custom Cutoff: >= {selected_threshold:.2f}"
        
    st.info(f"Active Threshold: {selected_threshold:.4f}\n\n{target_label}")
    st.divider()
    
    st.subheader("Review Queue Filters")
    channel_filter = st.multiselect(
        "Channel:",
        options=sorted(df_holdout['channel'].unique()) if data_loaded else [],
        default=sorted(df_holdout['channel'].unique()) if data_loaded else []
    )
    
    kyc_filter = st.multiselect(
        "KYC Tier:",
        options=sorted(df_holdout['kyc_tier'].unique()) if data_loaded else ['Basic', 'Verified', 'Enhanced'],
        default=sorted(df_holdout['kyc_tier'].unique()) if data_loaded else ['Basic', 'Verified', 'Enhanced']
    )
    
    account_filter = st.multiselect(
        "Account Type:",
        options=['Individual', 'Business'],
        default=['Individual', 'Business']
    )
    
    st.divider()
    st.caption("Model Champion: Random Forest (300 Trees, Depth 12)")
    if data_loaded:
        st.caption(f"Champion ROC-AUC: {eval_summary['benchmark_models']['random_forest_champion']['roc_auc']:.3f}")

# ---------------------------------------------------------
# Main Dashboard Header & KPI Metrics
# ---------------------------------------------------------
st.markdown('<div class="main-header">Finlora Transaction Risk & Fraud Review Queue</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Live fraud monitoring and review dashboard for customer transactions</div>', unsafe_allow_html=True)

if data_loaded:
    # Filter dataset according to threshold and sidebar controls
    flagged_df = df_holdout[
        (df_holdout['fraud_probability'] >= selected_threshold) &
        (df_holdout['channel'].isin(channel_filter)) &
        (df_holdout['kyc_tier'].isin(kyc_filter)) &
        (df_holdout['account_type'].isin(account_filter))
    ].copy()
    
    total_volume = len(df_holdout)
    flagged_count = len(flagged_df)
    alert_rate = (flagged_count / total_volume) * 100
    flagged_value = flagged_df['amount'].abs().sum()
    true_fraud_caught = flagged_df['is_fraud'].sum()
    estimated_precision = (true_fraud_caught / flagged_count * 100) if flagged_count > 0 else 0.0

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{total_volume:,} txns", "Test window")
    with col2:
        st.metric("Flagged for Review", f"{flagged_count:,} flagged", f"{alert_rate:.2f}% of all payments")
    with col3:
        st.metric("Flagged Dollar Amount", f"${flagged_value:,.2f}", f"{true_fraud_caught:,} real fraud cases")
    with col4:
        st.metric("Review Accuracy", f"{estimated_precision:.1f}%", f"{true_fraud_caught}/{flagged_count} confirmed fraud")

    st.markdown("---")

    # ---------------------------------------------------------
    # Dashboard Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "Prioritized Analyst Review Queue", 
        "Real-Time Transaction Scorer", 
        "Portfolio Diagnostics & Feature Drivers"
    ])

    # ---------------------------------------------------------
    # TAB 1: PRIORITIZED REVIEW QUEUE
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Analyst Worklist: Suspicious Transactions Ranked by Risk")
        st.write("Transactions are ordered from highest to lowest risk so investigators review the most suspicious cases first.")
        
        if flagged_df.empty:
            st.success("No transactions currently meet or exceed the selected fraud probability threshold.")
        else:
            # Display Table
            display_cols = [
                'transaction_id', 'account_id', 'amount', 'account_type', 
                'kyc_tier', 'channel', 'merchant_category', 'amount_to_avg_ratio',
                'is_new_device', 'is_cross_border', 'fraud_probability'
            ]
            
            show_df = flagged_df[display_cols].copy()
            show_df['amount'] = show_df['amount'].apply(lambda x: f"${abs(x):,.2f}")
            show_df['amount_to_avg_ratio'] = show_df['amount_to_avg_ratio'].apply(lambda x: f"{x:.1f}x")
            show_df['fraud_probability'] = show_df['fraud_probability'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(show_df, width="stretch" if hasattr(st, "dataframe") else True, height=280)
            
            # Case Investigation & Explainability Drawer
            st.subheader("Case Details & Warning Signals")
            selected_txn_id = st.selectbox(
                "Select Transaction ID to inspect contributing signals & take action:",
                options=flagged_df['transaction_id'].tolist()
            )
            
            case_row = flagged_df[flagged_df['transaction_id'] == selected_txn_id].iloc[0]
            
            inv_col1, inv_col2 = st.columns([1.2, 1.0])
            with inv_col1:
                st.markdown(f"#### Transaction: `{case_row['transaction_id']}` (Account: `{case_row['account_id']}`)")
                
                # Risk Probability Badge
                prob_val = case_row['fraud_probability']
                if prob_val >= 0.85:
                    badge_html = f'<span class="status-badge-critical">CRITICAL RISK: {prob_val*100:.1f}%</span>'
                elif prob_val >= 0.70:
                    badge_html = f'<span class="status-badge-high">HIGH RISK: {prob_val*100:.1f}%</span>'
                elif prob_val >= 0.50:
                    badge_html = f'<span class="status-badge-medium">ELEVATED RISK: {prob_val*100:.1f}%</span>'
                else:
                    badge_html = f'<span class="status-badge-low">LOW RISK: {prob_val*100:.1f}%</span>'
                
                st.markdown(f"Assigned Risk Tier: {badge_html}", unsafe_allow_html=True)
                st.write("")
                
                detail_colA, detail_colB = st.columns(2)
                with detail_colA:
                    st.write(f"Amount: ${abs(case_row['amount']):,.2f}")
                    st.write(f"Channel: {case_row['channel']}")
                    st.write(f"Category: {case_row['merchant_category']}")
                    st.write(f"KYC Tier: {case_row['kyc_tier']}")
                with detail_colB:
                    st.write(f"Account Type: {case_row['account_type']}")
                    st.write(f"Spending vs Average: {case_row['amount_to_avg_ratio']:.1f}x normal spending")
                    st.write(f"Payments in Last Hour: {case_row['transaction_velocity_1h']} payments")
                    st.write(f"Device: {'Unrecognized Device' if case_row['is_new_device'] == 1 else 'Known / Familiar'}")
                    st.write(f"Location: {'International (Outside home country)' if case_row['is_cross_border'] == 1 else 'Local / Domestic'}")

            with inv_col2:
                st.markdown("#### Why This Payment Was Flagged")
                signals = []
                if case_row['amount_to_avg_ratio'] >= 5.0:
                    signals.append(f"Spending spike: {case_row['amount_to_avg_ratio']:.1f}x higher than normal 30-day average")
                if case_row['is_new_device'] == 1:
                    signals.append("New device: Payment made from an unrecognized phone or computer")
                if case_row['is_cross_border'] == 1:
                    signals.append("Foreign location: Payment made outside the customer's home country")
                if case_row['transaction_velocity_1h'] >= 2:
                    signals.append(f"Rapid payments: {case_row['transaction_velocity_1h']} transactions within the last hour")
                if case_row['merchant_category'] in ['Wire Transfer', 'Payroll Transfer', 'Crypto Exchange']:
                    signals.append(f"High-risk store type: {case_row['merchant_category']}")
                if not signals:
                    signals.append("Normal activity: Transaction fits typical customer habits")
                    
                for sig in signals:
                    st.markdown(f"- {sig}")
                    
                st.markdown("---")
                st.markdown("##### Analyst Action Decision")
                act_col1, act_col2, act_col3 = st.columns(3)
                with act_col1:
                    if st.button("Approve", key=f"app_{selected_txn_id}"):
                        st.success(f"Transaction {selected_txn_id} approved as Legitimate. Logged to feedback store.")
                with act_col2:
                    if st.button("Step-Up 2FA", key=f"step_{selected_txn_id}"):
                        st.warning(f"OTP authentication dispatched for {selected_txn_id}.")
                with act_col3:
                    if st.button("Confirm Fraud", key=f"conf_{selected_txn_id}"):
                        st.error(f"Transaction {selected_txn_id} confirmed as FRAUD. Account soft-blocked.")

    # ---------------------------------------------------------
    # TAB 2: REAL-TIME TRANSACTION SCORING SIMULATOR
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Live Payment Scorer (Test Mode)")
        st.write("Enter payment details below to see how the system scores fraud risk and explains its decision in real time.")
        
        with st.form("scoring_form"):
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_amount = st.number_input("Transaction Amount ($):", min_value=1.0, value=85000.0, step=1000.0)
                sim_baseline = st.number_input("Customer Historical 30-Day Average ($):", min_value=1.0, value=8500.0, step=500.0)
                sim_velocity = st.slider("Trailing 1-Hour Velocity Count:", 0, 10, 3)
            with sim_col2:
                sim_account_type = st.selectbox("Account Type:", ["Individual", "Business"])
                sim_kyc_tier = st.selectbox("KYC Tier:", ["Basic", "Verified", "Enhanced"], index=0)
                sim_channel = st.selectbox("Payment Channel:", [
                    "Mobile App", "Web Dashboard", "API/Integration", 
                    "Card Not Present", "Card Present", "USSD"
                ], index=2)
            with sim_col3:
                sim_category = st.selectbox("Merchant Category:", [
                    "Wire Transfer", "Payroll Transfer", "Crypto Exchange", "Travel", 
                    "P2P Transfer", "Electronics", "Retail", "Groceries", "Utilities"
                ], index=0)
                sim_new_device = st.checkbox("Unrecognized / Novel Hardware Device", value=True)
                sim_cross_border = st.checkbox("Cross-Border Foreign Transaction", value=True)
                
            submitted = st.form_submit_button("Calculate Live Fraud Risk Score")
            
        if submitted:
            ratio = sim_amount / max(sim_baseline, 1.0)
            
            # Construct feature dictionary matching model schema
            row = {col: 0 for col in feature_cols}
            row['log_amount'] = np.log1p(abs(sim_amount))
            row['log_amount_to_avg_ratio'] = np.log1p(max(ratio, 0))
            row['log_avg_30d'] = np.log1p(max(sim_baseline, 1.0))
            row['transaction_velocity_1h'] = sim_velocity
            row['is_new_device'] = 1 if sim_new_device else 0
            row['is_cross_border'] = 1 if sim_cross_border else 0
            row['is_negative_amount'] = 0
            row['account_age_days'] = 180
            row['hour_of_day'] = 14
            
            # Dummy column assignments
            for prefix, val in [
                ('account_type_', sim_account_type),
                ('kyc_tier_', sim_kyc_tier),
                ('channel_', sim_channel),
                ('merchant_category_', sim_category)
            ]:
                dummy_col = f"{prefix}{val}"
                if dummy_col in row:
                    row[dummy_col] = 1
                    
            input_df = pd.DataFrame([row])[feature_cols]
            live_prob = float(rf_model.predict_proba(input_df)[0, 1])
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 1.2])
            with res_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=live_prob * 100,
                    number={'suffix': "%"},
                    title={'text': "Calculated Fraud Probability", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#d9534f" if live_prob >= selected_threshold else ("#f0ad4e" if live_prob >= 0.15 else "#5cb85c")},
                        'steps': [
                            {'range': [0, 15], 'color': "#e8f5e9"},
                            {'range': [15, selected_threshold * 100], 'color': "#fff9c4"},
                            {'range': [selected_threshold * 100, 100], 'color': "#ffebee"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': selected_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, width="stretch" if hasattr(st, "dataframe") else True)
                
            with res_col2:
                st.markdown("#### Automated Triage Decision")
                if live_prob < 0.15:
                    st.success("Action: AUTO-APPROVE (Low Risk)\n\nPayment is safe and approved immediately without bothering the customer.")
                elif live_prob < selected_threshold:
                    st.warning("Action: STEP-UP VERIFICATION (Medium Risk)\n\nAsks the customer for a quick one-time code (OTP) or fingerprint before approving.")
                else:
                    st.error("Action: ROUTE TO ANALYST REVIEW (High Risk)\n\nPayment is paused and sent to a fraud investigator for manual review.")
                    
                st.write(f"Spend Relative to Baseline: {ratio:.1f}x normal spend")
                st.write(f"Hardware Novelty: {'Unrecognized Device' if sim_new_device else 'Known Device'}")
                st.write(f"Cross-Border: {'International' if sim_cross_border else 'Domestic'}")

    # ---------------------------------------------------------
    # TAB 3: PORTFOLIO DIAGNOSTICS & FEATURE DRIVERS
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Model Comparison & Top Fraud Indicators")
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("#### Algorithm Benchmark Comparison")
            bench_df = pd.DataFrame([
                {
                    "Model": "Logistic Regression (Baseline)",
                    "ROC-AUC": f"{eval_summary['benchmark_models']['logistic_regression']['roc_auc']:.3f}",
                    "PR-AUC": f"{eval_summary['benchmark_models']['logistic_regression']['pr_auc']:.3f}",
                    "Default Precision": f"{eval_summary['benchmark_models']['logistic_regression']['precision_default']*100:.1f}%",
                    "Default Recall": f"{eval_summary['benchmark_models']['logistic_regression']['recall_default']*100:.1f}%"
                },
                {
                    "Model": "Random Forest (Champion Prototype)",
                    "ROC-AUC": f"{eval_summary['benchmark_models']['random_forest_champion']['roc_auc']:.3f}",
                    "PR-AUC": f"{eval_summary['benchmark_models']['random_forest_champion']['pr_auc']:.3f}",
                    "Default Precision": f"{eval_summary['benchmark_models']['random_forest_champion']['precision_default']*100:.1f}%",
                    "Default Recall": f"{eval_summary['benchmark_models']['random_forest_champion']['recall_default']*100:.1f}%"
                }
            ])
            st.table(bench_df)
            
            st.markdown("#### Tunable Threshold Operating Tradeoffs")
            thresh_table = pd.DataFrame(eval_summary['tunable_thresholds'])
            st.dataframe(thresh_table, width="stretch" if hasattr(st, "dataframe") else True)
            
        with diag_col2:
            st.markdown("#### Top Behavioral Drivers (Gini Importance)")
            feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            top_feats = feat_imp.tail(8)
            
            fig_imp = px.bar(
                x=top_feats.values,
                y=top_feats.index,
                orientation='h',
                labels={'x': 'Relative Gini Importance', 'y': 'Feature'},
                color=top_feats.values,
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_imp, width="stretch" if hasattr(st, "dataframe") else True)

        st.markdown("---")
        st.markdown("#### Predicted Fraud Risk Score Distribution")
        fig_hist = px.histogram(
            df_holdout,
            x='fraud_probability',
            color='is_fraud',
            nbins=50,
            labels={'fraud_probability': 'Predicted Fraud Probability', 'is_fraud': 'Actual Fraud Label'},
            color_discrete_map={0: '#2b5c8f', 1: '#d9534f'},
            barmode='overlay',
            log_y=True
        )
        fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig_hist, width="stretch" if hasattr(st, "dataframe") else True)

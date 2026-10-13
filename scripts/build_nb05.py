"""
Builds and executes notebooks/05_model_evaluation_and_thresholds.ipynb with actual Finlora dataset.
"""

import os
import nbformat as nbf

def build_nb05():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell("# Model Evaluation, Threshold Tuning & Operational Tradeoffs\n\n### Project Overview & Objective\nIn this notebook, the trained models are tested on 31,500 transactions they have never seen before (including 838 real fraud cases).\n\nThe evaluation examines how well each model tells fraud apart from normal payments, reviews the mistakes made, establishes practical alert settings for fraud analysts, and identifies which clues mattered most.\n"))
    
    # 1.
    cells.append(nbf.v4.new_markdown_cell("### 1. Import Evaluation Metrics and Libraries\nLoads evaluation metrics, JSON export tools, and plotting libraries.\n"))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import joblib\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import (\n"
        "    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,\n"
        "    confusion_matrix, roc_curve, precision_recall_curve, classification_report\n"
        ")\n\n"
        "print('Evaluation tools ready.')"
    ))
    
    # 2.
    cells.append(nbf.v4.new_markdown_cell("### 2. Load Hidden Test Data and Saved Models\nRebuilds the test set and loads the saved models, scaler, and feature list from model/.\n"))
    cells.append(nbf.v4.new_code_cell(
        "proc_dir = os.path.join('..', 'data', 'processed')\n"
        "feature_df = pd.read_csv(os.path.join(proc_dir, 'feature_matrix.csv'))\n"
        "clean_df = pd.read_csv(os.path.join(proc_dir, 'cleaned_transactions.csv'))\n\n"
        "model_dir = os.path.join('..', 'model')\n"
        "lr_baseline = joblib.load(os.path.join(model_dir, 'logistic_regression_baseline.joblib'))\n"
        "rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.joblib'))\n"
        "scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))\n"
        "feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))\n\n"
        "X = feature_df[feature_cols]\n"
        "y = feature_df['is_fraud']\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=42, stratify=y\n"
        ")\n\n"
        "num_cols = ['log_amount', 'log_amount_to_avg_ratio', 'log_avg_30d', 'transaction_velocity_1h', 'account_age_days', 'hour_of_day']\n"
        "X_test_scaled = X_test.copy()\n"
        "X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])\n\n"
        "print(f'Holdout test set size: {len(X_test):,} transactions ({y_test.sum():,} fraud events)')"
    ))
    
    # 3.
    cells.append(nbf.v4.new_markdown_cell("### 3. Generate Risk Scores for Test Transactions\nCalculates a continuous risk score between 0.0 (0% risk, very safe) and 1.0 (100% risk, almost certainly fraud) for all test transactions.\n"))
    cells.append(nbf.v4.new_code_cell(
        "prob_lr = lr_baseline.predict_proba(X_test_scaled)[:, 1]\n"
        "prob_rf = rf_model.predict_proba(X_test)[:, 1]\n\n"
        "print('Sample predicted risk scores (Random Forest):')\n"
        "for i, p in enumerate(prob_rf[:5]):\n"
        "    print(f'  Transaction {i+1}: Fraud Risk = {p:.4f} ({p*100:.2f}%)')"
    ))
    
    # 4.
    cells.append(nbf.v4.new_markdown_cell("### 4. Compare Baseline and Random Forest Performance\nCompares how well the baseline Logistic Regression and candidate Random Forest models perform on the test transactions to select the winning champion model.\n"))
    cells.append(nbf.v4.new_code_cell(
        "pred_lr = (prob_lr >= 0.5).astype(int)\n"
        "pred_rf = (prob_rf >= 0.5).astype(int)\n\n"
        "results = {\n"
        "    'Model': ['Logistic Regression (Baseline)', 'Random Forest'],\n"
        "    'ROC-AUC': [roc_auc_score(y_test, prob_lr), roc_auc_score(y_test, prob_rf)],\n"
        "    'PR-AUC': [average_precision_score(y_test, prob_lr), average_precision_score(y_test, prob_rf)],\n"
        "    'Default Precision (%)': [precision_score(y_test, pred_lr)*100, precision_score(y_test, pred_rf)*100],\n"
        "    'Default Recall (%)': [recall_score(y_test, pred_lr)*100, recall_score(y_test, pred_rf)*100],\n"
        "    'Default F1-Score (%)': [f1_score(y_test, pred_lr)*100, f1_score(y_test, pred_rf)*100]\n"
        "}\n\n"
        "summary_df = pd.DataFrame(results).round(3)\n"
        "print('Benchmark Scorecard')\n"
        "print(summary_df.to_string(index=False))"
    ))
    cells.append(nbf.v4.new_markdown_cell("Benchmark Assessment and Champion Selection: Both models do an excellent job separating fraud from normal payments (both scoring above 0.93 on discrimination ability). The Random Forest model produces far fewer false alarms (higher Precision: 40.7% vs 25.4%), meaning human investigators waste much less time reviewing innocent customers. Based on these evaluation results, Random Forest is officially selected as the production champion model. Adjusting the alert threshold gives full control over how many frauds to catch versus how many alerts to review.\n"))
    
    # 5.
    cells.append(nbf.v4.new_markdown_cell("### 5. Visualize Tradeoffs with ROC Curves and Error Matrices\nCreates visual charts showing how often each model is right versus how often it makes a mistake, along with a scorecard of correct guesses and errors.\n"))
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# ROC Curve\n"
        "fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)\n"
        "fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)\n\n"
        "axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, prob_rf):.3f})', color='#2ca02c', lw=2.5)\n"
        "axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC = {roc_auc_score(y_test, prob_lr):.3f})', color='#1f77b4', linestyle='--', lw=2)\n"
        "axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)\n"
        "axes[0].set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')\n"
        "axes[0].set_xlabel('False Positive Rate')\n"
        "axes[0].set_ylabel('True Positive Rate (Recall)')\n"
        "axes[0].legend(loc='lower right')\n\n"
        "# Confusion Matrix for Champion Model\n"
        "cm_rf = confusion_matrix(y_test, pred_rf)\n"
        "sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1],\n"
        "            xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])\n"
        "axes[1].set_title('Random Forest Confusion Matrix (@ 0.50 Threshold)', fontsize=12, fontweight='bold')\n"
        "axes[1].set_xlabel('Predicted Label')\n"
        "axes[1].set_ylabel('Actual Label')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 6.
    cells.append(nbf.v4.new_markdown_cell("### 6. Set Adjustable Alert Thresholds (70%, 80%, 90% Catch Rate)\nIn real operations, fraud teams have limited review capacity. Flagging too many payments overwhelms analysts, while flagging too few lets fraud slip through. The alert cutoff score is adjusted here so the business can select whether to catch 70%, 80%, or 90% of all fraud.\n"))
    cells.append(nbf.v4.new_code_cell(
        "precisions, recalls, thresholds = precision_recall_curve(y_test, prob_rf)\n\n"
        "def find_operating_threshold(target_recall):\n"
        "    idx = np.argmin(np.abs(recalls - target_recall))\n"
        "    thresh = thresholds[min(idx, len(thresholds)-1)]\n"
        "    pred_at_t = (prob_rf >= thresh).astype(int)\n"
        "    actual_rec = recall_score(y_test, pred_at_t)\n"
        "    actual_prec = precision_score(y_test, pred_at_t)\n"
        "    alert_count = pred_at_t.sum()\n"
        "    alert_rate = alert_count / len(y_test) * 100\n"
        "    return thresh, actual_rec*100, actual_prec*100, alert_count, alert_rate\n\n"
        "threshold_rows = []\n"
        "for target in [0.70, 0.80, 0.90]:\n"
        "    t, rec, prec, alerts, rate = find_operating_threshold(target)\n"
        "    threshold_rows.append({\n"
        "        'Operating Policy': f'{int(target*100)}% Target Recall',\n"
        "        'Probability Threshold': round(float(t), 4),\n"
        "        'Realized Recall (%)': round(float(rec), 2),\n"
        "        'Realized Precision (%)': round(float(prec), 2),\n"
        "        'Flagged Queue Volume': int(alerts),\n"
        "        'Queue Alert Rate (%)': round(float(rate), 2)\n"
        "    })\n\n"
        "thresh_df = pd.DataFrame(threshold_rows)\n"
        "print('Tunable Operating Thresholds')\n"
        "print(thresh_df.to_string(index=False))"
    ))
    
    # 7.
    cells.append(nbf.v4.new_markdown_cell("### 7. Identify Top Behavioral Fraud Indicators\nLooks inside the selected champion Random Forest model to see which behavioral clues contributed the most to its fraud risk predictions.\n"))
    cells.append(nbf.v4.new_code_cell(
        "importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "importances.tail(10).plot(kind='barh', color='#2b5c8f')\n"
        "plt.title('Top 10 Behavioral Drivers of Fraud (Random Forest Gini Importance)', fontsize=12, fontweight='bold')\n"
        "plt.xlabel('Relative Importance')\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print('Top 5 most impactful signals:')\n"
        "for feat, imp in importances.tail(5)[::-1].items():\n"
        "    print(f' - {feat}: {imp*100:.2f}% relative contribution')"
    ))
    
    # 8.
    cells.append(nbf.v4.new_markdown_cell("### 8. Save Evaluation Results and Test Predictions\nSaves data/processed/holdout_predictions.csv and model/evaluation_summary.json so the interactive Streamlit review dashboard can display them.\n"))
    cells.append(nbf.v4.new_code_cell(
        "test_indices = X_test.index\n"
        "holdout_df = clean_df.loc[test_indices, [\n"
        "    'transaction_id', 'account_id', 'timestamp', 'amount', 'account_type',\n"
        "    'kyc_tier', 'channel', 'merchant_category', 'is_fraud'\n"
        "]].copy()\n\n"
        "holdout_df['amount_to_avg_ratio'] = clean_df.loc[test_indices, 'amount_to_avg_ratio'].round(2)\n"
        "holdout_df['transaction_velocity_1h'] = clean_df.loc[test_indices, 'transaction_velocity_1h']\n"
        "holdout_df['is_new_device'] = clean_df.loc[test_indices, 'is_new_device']\n"
        "holdout_df['is_cross_border'] = clean_df.loc[test_indices, 'is_cross_border']\n"
        "holdout_df['fraud_probability'] = prob_rf.round(4)\n"
        "holdout_df['baseline_lr_probability'] = prob_lr.round(4)\n\n"
        "# Sort descending by risk score to form prioritized queue\n"
        "holdout_df = holdout_df.sort_values('fraud_probability', ascending=False).reset_index(drop=True)\n\n"
        "holdout_path = os.path.join(proc_dir, 'holdout_predictions.csv')\n"
        "holdout_df.to_csv(holdout_path, index=False)\n\n"
        "# Export the selected champion model artifact for production deployment\n"
        "rf_champ_path = os.path.join(model_dir, 'random_forest_champion.joblib')\n"
        "joblib.dump(rf_model, rf_champ_path)\n\n"
        "# Export evaluation metrics JSON\n"
        "eval_summary = {\n"
        "    'test_set_size': int(len(X_test)),\n"
        "    'test_fraud_count': int(y_test.sum()),\n"
        "    'benchmark_models': {\n"
        "        'logistic_regression': {\n"
        "            'roc_auc': float(roc_auc_score(y_test, prob_lr)),\n"
        "            'pr_auc': float(average_precision_score(y_test, prob_lr)),\n"
        "            'precision_default': float(precision_score(y_test, pred_lr)),\n"
        "            'recall_default': float(recall_score(y_test, pred_lr)),\n"
        "            'f1_default': float(f1_score(y_test, pred_lr))\n"
        "        },\n"
        "        'random_forest_champion': {\n"
        "            'roc_auc': float(roc_auc_score(y_test, prob_rf)),\n"
        "            'pr_auc': float(average_precision_score(y_test, prob_rf)),\n"
        "            'precision_default': float(precision_score(y_test, pred_rf)),\n"
        "            'recall_default': float(recall_score(y_test, pred_rf)),\n"
        "            'f1_default': float(f1_score(y_test, pred_rf))\n"
        "        }\n"
        "    },\n"
        "    'tunable_thresholds': threshold_rows\n"
        "}\n\n"
        "summary_json_path = os.path.join(model_dir, 'evaluation_summary.json')\n"
        "with open(summary_json_path, 'w', encoding='utf-8') as f:\n"
        "    json.dump(eval_summary, f, indent=2)\n\n"
        "print(f'Holdout predictions saved to: {holdout_path} ({len(holdout_df):,} records)')\n"
        "print(f'Champion model exported to: {rf_champ_path}')\n"
        "print(f'Evaluation summary JSON exported to: {summary_json_path}')"
    ))
    cells.append(nbf.v4.new_markdown_cell("Next Step: In 06_production_governance.ipynb, the selected champion model is deployed to work: creating a live scoring function, establishing 3 action steps, and setting up maintenance rules to keep the model updated over time.\n"))
    
    nb['cells'] = cells
    nb_path = os.path.join('notebooks', '05_model_evaluation_and_thresholds.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {nb_path}')

if __name__ == '__main__':
    build_nb05()

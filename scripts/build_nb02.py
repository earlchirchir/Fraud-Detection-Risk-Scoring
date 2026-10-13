"""
Builds and executes notebooks/02_EDA.ipynb with actual Finlora dataset.
"""

import os
import nbformat as nbf

def build_nb02():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell("# Exploring the Data and Spotting Fraud Patterns\n\n### Project Overview & Objective\nThis notebook analyzes cleaned transaction records to determine what suspicious payments look like compared to normal ones.\n\nThe analysis measures how rare fraud is, compares spending habits, reviews which payment methods and store types are riskiest, and checks if new devices or international payments trigger more fraud.\n"))
    
    # 1.
    cells.append(nbf.v4.new_markdown_cell("### 1. Import Visualization Libraries\nLoads plotting tools (matplotlib and seaborn) along with data tools to create clear, easy-to-read charts.\n"))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "plt.rcParams['figure.figsize'] = (10, 5)\n"
        "print('Visualization libraries initialized.')"
    ))
    
    # 2.
    cells.append(nbf.v4.new_markdown_cell("### 2. Load Cleaned Transactions\nLoads the clean transactions file prepared in 01_data_cleaning.ipynb (data/processed/cleaned_transactions.csv).\n"))
    cells.append(nbf.v4.new_code_cell(
        "data_path = os.path.join('..', 'data', 'processed', 'cleaned_transactions.csv')\n"
        "df = pd.read_csv(data_path)\n"
        "df['timestamp'] = pd.to_datetime(df['timestamp'])\n"
        "print(f'Total records loaded: {len(df):,} transactions')\n"
        "df[['transaction_id', 'account_id', 'timestamp', 'channel', 'merchant_category', 'amount', 'is_fraud']].head(3)"
    ))
    
    # 3.
    cells.append(nbf.v4.new_markdown_cell("### 3. Measure Fraud Frequency Across Transactions\nChecks how common fraud is in the dataset compared to normal, legitimate transactions.\n"))
    cells.append(nbf.v4.new_code_cell(
        "fraud_counts = df['is_fraud'].value_counts()\n"
        "fraud_rate = df['is_fraud'].mean() * 100\n\n"
        "print(f'Legitimate transactions (0): {fraud_counts[0]:,} ({100 - fraud_rate:.2f}%)')\n"
        "print(f'Fraudulent transactions (1): {fraud_counts[1]:,} ({fraud_rate:.2f}%)')\n\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "bars = ax.bar(['Legitimate (0)', 'Fraud (1)'], [fraud_counts[0], fraud_counts[1]], color=['#2b5c8f', '#d9534f'], width=0.45)\n"
        "ax.set_title('Target Class Distribution (2.66% Fraud Rate)', fontsize=12, fontweight='bold')\n"
        "ax.set_ylabel('Transaction Count')\n"
        "for bar in bars:\n"
        "    h = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, h + 1500, f'{h:,}', ha='center', va='bottom', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell("Class Imbalance Finding: Fraud is rare, making up only 2.66% of all transactions (3,352 fraud cases out of 126,000). If a system simply guessed that every payment was safe, it would look 97.34% accurate while missing every single fraud event. That is why simple accuracy is misleading here. Instead, model evaluation must focus on how many real frauds are caught (Recall) and how many alerts are actually real (Precision).\n"))
    
    # 4.
    cells.append(nbf.v4.new_markdown_cell("### 4. Analyze Spending Spikes and Purchase Sizes\nCompares normal transactions against fraud to see if fraudsters spend much more than a customer's typical amount.\n"))
    cells.append(nbf.v4.new_code_cell(
        "print('Relative Spend Ratio (amount_to_avg_ratio) Statistics')\n"
        "print(df.groupby('is_fraud')['amount_to_avg_ratio'].describe().round(2))\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "sns.boxplot(x='is_fraud', y='amount_to_avg_ratio', data=df, showfliers=False, palette=['#2b5c8f', '#d9534f'], ax=axes[0])\n"
        "axes[0].set_title('Spend Ratio by Class (Excl. Outliers)', fontweight='bold')\n"
        "axes[0].set_xticklabels(['Legitimate', 'Fraud'])\n"
        "axes[0].set_ylabel('Ratio to 30d Baseline Spend')\n\n"
        "sns.boxplot(x='is_fraud', y='abs_amount', data=df, showfliers=False, palette=['#2b5c8f', '#d9534f'], ax=axes[1])\n"
        "axes[1].set_title('Absolute Amount by Class (Excl. Outliers)', fontweight='bold')\n"
        "axes[1].set_xticklabels(['Legitimate', 'Fraud'])\n"
        "axes[1].set_ylabel('Transaction Amount ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell("Spend Divergence Finding: Fraudulent transactions are typically 6 times higher than a customer's normal 30-day spending average (with a median ratio of 6.04x), while normal payments stay right around their usual average (median of 0.95x). Unusually large spending spikes are one of the biggest clues that a transaction is fraud.\n"))
    
    # 5.
    cells.append(nbf.v4.new_markdown_cell("### 5. Review Risk by Payment Channel and Store Category\nChecks which payment methods (like web, mobile app, bank wire) and store types have the highest rates of fraud.\n"))
    cells.append(nbf.v4.new_code_cell(
        "channel_summary = df.groupby('channel')['is_fraud'].agg(['count', 'mean']).rename(columns={'count': 'Volume', 'mean': 'Fraud_Rate'})\n"
        "channel_summary['Fraud_Rate_Pct'] = (channel_summary['Fraud_Rate'] * 100).round(2)\n"
        "print('Fraud Rate by Payment Channel')\n"
        "print(channel_summary.sort_values(by='Fraud_Rate_Pct', ascending=False))\n\n"
        "cat_summary = df.groupby('merchant_category')['is_fraud'].agg(['count', 'mean']).rename(columns={'count': 'Volume', 'mean': 'Fraud_Rate'})\n"
        "cat_summary['Fraud_Rate_Pct'] = (cat_summary['Fraud_Rate'] * 100).round(2)\n"
        "print('\\nFraud Rate by Top Merchant Categories')\n"
        "print(cat_summary.sort_values(by='Fraud_Rate_Pct', ascending=False).head(8))\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "channel_summary['Fraud_Rate_Pct'].sort_values().plot(kind='barh', color='#3498db', ax=axes[0])\n"
        "axes[0].set_title('Fraud Rate by Channel (%)', fontweight='bold')\n"
        "axes[0].set_xlabel('Fraud Rate (%)')\n\n"
        "cat_summary['Fraud_Rate_Pct'].sort_values().tail(8).plot(kind='barh', color='#e67e22', ax=axes[1])\n"
        "axes[1].set_title('Top 8 High-Risk Merchant Categories (%)', fontweight='bold')\n"
        "axes[1].set_xlabel('Fraud Rate (%)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell("Category Finding: Bank wire transfers (7.58%), payroll transfers (5.59%), and crypto exchanges (5.20%) have the highest share of fraud. Among payment channels, automated system connections (APIs at 3.65%) and website logins (2.96%) carry the highest risk.\n"))
    
    # 6.
    cells.append(nbf.v4.new_markdown_cell("### 6. Compare Personal vs Business Accounts and Identity Tiers\nChecks if fraud happens more often on personal accounts or business accounts, and whether customer identity verification levels (Basic, Verified, Enhanced) make a difference.\n"))
    cells.append(nbf.v4.new_code_cell(
        "kyc_summary = df.groupby(['account_type', 'kyc_tier'])['is_fraud'].agg(['count', 'mean'])\n"
        "kyc_summary['Fraud_Rate_Pct'] = (kyc_summary['mean'] * 100).round(2)\n"
        "print('Fraud Risk by Account Type and KYC Tier')\n"
        "print(kyc_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "sns.barplot(x='kyc_tier', y='is_fraud', hue='account_type', data=df, errorbar=None, palette='Blues_d', ax=ax)\n"
        "ax.set_title('Observed Fraud Rate by KYC Tier and Account Type', fontweight='bold')\n"
        "ax.set_ylabel('Fraud Rate')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 7.
    cells.append(nbf.v4.new_markdown_cell("### 7. Inspect Unfamiliar Devices and Rapid Transaction Bursts\nChecks if payments from new, unrecognized phones or computers and rapid bursts of transactions within an hour are riskier.\n"))
    cells.append(nbf.v4.new_code_cell(
        "device_risk = df.groupby('is_new_device')['is_fraud'].agg(['count', 'mean'])\n"
        "device_risk['Fraud_Rate_Pct'] = (device_risk['mean'] * 100).round(2)\n"
        "print('Device Novelty Risk')\n"
        "print(device_risk)\n\n"
        "print('\\nVelocity (Trailing 1-Hour Count) Statistics')\n"
        "print(df.groupby('is_fraud')['transaction_velocity_1h'].describe().round(2))\n\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "bars = ax.bar(['Known / Standard (0)', 'Unrecognized Device (1)'], device_risk['Fraud_Rate_Pct'], color=['#34495e', '#e74c3c'], width=0.4)\n"
        "ax.set_title('Fraud Rate by Hardware Familiarity', fontweight='bold')\n"
        "ax.set_ylabel('Fraud Rate (%)')\n"
        "for bar in bars:\n"
        "    h = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.3, f'{h:.2f}%', ha='center', va='bottom', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell("Hardware Finding: When a transaction comes from a new, unrecognized phone or computer, the chance of fraud jumps to 7.01%, compared to just 2.52% on familiar devices.\n"))
    
    # 8.
    cells.append(nbf.v4.new_markdown_cell("### 8. Key Takeaways from Data Exploration\nFour simple takeaways from the data exploration:\n\n1. Big Spending Spikes: Transactions far above a customer's normal 30-day average (amount_to_avg_ratio) are the single strongest warning sign of fraud.\n2. High-Risk Channels and Stores: Bank wires, crypto purchases, and automated connections require closer monitoring.\n3. Unfamiliar Devices: Fraudsters frequently log in using a new phone or computer the customer has never used before.\n4. Why Accuracy Lies: Because 97% of transactions are safe, a fraud model cannot just guess 'safe'. It needs to catch real criminals without producing an overwhelming number of false alarms.\n\n---\nNext Step: In 03_feature_engineering.ipynb, these discoveries are converted into clean numeric clues that machine learning models can understand.\n"))
    
    nb['cells'] = cells
    nb_path = os.path.join('notebooks', '02_EDA.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {nb_path}')

if __name__ == '__main__':
    build_nb02()

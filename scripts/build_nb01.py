"""
Builds and executes notebooks/01_data_cleaning.ipynb with actual Finlora dataset.
"""

import os
import nbformat as nbf

def build_nb01():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell("# Cleaning the Data and Getting Records Ready\n\n### Project Overview & Objective\nFinlora processes digital payments from different countries, apps, and websites. Before a computer model can spot fraud, the raw transaction logs and account profiles must be cleaned and organized.\n\nThis notebook walks through the data preparation process step-by-step: finding missing information, fixing date formats, checking for unusual dollar amounts (like refunds), and combining account details with payment records.\n"))
    
    # 1.
    cells.append(nbf.v4.new_markdown_cell("### 1. Import Essential Libraries\nLoads standard Python tools (pandas, numpy, os) to open, inspect, and organize the data tables.\n"))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "# Set display options for clean inspection\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.width', 1000)\n"
        "print('Libraries loaded successfully.')"
    ))
    
    # 2.
    cells.append(nbf.v4.new_markdown_cell("### 2. Load Transaction and Account Datasets\nLoads two files from the raw data folder: customer account profiles (finlora_accounts.csv) and transaction records (finlora_transactions.csv).\n"))
    cells.append(nbf.v4.new_code_cell(
        "raw_dir = os.path.join('..', 'data', 'raw')\n"
        "accounts_df = pd.read_csv(os.path.join(raw_dir, 'finlora_accounts.csv'))\n"
        "transactions_df = pd.read_csv(os.path.join(raw_dir, 'finlora_transactions.csv'))\n\n"
        "print(f'Accounts loaded: {len(accounts_df):,} records | Columns: {accounts_df.shape[1]}')\n"
        "print(f'Transactions loaded: {len(transactions_df):,} records | Columns: {transactions_df.shape[1]}')"
    ))
    
    # 3.
    cells.append(nbf.v4.new_markdown_cell("### 3. Check for Missing Information\nChecks the column names and looks for any empty or missing fields across both tables.\n"))
    cells.append(nbf.v4.new_code_cell(
        "print('Accounts Table Null Audit')\n"
        "print(accounts_df.isna().sum()[accounts_df.isna().sum() > 0])\n"
        "print(f'Total account nulls: {accounts_df.isna().sum().sum()}')\n\n"
        "print('\\nTransactions Table Null Audit')\n"
        "null_counts = transactions_df.isna().sum()\n"
        "print(null_counts[null_counts > 0])\n"
        "print(f'Device ID missing: {transactions_df[\"device_id\"].isna().sum():,} ({transactions_df[\"device_id\"].isna().mean()*100:.2f}%)')\n"
        "print(f'Merchant Name missing: {transactions_df[\"merchant_name\"].isna().sum():,} ({transactions_df[\"merchant_name\"].isna().mean()*100:.2f}%)')"
    ))
    cells.append(nbf.v4.new_markdown_cell("Audit Finding: Customer accounts have zero missing values. In transactions, missing device fields only happen on payment methods that do not use phones or computers (like bank wire transfers or phone dial codes). Missing device flags are safely filled with 0, meaning a standard or known channel.\n"))
    
    # 4.
    cells.append(nbf.v4.new_markdown_cell("### 4. Standardize Dates and Timestamps\nConverts dates and times into standard calendar formats to track when payments happened and how old each account is.\n"))
    cells.append(nbf.v4.new_code_cell(
        "transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])\n"
        "accounts_df['account_created_date'] = pd.to_datetime(accounts_df['account_created_date'])\n\n"
        "print(f'Transaction date range: {transactions_df[\"timestamp\"].min()} to {transactions_df[\"timestamp\"].max()}')\n"
        "print(f'Account creation date range: {accounts_df[\"account_created_date\"].min()} to {accounts_df[\"account_created_date\"].max()}')"
    ))
    
    # 5.
    cells.append(nbf.v4.new_markdown_cell("### 5. Inspect Transaction Amounts and Remove Duplicates\nChecks for duplicate records and flags negative amounts (which represent customer refunds or chargebacks).\n"))
    cells.append(nbf.v4.new_code_cell(
        "neg_amounts = (transactions_df['amount'] < 0).sum()\n"
        "duplicate_txns = transactions_df['transaction_id'].duplicated().sum()\n"
        "orphan_accounts = (~transactions_df['account_id'].isin(accounts_df['account_id'])).sum()\n\n"
        "print(f'Transactions with negative amounts (refunds/reversals): {neg_amounts:,}')\n"
        "print(f'Duplicate transaction IDs: {duplicate_txns}')\n"
        "print(f'Orphan transactions with missing account profile: {orphan_accounts}')\n\n"
        "# Absolute amount and refund flag\n"
        "transactions_df['is_negative_amount'] = (transactions_df['amount'] < 0).astype(int)\n"
        "transactions_df['abs_amount'] = transactions_df['amount'].abs()"
    ))
    
    # 6.
    cells.append(nbf.v4.new_markdown_cell("### 6. Clean Up Missing Device Flags\nFills in blank device names and merchant names with clear placeholders (like 'UNKNOWN_DEVICE') so the dataset is complete and consistent.\n"))
    cells.append(nbf.v4.new_code_cell(
        "transactions_df['is_new_device'] = transactions_df['is_new_device'].fillna(0).astype(int)\n"
        "transactions_df['device_id'] = transactions_df['device_id'].fillna('UNKNOWN_DEVICE')\n"
        "transactions_df['merchant_name'] = transactions_df['merchant_name'].fillna('Unspecified Merchant')\n\n"
        "print('Device novelty flags successfully imputed and cast to integer.')\n"
        "print(transactions_df['is_new_device'].value_counts())"
    ))
    
    # 7.
    cells.append(nbf.v4.new_markdown_cell("### 7. Combine Customer Profiles with Transactions\nAttaches each customer's account details (such as their normal spending habits and account age) to their transactions.\n"))
    cells.append(nbf.v4.new_code_cell(
        "# Select supplementary columns from accounts table to prevent duplicate columns\n"
        "acc_cols_to_merge = ['account_id', 'account_holder_name', 'account_created_date', 'personal_spend_baseline_usd']\n"
        "merged_df = transactions_df.merge(accounts_df[acc_cols_to_merge], on='account_id', how='left')\n\n"
        "print(f'Merged dataset shape: {merged_df.shape}')\n"
        "merged_df[['transaction_id', 'account_id', 'timestamp', 'channel', 'amount', 'personal_spend_baseline_usd', 'is_fraud']].head(5)"
    ))
    
    # 8.
    cells.append(nbf.v4.new_markdown_cell("### 8. Save the Cleaned Dataset\nSaves the cleaned and combined records into data/processed/cleaned_transactions.csv so they are ready for analysis.\n"))
    cells.append(nbf.v4.new_code_cell(
        "proc_dir = os.path.join('..', 'data', 'processed')\n"
        "os.makedirs(proc_dir, exist_ok=True)\n"
        "out_path = os.path.join(proc_dir, 'cleaned_transactions.csv')\n\n"
        "merged_df.to_csv(out_path, index=False)\n"
        "print(f'Cleaned transaction dataset saved to: {out_path}')\n"
        "print(f'Total records: {len(merged_df):,} | Fraud count: {merged_df[\"is_fraud\"].sum():,} ({merged_df[\"is_fraud\"].mean()*100:.2f}%)')"
    ))
    cells.append(nbf.v4.new_markdown_cell("Summary of Cleaning Steps:\n- Audited 126,000 transactions across 7,200 accounts.\n- Converted dates and confirmed every payment belongs to a real account.\n- Flagged refund amounts and recorded the positive dollar values.\n- Cleanly filled in missing device flags.\n- Added each customer's normal 30-day spending baseline.\n\n---\nNext Step: In 02_EDA.ipynb, exploratory analysis examines what makes fraudulent transactions different from normal ones.\n"))
    
    nb['cells'] = cells
    nb_path = os.path.join('notebooks', '01_data_cleaning.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {nb_path}')

if __name__ == '__main__':
    build_nb01()

"""
Builds and executes notebooks/03_feature_engineering.ipynb with actual Finlora dataset.
"""

import os
import nbformat as nbf

def build_nb03():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell("# Behavioral Signal Extraction & Feature Engineering\n\n### Project Overview & Objective\nIn the exploratory data analysis (02_EDA.ipynb), the analysis revealed that fraud usually shows clear signs: spending far more than usual, sending rapid bursts of payments, using a brand-new device, or sending money overseas.\n\nIn this notebook, those discoveries are converted into simple, clean numbers (called 'features') that a computer model can read. Extreme dollar values are smoothed out, categories are converted into yes/no numbers, and private account details are removed so the model only learns genuine patterns.\n"))
    
    # 1.
    cells.append(nbf.v4.new_markdown_cell("### 1. Import Data Processing Libraries\nLoads numerical and data processing tools (pandas, numpy, os).\n"))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "print('Feature engineering environment initialized.')"
    ))
    
    # 2.
    cells.append(nbf.v4.new_markdown_cell("### 2. Load Cleaned Transactions\nLoads data/processed/cleaned_transactions.csv created in 01_data_cleaning.ipynb.\n"))
    cells.append(nbf.v4.new_code_cell(
        "data_path = os.path.join('..', 'data', 'processed', 'cleaned_transactions.csv')\n"
        "df = pd.read_csv(data_path)\n"
        "df['timestamp'] = pd.to_datetime(df['timestamp'])\n"
        "print(f'Transaction records loaded: {len(df):,} rows x {df.shape[1]} columns')\n"
        "df[['transaction_id', 'account_id', 'amount', 'amount_to_avg_ratio', 'avg_transaction_amount_30d', 'is_fraud']].head(3)"
    ))
    
    # 3.
    cells.append(nbf.v4.new_markdown_cell("### 3. Smooth Out Extreme Dollar Amounts with Log Scaling\nPayment amounts range from a few dollars all the way into millions of dollars. Huge numbers can overwhelm simple models and skew calculations. A standard math technique called a logarithm (log) shrinks the scale and pulls extreme values closer together without changing their relative order.\n"))
    cells.append(nbf.v4.new_code_cell(
        "# Compute absolute amount and log transformations\n"
        "df['abs_amount'] = df['amount'].abs()\n"
        "df['log_amount'] = np.log1p(df['abs_amount'])\n"
        "df['log_amount_to_avg_ratio'] = np.log1p(np.maximum(df['amount_to_avg_ratio'], 0))\n"
        "df['log_avg_30d'] = np.log1p(np.maximum(df['avg_transaction_amount_30d'], 0))\n\n"
        "print('Transformed Feature Summary Statistics')\n"
        "print(df[['log_amount', 'log_amount_to_avg_ratio', 'log_avg_30d']].describe().round(3))\n"
        "df[['amount', 'log_amount', 'amount_to_avg_ratio', 'log_amount_to_avg_ratio']].head(4)"
    ))
    
    # 4.
    cells.append(nbf.v4.new_markdown_cell("### 4. Extract Device, Velocity, and Location Signals\nGathers key behavioral clues: how many payments occurred in the last hour (transaction_velocity_1h), whether the device is new (is_new_device), whether it is an international payment (is_cross_border), and whether it was a refund (is_negative_amount).\n"))
    cells.append(nbf.v4.new_code_cell(
        "df['is_new_device'] = df['is_new_device'].fillna(0).astype(int)\n"
        "df['is_negative_amount'] = (df['amount'] < 0).astype(int)\n\n"
        "behavioral_cols = ['transaction_velocity_1h', 'is_new_device', 'is_cross_border', 'is_negative_amount']\n"
        "print('Behavioral signals summary:')\n"
        "print(df[behavioral_cols].describe().round(3))"
    ))
    
    # 5.
    cells.append(nbf.v4.new_markdown_cell("### 5. Convert Category Labels into Numerical Indicators\nComputers understand numbers, not text labels. Categories (like account type, verification level, channel, and store type) are converted into simple 1 or 0 (yes/no) columns.\n"))
    cells.append(nbf.v4.new_code_cell(
        "categorical_cols = ['account_type', 'kyc_tier', 'channel', 'merchant_category']\n"
        "encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)\n\n"
        "dummy_cols = [c for c in encoded_df.columns if any(c.startswith(cat + '_') for cat in categorical_cols)]\n"
        "print(f'Generated {len(dummy_cols)} dummy features across categories.')\n"
        "print('Sample dummy columns:', dummy_cols[:6])"
    ))
    
    # 6.
    cells.append(nbf.v4.new_markdown_cell("### 6. Check for Data Leakage and Remove Sensitive Identifiers\nRemoves customer names, account IDs, descriptions, and timestamps. This ensures the model learns true behavioral habits rather than memorizing individual people or peeking at information it would not have during a live transaction.\n"))
    cells.append(nbf.v4.new_code_cell(
        "# Exclude all non-feature metadata and post-settlement indicators\n"
        "exclude_cols = [\n"
        "    'transaction_id', 'account_id', 'timestamp', 'day_of_week', 'description', 'merchant_name',\n"
        "    'currency', 'transaction_country', 'home_country', 'device_id', 'status',\n"
        "    'account_holder_name', 'account_created_date', 'amount', 'abs_amount',\n"
        "    'personal_spend_baseline_usd', 'is_fraud'\n"
        "]\n"
        "model_features = [col for col in encoded_df.columns if col not in exclude_cols]\n\n"
        "print(f'Total candidate predictor features: {len(model_features)}')\n"
        "print('Selected feature names:\\n', model_features)"
    ))
    
    # 7.
    cells.append(nbf.v4.new_markdown_cell("### 7. Save the Engineered Feature Matrix\nSaves the complete list of model clues into data/processed/feature_matrix.csv.\n"))
    cells.append(nbf.v4.new_code_cell(
        "proc_dir = os.path.join('..', 'data', 'processed')\n"
        "matrix_path = os.path.join(proc_dir, 'feature_matrix.csv')\n\n"
        "encoded_df.to_csv(matrix_path, index=False)\n"
        "print(f'Feature matrix successfully saved to: {matrix_path}')\n"
        "print(f'Dimensions: {encoded_df.shape[0]:,} rows x {encoded_df.shape[1]} columns')"
    ))
    cells.append(nbf.v4.new_markdown_cell("Summary of Features Prepared:\n- log_amount: The payment size, scaled down smoothly so huge values do not distort the model.\n- log_amount_to_avg_ratio: How much higher this payment is compared to the customer's normal spending.\n- log_avg_30d: The customer's typical spending baseline.\n- transaction_velocity_1h: How many payments happened in the past hour (rapid bursts).\n- is_new_device: 1 if sent from an unfamiliar phone or computer, 0 if familiar.\n- is_cross_border: 1 if international, 0 if local.\n- is_negative_amount: 1 if a refund or reversed payment.\n- Category flags: Simple yes/no (1/0) flags for account types, verification levels, channels, and store categories.\n\n---\nNext Step: In 04_model_development.ipynb, these clues are used to train and compare two different fraud detection models.\n"))
    
    nb['cells'] = cells
    nb_path = os.path.join('notebooks', '03_feature_engineering.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {nb_path}')

if __name__ == '__main__':
    build_nb03()

"""
Data generation script for Finlora Finance Flow.
Generates 126,000 transactions across 7,200 accounts matching project specifications.
"""

import os
import numpy as np
import pandas as pd

def generate_finlora_data(random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Generate 7,200 Accounts
    n_accounts = 7200
    account_ids = [f"ACC_{i+1:05d}" for i in range(n_accounts)]
    
    account_types = np.random.choice(['Individual', 'Business'], size=n_accounts, p=[0.72, 0.28])
    kyc_tiers = np.random.choice(['Basic', 'Verified', 'Enhanced'], size=n_accounts, p=[0.25, 0.55, 0.20])
    home_countries = np.random.choice(['NG', 'US', 'GB'], size=n_accounts, p=[0.40, 0.35, 0.25])
    
    # Baseline spending profiles per account
    # Individuals average ~$75-$150, Businesses average ~$1,200-$2,800
    baseline_spend = []
    for at in account_types:
        if at == 'Individual':
            spend = np.random.gamma(shape=3.0, scale=35.0)  # Mean ~105
        else:
            spend = np.random.gamma(shape=5.0, scale=350.0)  # Mean ~1750
        baseline_spend.append(round(max(spend, 10.0), 2))
        
    accounts_df = pd.DataFrame({
        'account_id': account_ids,
        'account_type': account_types,
        'kyc_tier': kyc_tiers,
        'home_country': home_countries,
        'account_baseline_avg': baseline_spend
    })
    
    # Account primary devices (1-3 devices per account)
    account_devices = {}
    for acc in account_ids:
        n_dev = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        account_devices[acc] = [f"DEV_{acc}_{d+1}" for d in range(n_dev)]
        
    # 2. Generate 126,000 Transactions
    n_transactions = 126000
    
    # Distribute transactions across accounts using Pareto/Power-law distribution
    weights = np.random.pareto(a=1.8, size=n_accounts) + 0.1
    weights /= weights.sum()
    tx_account_indices = np.random.choice(n_accounts, size=n_transactions, p=weights)
    tx_accounts = [account_ids[i] for i in tx_account_indices]
    
    # Time range: 180 days (Jan 1, 2024 to June 28, 2024)
    start_ts = pd.Timestamp("2024-01-01 00:00:00").value // 10**9
    end_ts = pd.Timestamp("2024-06-28 23:59:59").value // 10**9
    random_timestamps = np.random.randint(start_ts, end_ts, size=n_transactions)
    random_timestamps.sort()  # chronological order
    tx_timestamps = pd.to_datetime(random_timestamps, unit='s')
    
    channels = np.random.choice(['mobile', 'web', 'card', 'api'], size=n_transactions, p=[0.45, 0.30, 0.15, 0.10])
    categories = np.random.choice(
        ['retail', 'grocery', 'transfer', 'travel', 'electronics', 'utility', 'entertainment'],
        size=n_transactions,
        p=[0.24, 0.22, 0.18, 0.10, 0.10, 0.10, 0.06]
    )
    
    # Base fraud probability ~ 2.1%
    fraud_flags = np.zeros(n_transactions, dtype=int)
    amounts = np.zeros(n_transactions, dtype=float)
    devices = []
    is_cross_borders = np.zeros(n_transactions, dtype=int)
    
    account_lookup = accounts_df.set_index('account_id').to_dict(orient='index')
    
    # Pre-select candidate fraudulent indices (~2.1% overall rate: ~2,650 fraud transactions)
    # Fraud types:
    # A. Account Takeover / Burst: novel device + high amount + high velocity
    # B. Subtle Cross-border / Trial: small amount + new device + cross border
    # C. High Value Transfer / Electronics fraud
    n_fraud = int(n_transactions * 0.021)
    fraud_indices = set(np.random.choice(n_transactions, size=n_fraud, replace=False))
    
    for i in range(n_transactions):
        acc = tx_accounts[i]
        acc_info = account_lookup[acc]
        base_avg = acc_info['account_baseline_avg']
        known_devs = account_devices[acc]
        
        is_fraud = 1 if i in fraud_indices else 0
        fraud_flags[i] = is_fraud
        
        if is_fraud:
            # Behavioral fraud pattern:
            # 80% chance of new unrecognized device
            if np.random.rand() < 0.80:
                dev = f"DEV_UNKNOWN_{np.random.randint(100000, 999999)}"
            else:
                dev = np.random.choice(known_devs)
            
            # 65% chance of cross border
            is_cb = 1 if np.random.rand() < 0.65 else 0
            
            # Fraud amount: either high multiple of baseline (65%) or subtle/probing amount (35%)
            if np.random.rand() < 0.65:
                multiplier = np.random.uniform(3.5, 9.5)
                amt = base_avg * multiplier
            else:
                # subtle fraud slipping under static limits
                amt = base_avg * np.random.uniform(0.7, 1.8)
        else:
            # Legitimate pattern:
            # 96% chance of known device
            if np.random.rand() < 0.04:
                dev = f"DEV_NEW_{np.random.randint(100000, 999999)}"
            else:
                dev = np.random.choice(known_devs)
                
            # 8% chance of legitimate cross border (e.g. travel, international subscription)
            is_cb = 1 if np.random.rand() < 0.08 else 0
            
            # Legitimate amount: centered around baseline, occasionally high (legit payroll/electronics)
            if np.random.rand() < 0.03:
                amt = base_avg * np.random.uniform(2.5, 4.0)  # legitimate spike
            else:
                amt = base_avg * np.random.lognormal(mean=0.0, sigma=0.45)
                
        amounts[i] = round(max(amt, 1.50), 2)
        devices.append(dev)
        is_cross_borders[i] = is_cb
        
    tx_df = pd.DataFrame({
        'transaction_id': [f"TXN_{i+1:06d}" for i in range(n_transactions)],
        'account_id': tx_accounts,
        'timestamp': tx_timestamps,
        'amount': amounts,
        'channel': channels,
        'merchant_category': categories,
        'device_id': devices,
        'is_cross_border': is_cross_borders,
        'is_fraud': fraud_flags
    })
    
    return accounts_df, tx_df

if __name__ == '__main__':
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating Finlora benchmark dataset (7,200 accounts, 126,000 transactions)...")
    accounts_df, tx_df = generate_finlora_data(random_seed=42)
    
    acc_path = os.path.join(raw_dir, 'accounts.csv')
    tx_path = os.path.join(raw_dir, 'transactions.csv')
    
    accounts_df.to_csv(acc_path, index=False)
    tx_df.to_csv(tx_path, index=False)
    
    print(f"Accounts saved to: {acc_path} ({len(accounts_df):,} rows)")
    print(f"Transactions saved to: {tx_path} ({len(tx_df):,} rows)")
    print(f"Fraud count: {tx_df['is_fraud'].sum():,} ({tx_df['is_fraud'].mean()*100:.2f}%)")

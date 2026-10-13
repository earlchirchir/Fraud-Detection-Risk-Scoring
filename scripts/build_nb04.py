"""
Builds and executes notebooks/04_model_development.ipynb with actual Finlora dataset.
"""

import os
import nbformat as nbf

def build_nb04():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Scikit-Learn Model Development & Algorithmic Benchmarking\n\n### Project Overview & Objective\nIn this notebook, two machine learning models are trained to recognize fraud:\n\n1. Logistic Regression (Baseline): A simple, fast formula that gives a clear score and shows exactly how each clue pushes risk up or down.\n2. Random Forest (Ensemble Model): A team of 300 decision trees working together. It spots subtle combinations of clues (such as high velocity from an unfamiliar device during an international wire transfer).\n\nThe data is split to test the models fairly, numerical scales are adjusted, both models are trained, and the artifacts are saved for production use.\n"))
    
    # 1.
    cells.append(nbf.v4.new_markdown_cell("### 1. Import Machine Learning Libraries\nLoads scikit-learn tools to split data, scale numbers, train the models, and save them to files.\n"))
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import joblib\n\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "print('Machine learning environment ready.')"
    ))
    
    # 2.
    cells.append(nbf.v4.new_markdown_cell("### 2. Load Engineered Features\nLoads data/processed/feature_matrix.csv created in 03_feature_engineering.ipynb.\n"))
    cells.append(nbf.v4.new_code_cell(
        "data_path = os.path.join('..', 'data', 'processed', 'feature_matrix.csv')\n"
        "df = pd.read_csv(data_path)\n"
        "print(f'Feature dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')\n"
        "df[['transaction_id', 'log_amount', 'log_amount_to_avg_ratio', 'transaction_velocity_1h', 'is_new_device', 'is_fraud']].head(3)"
    ))
    
    # 3.
    cells.append(nbf.v4.new_markdown_cell("### 3. Separate Input Features from the Fraud Label\nSeparates the target prediction column (whether a payment is fraudulent) from all the clues the model is allowed to use.\n"))
    cells.append(nbf.v4.new_code_cell(
        "exclude_cols = [\n"
        "    'transaction_id', 'account_id', 'timestamp', 'day_of_week', 'description', 'merchant_name',\n"
        "    'currency', 'transaction_country', 'home_country', 'device_id', 'status',\n"
        "    'account_holder_name', 'account_created_date', 'amount', 'abs_amount',\n"
        "    'personal_spend_baseline_usd', 'is_fraud'\n"
        "]\n"
        "feature_cols = [c for c in df.columns if c not in exclude_cols]\n\n"
        "X = df[feature_cols]\n"
        "y = df['is_fraud']\n\n"
        "print(f'Predictor feature count: {len(feature_cols)}')\n"
        "print(f'Target distribution: {y.value_counts().to_dict()}')"
    ))
    
    # 4.
    cells.append(nbf.v4.new_markdown_cell("### 4. Split Data into Training and Test Sets\nDivides the data into two groups: 75% for the models to study and learn from (the training set, 94,500 records), and 25% kept hidden away to test them fairly afterwards (the holdout test set, 31,500 records). Both groups preserve the exact same 2.66% fraud rate.\n"))
    cells.append(nbf.v4.new_code_cell(
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=42, stratify=y\n"
        ")\n\n"
        "print(f'Training set: {len(X_train):,} observations ({y_train.sum():,} fraud cases, {y_train.mean()*100:.2f}%)')\n"
        "print(f'Holdout test set: {len(X_test):,} observations ({y_test.sum():,} fraud cases, {y_test.mean()*100:.2f}%)')"
    ))
    
    # 5.
    cells.append(nbf.v4.new_markdown_cell("### 5. Scale Numerical Features for Fair Comparison\nSome numbers are in the thousands (like dollar amounts), while others are small counts (like 1 to 5 transactions per hour). Rescaling puts all clues on a fair, comparable footing so no single large number dominates just because of its units.\n"))
    cells.append(nbf.v4.new_code_cell(
        "num_cols = ['log_amount', 'log_amount_to_avg_ratio', 'log_avg_30d', 'transaction_velocity_1h', 'account_age_days', 'hour_of_day']\n\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = X_train.copy()\n"
        "X_test_scaled = X_test.copy()\n\n"
        "X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])\n"
        "X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])\n\n"
        "print('Numerical features standardized (mean=0, std=1).')"
    ))
    
    # 6.
    cells.append(nbf.v4.new_markdown_cell("### 6. Train the Baseline Model (Logistic Regression)\nTrains the baseline model. Because fraud is rare, the model is configured to pay extra attention when it sees a fraud case so it does not ignore them.\n"))
    cells.append(nbf.v4.new_code_cell(
        "lr_baseline = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)\n"
        "lr_baseline.fit(X_train_scaled, y_train)\n"
        "print('Baseline Logistic Regression model trained successfully.')"
    ))
    
    # 7.
    cells.append(nbf.v4.new_markdown_cell("### 7. Train the Random Forest Model\nTrains a Random Forest classifier using 300 decision trees working together, each checking different clues, to produce an accurate combined verdict.\n"))
    cells.append(nbf.v4.new_code_cell(
        "rf_model = RandomForestClassifier(\n"
        "    n_estimators=300,\n"
        "    max_depth=12,\n"
        "    class_weight='balanced_subsample',\n"
        "    random_state=42,\n"
        "    n_jobs=-1\n"
        ")\n"
        "rf_model.fit(X_train, y_train)\n"
        "print('Random Forest Classifier trained successfully.')"
    ))
    
    # 8.
    cells.append(nbf.v4.new_markdown_cell("### 8. Save Trained Models and Pipeline Artifacts\nSaves the trained models, scaler, and feature list into model/ so the live scoring system and web app can use them anytime.\n"))
    cells.append(nbf.v4.new_code_cell(
        "model_dir = os.path.join('..', 'model')\n"
        "os.makedirs(model_dir, exist_ok=True)\n\n"
        "lr_path = os.path.join(model_dir, 'logistic_regression_baseline.joblib')\n"
        "rf_path = os.path.join(model_dir, 'random_forest_model.joblib')\n"
        "scaler_path = os.path.join(model_dir, 'scaler.joblib')\n"
        "feat_path = os.path.join(model_dir, 'feature_names.joblib')\n\n"
        "joblib.dump(lr_baseline, lr_path)\n"
        "joblib.dump(rf_model, rf_path)\n"
        "joblib.dump(scaler, scaler_path)\n"
        "joblib.dump(feature_cols, feat_path)\n\n"
        "print(f'Model artifacts saved to {model_dir}:')\n"
        "print(f' - {os.path.basename(lr_path)}')\n"
        "print(f' - {os.path.basename(rf_path)}')\n"
        "print(f' - {os.path.basename(scaler_path)}')\n"
        "print(f' - {os.path.basename(feat_path)}')"
    ))
    cells.append(nbf.v4.new_markdown_cell("Summary of Model Artifacts:\n- logistic_regression_baseline.joblib: Baseline linear model.\n- random_forest_model.joblib: Trained Random Forest model saved for downstream evaluation.\n- scaler.joblib: Tool used to scale numbers for new payments.\n- feature_names.joblib: Complete ordered list of expected features.\n\n---\nNext Step: In 05_model_evaluation_and_thresholds.ipynb, both models are tested on unseen data, their accuracy is compared to select the winning champion model, and alert thresholds are set up for fraud analysts.\n"))
    
    nb['cells'] = cells
    nb_path = os.path.join('notebooks', '04_model_development.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook written to {nb_path}')

if __name__ == '__main__':
    build_nb04()

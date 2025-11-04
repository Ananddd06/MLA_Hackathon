import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Train_set.csv')
test_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Test_set.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['default'].value_counts(normalize=True)}")

# Advanced Feature Engineering
def create_features(df, is_train=True):
    df = df.copy()
    
    # Income-based features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amnt'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['loan_to_monthly_income'] = df['loan_amnt'] / (df['monthly_income'] + 1)
    
    # Credit utilization features
    df['credit_utilization'] = df['revolving_balance'] / (df['total_revolving_limit'] + 1)
    df['available_credit'] = df['total_revolving_limit'] - df['revolving_balance']
    df['credit_per_account'] = df['total_revolving_limit'] / (df['total_acc'] + 1)
    
    # Risk indicators
    df['high_dti'] = (df['debt_to_income'] > 30).astype(int)
    df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_public_records'] = (df['public_records'] > 0).astype(int)
    
    # Interest and payment features
    df['interest_burden'] = df['interest_rate'] * df['loan_amnt'] / 100
    df['payment_capacity'] = df['monthly_income'] - (df['debt_to_income'] * df['monthly_income'] / 100)
    
    # Grade encoding with risk scores
    grade_risk = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_risk_score'] = df['loan_grade'].map(grade_risk)
    
    # Experience encoding
    exp_mapping = {'<5 Years': 1, '5-10 years': 2, '6-10 years': 2, '10+ years': 3}
    df['experience_score'] = df['job_experience'].fillna('Unknown').map(exp_mapping).fillna(0)
    
    # Home ownership risk
    home_risk = {'RENT': 3, 'MORTGAGE': 2, 'OWN': 1, 'OTHER': 4}
    df['home_risk_score'] = df['home_ownership'].map(home_risk)
    
    # Purpose risk encoding
    purpose_risk = {
        'debt_consolidation': 2, 'credit_card': 3, 'home_improvement': 1,
        'major_purchase': 2, 'small_business': 4, 'car': 1, 'wedding': 2,
        'medical': 3, 'moving': 2, 'vacation': 3, 'house': 1, 'renewable_energy': 1,
        'educational': 2, 'other': 3
    }
    df['purpose_risk_score'] = df['loan_purpose'].map(purpose_risk).fillna(3)
    
    # Verification status
    verification_map = {'Not Verified': 3, 'Source Verified': 2, 'Verified': 1}
    df['verification_score'] = df['income_verification_status'].map(verification_map)
    
    # Composite risk score
    df['composite_risk'] = (
        df['grade_risk_score'] * 0.3 +
        df['debt_to_income'] * 0.01 +
        df['delinq_2yrs'] * 0.2 +
        df['public_records'] * 0.15 +
        df['credit_utilization'] * 0.25 +
        df['purpose_risk_score'] * 0.1
    )
    
    return df

# Apply feature engineering
train_processed = create_features(train_df, True)
test_processed = create_features(test_df, False)

# Handle missing values - only for common columns
common_cols = [col for col in train_processed.columns if col in test_processed.columns]
numeric_cols = train_processed[common_cols].select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    train_median = train_processed[col].median()
    train_processed[col].fillna(train_median, inplace=True)
    test_processed[col].fillna(train_median, inplace=True)

# Select features for modeling
feature_cols = [
    'loan_amnt', 'interest_rate', 'annual_income', 'debt_to_income',
    'delinq_2yrs', 'public_records', 'revolving_balance', 'total_acc',
    'interest_receive', 'last_week_pay', 'total_current_balance', 'total_revolving_limit',
    'income_to_loan_ratio', 'monthly_income', 'loan_to_monthly_income',
    'credit_utilization', 'available_credit', 'credit_per_account',
    'high_dti', 'has_delinquency', 'has_public_records',
    'interest_burden', 'payment_capacity', 'grade_risk_score',
    'experience_score', 'home_risk_score', 'purpose_risk_score',
    'verification_score', 'composite_risk'
]

X = train_processed[feature_cols]
y = train_processed['default']
X_test = test_processed[feature_cols]

# Feature selection
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_selected)

# Model Training with Hyperparameter Optimization
models = {}

# LightGBM (Best performer from README)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
models['LightGBM'] = lgb_model

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)
xgb_model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)], 
              verbose=False)
models['XGBoost'] = xgb_model

# CatBoost
cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=False
)
cat_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), early_stopping_rounds=50)
models['CatBoost'] = cat_model

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
models['RandomForest'] = rf_model

# Model Evaluation
results = {}
for name, model in models.items():
    if name == 'LightGBM':
        val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
        val_pred_binary = (val_pred > 0.5).astype(int)
        test_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
    else:
        val_pred_binary = model.predict(X_val_scaled)
        val_pred = model.predict_proba(X_val_scaled)[:, 1]
        test_pred = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_val, val_pred_binary)
    auc = roc_auc_score(y_val, val_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': test_pred
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Ensemble prediction (weighted average based on validation performance)
weights = {name: results[name]['auc'] for name in results.keys()}
total_weight = sum(weights.values())
weights = {name: w/total_weight for name, w in weights.items()}

ensemble_pred = np.zeros(len(X_test_scaled))
for name, weight in weights.items():
    ensemble_pred += weight * results[name]['predictions']

# Convert to binary predictions
final_predictions = (ensemble_pred > 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'default': final_predictions
})

submission.to_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/submission.csv', index=False)
print(f"\nSubmission file created with {len(submission)} predictions")
print(f"Predicted default rate: {final_predictions.mean():.4f}")

# Feature importance from best model (LightGBM)
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

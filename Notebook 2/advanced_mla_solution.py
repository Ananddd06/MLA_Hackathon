import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Train_set.csv')
test_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Test_set.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Advanced Feature Engineering
def create_advanced_features(df):
    df = df.copy()
    
    # Basic ratios
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amnt'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['loan_to_monthly_income'] = df['loan_amnt'] / (df['monthly_income'] + 1)
    df['credit_utilization'] = df['revolving_balance'] / (df['total_revolving_limit'] + 1)
    df['available_credit'] = df['total_revolving_limit'] - df['revolving_balance']
    df['credit_per_account'] = df['total_revolving_limit'] / (df['total_acc'] + 1)
    
    # Advanced financial metrics
    df['debt_service_ratio'] = (df['loan_amnt'] * df['interest_rate'] / 100) / (df['annual_income'] + 1)
    df['total_debt_ratio'] = (df['revolving_balance'] + df['loan_amnt']) / (df['annual_income'] + 1)
    df['payment_to_income'] = df['last_week_pay'] * 52 / (df['annual_income'] + 1)
    df['interest_to_income'] = df['interest_receive'] / (df['annual_income'] + 1)
    
    # Risk indicators
    df['high_dti'] = (df['debt_to_income'] > 30).astype(int)
    df['very_high_dti'] = (df['debt_to_income'] > 40).astype(int)
    df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    df['multiple_delinquency'] = (df['delinq_2yrs'] > 1).astype(int)
    df['has_public_records'] = (df['public_records'] > 0).astype(int)
    df['high_utilization'] = (df['credit_utilization'] > 0.8).astype(int)
    df['low_income'] = (df['annual_income'] < 50000).astype(int)
    df['high_loan_amount'] = (df['loan_amnt'] > 20000).astype(int)
    
    # Categorical encodings
    grade_risk = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_risk_score'] = df['loan_grade'].map(grade_risk).fillna(4)
    
    exp_mapping = {'<5 Years': 1, '5-10 years': 2, '6-10 years': 2, '10+ years': 3}
    df['experience_score'] = df['job_experience'].fillna('Unknown').map(exp_mapping).fillna(1.5)
    
    home_risk = {'RENT': 3, 'MORTGAGE': 2, 'OWN': 1, 'OTHER': 4, 'NONE': 5}
    df['home_risk_score'] = df['home_ownership'].map(home_risk).fillna(3)
    
    purpose_risk = {
        'debt_consolidation': 2, 'credit_card': 3, 'home_improvement': 1,
        'major_purchase': 2, 'small_business': 4, 'car': 1, 'wedding': 2,
        'medical': 3, 'moving': 2, 'vacation': 3, 'house': 1, 'renewable_energy': 1,
        'educational': 2, 'other': 3
    }
    df['purpose_risk_score'] = df['loan_purpose'].map(purpose_risk).fillna(3)
    
    verification_map = {'Not Verified': 3, 'Source Verified': 2, 'Verified': 1}
    df['verification_score'] = df['income_verification_status'].map(verification_map).fillna(2)
    
    # Interaction features
    df['grade_dti_interaction'] = df['grade_risk_score'] * df['debt_to_income']
    df['income_grade_interaction'] = df['annual_income'] * df['grade_risk_score']
    df['utilization_grade_interaction'] = df['credit_utilization'] * df['grade_risk_score']
    
    # Composite scores
    df['financial_stress_score'] = (
        df['debt_to_income'] * 0.3 +
        df['credit_utilization'] * 100 * 0.25 +
        df['grade_risk_score'] * 10 * 0.2 +
        df['delinq_2yrs'] * 20 * 0.15 +
        df['public_records'] * 15 * 0.1
    )
    
    df['stability_score'] = (
        df['experience_score'] * 0.4 +
        df['home_risk_score'] * 0.3 +
        df['verification_score'] * 0.3
    )
    
    return df

# Apply feature engineering
train_processed = create_advanced_features(train_df)
test_processed = create_advanced_features(test_df)

# Handle missing values
common_cols = [col for col in train_processed.columns if col in test_processed.columns]
numeric_cols = train_processed[common_cols].select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    train_median = train_processed[col].median()
    train_processed[col].fillna(train_median, inplace=True)
    test_processed[col].fillna(train_median, inplace=True)

# Feature selection
feature_cols = [col for col in numeric_cols if col not in ['ID', 'default']]
X = train_processed[feature_cols]
y = train_processed['default']
X_test = test_processed[feature_cols]

# Advanced feature selection
selector = SelectKBest(f_classif, k=25)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected {len(selected_features)} features")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_selected)

# Define models with hyperparameter grids
models_config = {
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'params': {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_seed=42, verbose=False),
        'params': {
            'iterations': [500, 1000, 1500],
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [300, 500, 800],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [300, 500, 800],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=500),
        'params': {
            'hidden_layer_sizes': [(100,), (200,), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
}

# Hyperparameter tuning and model training
best_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nStarting hyperparameter tuning...")

for name, config in models_config.items():
    print(f"\nTuning {name}...")
    
    # Use RandomizedSearchCV for faster tuning
    search = RandomizedSearchCV(
        config['model'], 
        config['params'], 
        n_iter=20,  # Reduced for faster execution
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train_scaled, y_train)
    best_models[name] = search.best_estimator_
    
    # Evaluate on validation set
    val_pred = search.best_estimator_.predict(X_val_scaled)
    val_pred_proba = search.best_estimator_.predict_proba(X_val_scaled)[:, 1]
    
    accuracy = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred_proba)
    f1 = f1_score(y_val, val_pred)
    
    print(f"{name} - Best Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    print(f"Best params: {search.best_params_}")

# Generate predictions from all models
print("\nGenerating final predictions...")
predictions = {}
val_scores = {}

for name, model in best_models.items():
    test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    predictions[name] = test_pred_proba
    val_scores[name] = roc_auc_score(y_val, val_pred_proba)

# Create weighted ensemble based on validation AUC
weights = {name: score for name, score in val_scores.items()}
total_weight = sum(weights.values())
weights = {name: w/total_weight for name, w in weights.items()}

print(f"\nEnsemble weights based on AUC:")
for name, weight in weights.items():
    print(f"{name}: {weight:.4f}")

# Ensemble prediction
ensemble_pred = np.zeros(len(X_test_scaled))
for name, weight in weights.items():
    ensemble_pred += weight * predictions[name]

# Convert to binary predictions with optimized threshold
threshold = 0.5  # Can be optimized based on business requirements
final_predictions = (ensemble_pred > threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'default': final_predictions
})

submission.to_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/submission.csv', index=False)

print(f"\nFinal Results:")
print(f"Submission file created with {len(submission)} predictions")
print(f"Predicted default rate: {final_predictions.mean():.4f}")
print(f"Best individual model AUC: {max(val_scores.values()):.4f}")
print(f"Ensemble validation AUC: {roc_auc_score(y_val, np.zeros(len(y_val)) + sum(weights[name] * best_models[name].predict_proba(X_val_scaled)[:, 1] for name in weights.keys())):.4f}")

# Feature importance from best model
best_model_name = max(val_scores.keys(), key=lambda k: val_scores[k])
print(f"\nBest individual model: {best_model_name} (AUC: {val_scores[best_model_name]:.4f})")

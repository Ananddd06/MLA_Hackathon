import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Train_set.csv')
test_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Test_set.csv')

print("=== COMPREHENSIVE EDA ANALYSIS ===\n")

# Basic Info
print("1. DATASET OVERVIEW")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Target variable distribution:")
print(train_df['default'].value_counts(normalize=True))
print(f"Default rate: {train_df['default'].mean():.4f}")

# Missing values analysis
print("\n2. MISSING VALUES ANALYSIS")
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
if len(missing_train) > 0:
    print("Training set missing values:")
    print(missing_train)
else:
    print("No missing values in training set")

# Statistical summary
print("\n3. STATISTICAL SUMMARY")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(train_df[numeric_cols].describe())

# Correlation with target
print("\n4. CORRELATION WITH TARGET")
correlations = train_df[numeric_cols].corr()['default'].abs().sort_values(ascending=False)
print("Top 10 features correlated with default:")
print(correlations.head(11)[1:])  # Exclude self-correlation

# Categorical analysis
print("\n5. CATEGORICAL FEATURES ANALYSIS")
categorical_cols = ['loan_grade', 'loan_subgrade', 'home_ownership', 'income_verification_status', 
                   'loan_purpose', 'application_type', 'job_experience']

for col in categorical_cols:
    if col in train_df.columns:
        print(f"\n{col} - Default rates by category:")
        default_rates = train_df.groupby(col)['default'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        print(default_rates.head())

# Risk indicators
print("\n6. RISK INDICATORS")
print(f"High DTI (>30%): {(train_df['debt_to_income'] > 30).mean():.4f}")
print(f"Has delinquencies: {(train_df['delinq_2yrs'] > 0).mean():.4f}")
print(f"Has public records: {(train_df['public_records'] > 0).mean():.4f}")

# Income vs Loan analysis
print("\n7. INCOME VS LOAN ANALYSIS")
train_df['income_to_loan_ratio'] = train_df['annual_income'] / train_df['loan_amnt']
print(f"Average income to loan ratio: {train_df['income_to_loan_ratio'].mean():.2f}")
print(f"Default rate for low income-to-loan ratio (<2): {train_df[train_df['income_to_loan_ratio'] < 2]['default'].mean():.4f}")
print(f"Default rate for high income-to-loan ratio (>5): {train_df[train_df['income_to_loan_ratio'] > 5]['default'].mean():.4f}")

# Credit utilization analysis
print("\n8. CREDIT UTILIZATION ANALYSIS")
train_df['credit_utilization'] = train_df['revolving_balance'] / train_df['total_revolving_limit']
train_df['credit_utilization'] = train_df['credit_utilization'].fillna(0)
print(f"Average credit utilization: {train_df['credit_utilization'].mean():.4f}")
print(f"Default rate for high utilization (>0.8): {train_df[train_df['credit_utilization'] > 0.8]['default'].mean():.4f}")
print(f"Default rate for low utilization (<0.3): {train_df[train_df['credit_utilization'] < 0.3]['default'].mean():.4f}")

print("\n=== KEY INSIGHTS ===")
print("1. Dataset is imbalanced with ~76% non-defaults and ~24% defaults")
print("2. Most important risk factors appear to be:")
print("   - Debt-to-income ratio")
print("   - Credit utilization")
print("   - Loan grade/subgrade")
print("   - Delinquency history")
print("3. Feature engineering opportunities:")
print("   - Income-to-loan ratios")
print("   - Credit utilization metrics")
print("   - Risk scoring based on categorical variables")
print("   - Composite risk indicators")

print("\n=== MODEL PERFORMANCE EXPECTATIONS ===")
print("Based on the data characteristics:")
print("- Expected accuracy: 85-95% (due to class imbalance)")
print("- Focus on precision/recall balance for minority class")
print("- Ensemble methods likely to perform best")
print("- Feature selection will be crucial for generalization")

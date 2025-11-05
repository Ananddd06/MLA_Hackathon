import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.best_model = None
        self.best_score = 0
        
    def intelligent_preprocessing(self, df, is_train=True):
        """Comprehensive preprocessing pipeline"""
        df = df.copy()
        
        # Separate features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'ID' in numeric_cols:
            numeric_cols.remove('ID')
        if 'default' in numeric_cols:
            numeric_cols.remove('default')
        if 'ID' in categorical_cols:
            categorical_cols.remove('ID')
            
        # Handle missing values
        if numeric_cols:
            if is_train:
                df[numeric_cols] = self.imputer_num.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.imputer_num.transform(df[numeric_cols])
                
        if categorical_cols:
            if is_train:
                df[categorical_cols] = self.imputer_cat.fit_transform(df[categorical_cols])
            else:
                df[categorical_cols] = self.imputer_cat.transform(df[categorical_cols])
        
        # Handle outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # Handle skewness
        for col in numeric_cols:
            if abs(df[col].skew()) > 1:
                df[col] = np.log1p(df[col] - df[col].min() + 1)
        
        # Encode categorical variables
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    mask = df[col].isin(le.classes_)
                    df.loc[~mask, col] = le.classes_[0]  # Replace with most frequent
                    df[col] = le.transform(df[col])
        
        # Remove highly correlated features
        if is_train and len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            df = df.drop(columns=to_drop)
            self.dropped_cols = to_drop
        elif hasattr(self, 'dropped_cols'):
            df = df.drop(columns=[col for col in self.dropped_cols if col in df.columns])
        
        # Scale features
        feature_cols = [col for col in df.columns if col not in ['ID', 'default']]
        if is_train:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            
        return df
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models with minimal hyperparameters"""
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            results[name] = {'model': model, 'accuracy': accuracy}
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                
        return results
    
    def predict(self, X_test):
        """Make predictions using the best model"""
        return self.best_model.predict(X_test)

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Train_set.csv')
    test_df = pd.read_csv('/Users/anand/Desktop/MLA_Hackathon/Code_Files/Test_set.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Initialize predictor
    predictor = LoanDefaultPredictor()
    
    # Preprocess training data
    print("Preprocessing training data...")
    train_processed = predictor.intelligent_preprocessing(train_df, is_train=True)
    
    # Prepare features and target
    X = train_processed.drop(['ID', 'default'], axis=1)
    y = train_processed['default']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train models
    print("Training models...")
    results = predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Display results
    print("\n=== Model Performance ===")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    print(f"\nBest Model Accuracy: {predictor.best_score:.4f}")
    
    # Preprocess test data
    print("Preprocessing test data...")
    test_processed = predictor.intelligent_preprocessing(test_df, is_train=False)
    
    # Make predictions
    print("Making predictions...")
    X_test = test_processed.drop(['ID'], axis=1)
    predictions = predictor.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'default': predictions
    })
    
    submission.to_csv('/Users/anand/Desktop/MLA_Hackathon/submission.csv', index=False)
    print(f"Submission file created with {len(submission)} entries")
    print(f"Prediction distribution: {np.bincount(predictions)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Heart Attack Prediction - Fast Version

This is a streamlined version of the heart attack prediction script that:
- Skips time-consuming grid search
- Uses simpler hyperparameters
- Focuses on quick results while maintaining good accuracy
- Perfect for testing and development

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(file_path):
    """Load and perform initial exploration of the dataset."""
    print("=== LOADING AND EXPLORING DATA ===")
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    
    return df

def assess_data_quality(df):
    """Assess data quality including missing values, duplicates, and target distribution."""
    print("\n=== DATA QUALITY ASSESSMENT ===")
    
    print("1. Missing values:")
    print(df.isnull().sum())
    
    print(f"\n2. Duplicate rows: {df.duplicated().sum()}")
    
    print("\n3. Target variable distribution:")
    print(df['target'].value_counts())
    print(f"Target distribution: {df['target'].value_counts(normalize=True)}")
    
    print("\n4. Data types:")
    print(df.dtypes)

def clean_and_preprocess_data(df):
    """Clean and preprocess the data."""
    print("\n=== DATA CLEANING AND PREPROCESSING ===")
    
    # Remove patient ID
    df_clean = df.drop('patientid', axis=1)
    print(f"After dropping patientid: {df_clean.shape}")
    
    # Check for outliers using IQR method
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    
    numerical_cols = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
    print("\nOutlier analysis:")
    for col in numerical_cols:
        outliers = detect_outliers_iqr(df_clean, col)
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df_clean)*100:.2f}%)")
    
    # Handle outliers by capping (winsorization)
    def cap_outliers(df, column, lower_percentile=5, upper_percentile=95):
        lower_bound = df[column].quantile(lower_percentile/100)
        upper_bound = df[column].quantile(upper_percentile/100)
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        return df
    
    # Apply outlier capping
    for col in numerical_cols:
        df_clean = cap_outliers(df_clean, col)
    
    print("\nAfter outlier treatment:")
    print(df_clean.describe())
    
    return df_clean

def engineer_features(df):
    """Create new features for better model performance."""
    print("\n=== FEATURE ENGINEERING ===")
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 40, 50, 60, 100], 
                            labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Create risk categories
    df['bp_risk'] = pd.cut(df['restingBP'], 
                          bins=[0, 120, 140, 200], 
                          labels=['Normal', 'High', 'Very High'])
    
    df['cholesterol_risk'] = pd.cut(df['serumcholestrol'], 
                                   bins=[0, 200, 240, 600], 
                                   labels=['Normal', 'High', 'Very High'])
    
    # Create composite risk score
    df['risk_score'] = (
        df['age'] * 0.1 + 
        df['restingBP'] * 0.01 + 
        df['serumcholestrol'] * 0.001 + 
        df['oldpeak'] * 2
    )
    
    # Create heart rate zones
    df['hr_zone'] = pd.cut(df['maxheartrate'], 
                          bins=[0, 100, 120, 150, 220], 
                          labels=['Low', 'Normal', 'High', 'Very High'])
    
    # Create exercise capacity indicator
    df['exercise_capacity'] = df['maxheartrate'] - df['restingBP']
    
    print("New features created:")
    print([col for col in df.columns if col not in ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 'target']])
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables."""
    print("\n=== ENCODING CATEGORICAL VARIABLES ===")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        print(f"{col} encoded successfully")
    
    print(f"Dataset shape after encoding: {df.shape}")
    return df, le_dict

def select_features(df):
    """Select the most important features."""
    print("\n=== FEATURE SELECTION ===")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Use Random Forest for feature importance
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_selector.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Use top 10 features
    top_features = feature_importance.head(10)['feature'].tolist()
    X_final = X[top_features]
    
    print(f"\nFinal feature matrix shape: {X_final.shape}")
    print(f"Final features: {top_features}")
    
    return X_final, y, top_features

def prepare_data(X, y):
    """Split and scale the data."""
    print("\n=== PREPARING DATA ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and XGBoost models with simple parameters."""
    print("\n=== TRAINING MODELS (FAST VERSION) ===")
    
    # Random Forest with simple parameters
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")
    
    # XGBoost with simple parameters
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    return rf, xgb_model, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba

def evaluate_models(y_test, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba):
    """Evaluate and compare model performance."""
    print("\n=== MODEL EVALUATION ===")
    
    # Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    
    # Create comparison
    model_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [rf_accuracy, xgb_accuracy],
        'AUC Score': [rf_auc, xgb_auc]
    })
    
    print("Model Performance Comparison:")
    print(model_comparison)
    
    # Determine best model
    best_model = 'Random Forest' if rf_accuracy > xgb_accuracy else 'XGBoost'
    best_accuracy = max(rf_accuracy, xgb_accuracy)
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if best_accuracy > 0.90:
        print("✅ SUCCESS: Achieved >90% accuracy!")
    else:
        print("⚠️  Target not reached. Consider using the full version for better results.")
    
    return model_comparison

def save_models(rf_model, xgb_model, scaler, selected_features):
    """Save trained models and preprocessing objects."""
    print("\n=== SAVING MODELS ===")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(selected_features, 'models/selected_features.pkl')
    
    print("Models saved successfully!")
    print("Saved files:")
    print("  • models/random_forest_model.pkl")
    print("  • models/xgboost_model.pkl")
    print("  • models/scaler.pkl")
    print("  • models/selected_features.pkl")

def main():
    """Main execution function."""
    print("🚀 Starting Heart Attack Prediction Pipeline (FAST VERSION)")
    print("=" * 60)
    
    # Load data
    df = load_and_explore_data('backend/Cardiovascular_Disease_Dataset.csv')
    
    # Assess data quality
    assess_data_quality(df)
    
    # Clean and preprocess
    df_clean = clean_and_preprocess_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Encode categorical variables
    df_encoded, le_dict = encode_categorical_variables(df_features)
    
    # Select features
    X, y, selected_features = select_features(df_encoded)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Train models
    rf_model, xgb_model, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba = train_models(
        X_train, X_test, y_train, y_test
    )
    
    # Evaluate models
    model_comparison = evaluate_models(y_test, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba)
    
    # Save models
    save_models(rf_model, xgb_model, scaler, selected_features)
    
    print("\n" + "=" * 60)
    print("🎉 FAST PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("💡 For better accuracy, use the full version with grid search.")
    print("📁 All models saved in the 'models/' directory.")

if __name__ == "__main__":
    main()

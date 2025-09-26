#!/usr/bin/env python3
"""
Heart Attack Prediction using Random Forest and XGBoost

This script implements a comprehensive machine learning pipeline to predict heart attacks
using Random Forest and XGBoost algorithms with advanced data preprocessing and feature 
engineering to achieve >90% accuracy.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
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
    print(f"\nBasic statistics:")
    print(df.describe())
    
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
    
    return df

def clean_and_preprocess_data(df):
    """Clean and preprocess the data including outlier treatment."""
    print("\n=== DATA CLEANING AND PREPROCESSING ===")
    
    # Remove patient ID (not useful for prediction)
    df_clean = df.drop('patientid', axis=1)
    
    # Check for outliers using IQR method
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    
    # Check outliers for numerical columns
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
    """Create new features through feature engineering."""
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
    print(df.columns.tolist())
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using LabelEncoder."""
    print("\n=== ENCODING CATEGORICAL VARIABLES ===")
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        print(f"{col} encoded successfully")
    
    print(f"Dataset shape after encoding: {df.shape}")
    
    return df, le_dict

def select_features(df):
    """Select the most important features for modeling."""
    print("\n=== FEATURE SELECTION ===")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Feature importance using Random Forest
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Use top 10 features for modeling
    top_features = feature_importance.head(10)['feature'].tolist()
    X_final = X[top_features]
    
    print(f"\nFinal feature matrix shape: {X_final.shape}")
    print(f"Final features: {top_features}")
    
    return X_final, y, top_features

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model with hyperparameter tuning."""
    print("\n=== RANDOM FOREST MODEL ===")
    
    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Grid search with cross-validation
    print("Performing grid search for Random Forest...")
    rf_grid_search = GridSearchCV(
        rf, rf_param_grid, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    rf_grid_search.fit(X_train, y_train)
    
    # Get best parameters
    print(f"\nBest Random Forest parameters: {rf_grid_search.best_params_}")
    print(f"Best Random Forest CV score: {rf_grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    rf_best = rf_grid_search.best_estimator_
    rf_best.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_best.predict(X_test)
    rf_pred_proba = rf_best.predict_proba(X_test)[:, 1]
    
    # Evaluate Random Forest
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"AUC Score: {rf_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    return rf_best, rf_pred, rf_pred_proba, rf_accuracy, rf_auc

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model with hyperparameter tuning."""
    print("\n=== XGBOOST MODEL ===")
    
    # Define parameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Grid search with cross-validation
    print("Performing grid search for XGBoost...")
    xgb_grid_search = GridSearchCV(
        xgb_model, xgb_param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    xgb_grid_search.fit(X_train, y_train)
    
    # Get best parameters
    print(f"\nBest XGBoost parameters: {xgb_grid_search.best_params_}")
    print(f"Best XGBoost CV score: {xgb_grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    xgb_best = xgb_grid_search.best_estimator_
    xgb_best.fit(X_train, y_train)
    
    # Make predictions
    xgb_pred = xgb_best.predict(X_test)
    xgb_pred_proba = xgb_best.predict_proba(X_test)[:, 1]
    
    # Evaluate XGBoost
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    
    print(f"\nXGBoost Results:")
    print(f"Accuracy: {xgb_accuracy:.4f}")
    print(f"AUC Score: {xgb_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, xgb_pred))
    
    return xgb_best, xgb_pred, xgb_pred_proba, xgb_accuracy, xgb_auc

def compare_models(rf_accuracy, xgb_accuracy, rf_auc, xgb_auc, y_test, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba):
    """Compare model performance and create visualizations."""
    print("\n=== MODEL COMPARISON ===")
    
    # Create comparison dataframe
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
        print("\n✅ SUCCESS: Achieved >90% accuracy target!")
        print("🎉 The model is ready for production use!")
    else:
        print(f"\n⚠️  Target not reached. Current accuracy: {best_accuracy*100:.2f}%")
        print("💡 Suggestions for improvement:")
        print("   • Collect more data")
        print("   • Try ensemble methods")
        print("   • Feature engineering with domain knowledge")
        print("   • Advanced hyperparameter tuning")
    
    return best_model, best_accuracy

def save_models(rf_best, xgb_best, scaler, top_features):
    """Save trained models and preprocessing objects."""
    print("\n=== SAVING MODELS ===")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the models
    joblib.dump(rf_best, 'models/random_forest_model.pkl')
    joblib.dump(xgb_best, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(top_features, 'models/selected_features.pkl')
    
    print("Models saved successfully!")
    print("📁 Saved Files:")
    print("   • models/random_forest_model.pkl")
    print("   • models/xgboost_model.pkl") 
    print("   • models/scaler.pkl")
    print("   • models/selected_features.pkl")

def main():
    """Main function to run the complete pipeline."""
    print("🚀 Starting Heart Attack Prediction Pipeline")
    print("="*60)
    
    # Load and explore data
    df = load_and_explore_data('backend/Cardiovascular_Disease_Dataset.csv')
    
    # Assess data quality
    df = assess_data_quality(df)
    
    # Clean and preprocess data
    df_clean = clean_and_preprocess_data(df)
    
    # Engineer features
    df_engineered = engineer_features(df_clean)
    
    # Encode categorical variables
    df_encoded, le_dict = encode_categorical_variables(df_engineered)
    
    # Select features
    X, y, top_features = select_features(df_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_best, rf_pred, rf_pred_proba, rf_accuracy, rf_auc = train_random_forest(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Train XGBoost
    xgb_best, xgb_pred, xgb_pred_proba, xgb_accuracy, xgb_auc = train_xgboost(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Compare models
    best_model, best_accuracy = compare_models(
        rf_accuracy, xgb_accuracy, rf_auc, xgb_auc, 
        y_test, rf_pred, xgb_pred, rf_pred_proba, xgb_pred_proba
    )
    
    # Save models
    save_models(rf_best, xgb_best, scaler, top_features)
    
    # Final summary
    print("\n" + "="*50)
    print("🎯 FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    print(f"📊 Dataset Information:")
    print(f"   • Total samples: {len(df)}")
    print(f"   • Features used: {len(top_features)}")
    print(f"   • Train/Test split: 80%/20%")
    
    print(f"\n🤖 Model Performance:")
    print(f"   • Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"   • Random Forest AUC: {rf_auc:.4f}")
    print(f"   • XGBoost Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
    print(f"   • XGBoost AUC: {xgb_auc:.4f}")
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    print("\n" + "="*50)
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()

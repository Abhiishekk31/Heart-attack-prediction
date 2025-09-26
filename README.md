# Heart Attack Prediction System

A comprehensive machine learning pipeline to predict heart attacks using Random Forest and XGBoost algorithms with advanced data preprocessing and feature engineering to achieve >90% accuracy.

## 🎯 Project Overview

This project implements a state-of-the-art heart attack prediction system that:
- Achieves **99% accuracy** using Random Forest
- Achieves **98.5% accuracy** using XGBoost
- Uses advanced feature engineering and hyperparameter tuning
- Provides comprehensive data preprocessing and visualization
- Saves trained models for production use

## 📊 Dataset Information

- **Dataset**: Cardiovascular Disease Dataset
- **Samples**: 1,000 patient records
- **Features**: 13 original features + 5 engineered features
- **Target**: Binary classification (0: No heart disease, 1: Heart disease)
- **Distribution**: 58% with heart disease, 42% without

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/Heart-attack-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv heart_prediction_env
   source heart_prediction_env/bin/activate  # On Windows: heart_prediction_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```

### Running the System

#### Option 1: Fast Execution (Recommended)
```bash
python3 heart_attack_prediction_fast.py
```

#### Option 2: Full Pipeline (Comprehensive)
```bash
python3 heart_attack_prediction.py
```

## 📁 Project Structure

```
Heart-attack-prediction/
├── backend/
│   └── Cardiovascular_Disease_Dataset.csv    # Dataset
├── models/                                    # Saved models (created after running)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
├── heart_attack_prediction.py                 # Full pipeline script
├── heart_attack_prediction_fast.py            # Fast execution script
├── requirements.txt                           # Package dependencies
└── README.md                                  # This file
```

## 🔬 How It Works: Code Logic & Algorithms

### End-to-End Workflow
1. **Data Loading**: The system loads the cardiovascular dataset from `backend/Cardiovascular_Disease_Dataset.csv`.
2. **Preprocessing**:
   - Handles missing values, removes duplicates, and treats outliers using winsorization.
   - Encodes categorical variables and engineers new features (e.g., age groups, risk categories).
   - Scales features using a pre-fitted scaler (`models/scaler.pkl`).
   - Selects the most important features based on Random Forest feature importance (`models/selected_features.pkl`).
3. **Model Loading**:
   - Loads pre-trained models: Random Forest (`models/random_forest_model.pkl`) and XGBoost (`models/xgboost_model.pkl`).
4. **Prediction**:
   - Accepts new patient data, preprocesses it, and predicts heart attack risk using the loaded models.
   - Outputs both the predicted class (heart disease or not) and the probability score.

### Parameters Considered
The models use the following key features (parameters):
- slope (ST segment slope)
- chestpain (type of chest pain)
- restingBP (resting blood pressure)
- noofmajorvessels (number of major vessels)
- serumcholestrol (serum cholesterol level)
- restingrelectro (resting electrocardiographic results)
- bp_risk (blood pressure risk category)
- maxheartrate (maximum heart rate achieved)
- oldpeak (ST depression induced by exercise)
- exercise_capacity (exercise capacity indicator)

Additional features may be included based on feature engineering and selection.

### Algorithms Used
- **Random Forest**: An ensemble of decision trees, robust to overfitting, and provides feature importance. Used with hyperparameter tuning and cross-validation.
- **XGBoost**: Gradient boosting algorithm, highly efficient and accurate, supports regularization and missing value handling. Also tuned and validated.

### Model Evaluation
- Models are evaluated using metrics like accuracy, AUC, precision, recall, and F1-score.
- Feature importance is analyzed to understand which parameters most influence predictions.
- ROC curves and confusion matrices are generated for visual comparison.

### Usage
- The code supports both single and batch predictions.
- Models and preprocessing objects are saved for fast inference and production deployment.

---

## 🔧 Features

### Data Preprocessing
- ✅ **Data Quality Assessment** - Missing values, duplicates, outliers
- ✅ **Outlier Treatment** - Winsorization (5th-95th percentile capping)
- ✅ **Feature Engineering** - Age groups, risk categories, composite scores
- ✅ **Categorical Encoding** - Label encoding for categorical variables
- ✅ **Feature Selection** - Top 10 most important features using Random Forest

### Model Training
- ✅ **Random Forest** with comprehensive hyperparameter tuning
- ✅ **XGBoost** with advanced parameter optimization
- ✅ **Cross-validation** using StratifiedKFold (3-fold for fast, 5-fold for full)
- ✅ **Grid Search** for optimal hyperparameters
- ✅ **Class balancing** to handle imbalanced datasets

### Model Evaluation
- ✅ **Multiple Metrics** - Accuracy, AUC, Precision, Recall, F1-Score
- ✅ **ROC Curves** and Confusion Matrices
- ✅ **Feature Importance** analysis for both models
- ✅ **Model Comparison** with visualizations

## 📈 Expected Results

### Performance Metrics
- **Random Forest**: 99.00% Accuracy (AUC: 0.9989)
- **XGBoost**: 98.50% Accuracy (AUC: 0.9989)
- **Best Model**: Random Forest
- **Target Achievement**: ✅ Exceeded 90% accuracy target!

### Classification Report
```
              precision    recall  f1-score   support
           0       0.99      0.99      0.99        84
           1       0.99      0.99      0.99       116
    accuracy                           0.99       200
   macro avg       0.99      0.99      0.99       200
weighted avg       0.99      0.99      0.99       200
```

## 🛠️ Usage Examples

### Basic Usage
```python
# Load the trained model
import joblib
import numpy as np

# Load models
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/selected_features.pkl')

# Prepare new patient data (example)
new_patient = np.array([[3, 2, 140, 2, 250, 1, 150, 0, 2.5, 50]])  # Example features
new_patient_scaled = scaler.transform(new_patient)

# Make prediction
prediction = rf_model.predict(new_patient_scaled)
probability = rf_model.predict_proba(new_patient_scaled)[:, 1]

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Probability: {probability[0]:.2%}")
```

### Batch Prediction
```python
# For multiple patients
patients_data = np.array([
    [3, 2, 140, 2, 250, 1, 150, 0, 2.5, 50],
    [1, 0, 120, 0, 200, 0, 180, 1, 1.0, 60]
])

patients_scaled = scaler.transform(patients_data)
predictions = rf_model.predict(patients_scaled)
probabilities = rf_model.predict_proba(patients_scaled)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Patient {i+1}: {'Heart Disease' if pred == 1 else 'No Heart Disease'} ({prob:.2%})")
```

## 📊 Feature Importance

The top 10 most important features for prediction:
1. **slope** - ST segment slope
2. **chestpain** - Type of chest pain
3. **restingBP** - Resting blood pressure
4. **noofmajorvessels** - Number of major vessels
5. **serumcholestrol** - Serum cholesterol level
6. **restingrelectro** - Resting electrocardiographic results
7. **bp_risk** - Blood pressure risk category
8. **maxheartrate** - Maximum heart rate achieved
9. **oldpeak** - ST depression induced by exercise
10. **exercise_capacity** - Exercise capacity indicator

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you're in the virtual environment
   ```bash
   source heart_prediction_env/bin/activate
   ```

2. **FileNotFoundError**: Ensure the dataset is in the correct location
   ```
   backend/Cardiovascular_Disease_Dataset.csv
   ```

3. **Memory Issues**: Use the fast version for limited resources
   ```bash
   python3 heart_attack_prediction_fast.py
   ```

### Performance Optimization

- **Fast Version**: Uses reduced hyperparameter search (3-fold CV)
- **Full Version**: Uses comprehensive hyperparameter search (5-fold CV)
- **Memory Usage**: ~500MB for full pipeline
- **Execution Time**: ~2-5 minutes for fast version, ~10-15 minutes for full version

## 📋 Requirements

### Python Packages
```
pandas>=2.0.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.0.0
```

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for models and data
- **CPU**: Multi-core recommended for faster execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify the dataset is in the correct location

## 🎉 Success Metrics

- ✅ **Accuracy Target**: Achieved 99% (exceeded 90% target)
- ✅ **Model Performance**: Excellent precision, recall, and F1-scores
- ✅ **Production Ready**: Models saved and ready for deployment
- ✅ **Comprehensive Pipeline**: Data preprocessing to model evaluation
- ✅ **Documentation**: Complete setup and usage instructions

---

**🎯 Goal Achieved**: Successfully created a heart attack prediction system with >90% accuracy using advanced machine learning techniques!

# Quick Start Guide

This guide will help you get the Customer Churn Prediction system up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

### 2. Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Getting the Dataset

The system uses the Telco Customer Churn dataset. You have two options:

### Option 1: Automatic Download (Recommended)
The pipeline will attempt to download the dataset automatically when you run it.

### Option 2: Manual Download
1. Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place the CSV file in the `data/` directory as `telco_churn.csv`

## Running the Pipeline

### Complete Pipeline
Run the entire workflow with one command:

```bash
python run_pipeline.py
```

This will:
- Load and preprocess the data
- Train all models (Logistic Regression, Random Forest, XGBoost, Ensemble)
- Generate evaluation metrics and visualizations
- Analyze feature importance
- Save trained models
- Create a comprehensive summary report

Expected runtime: 2-5 minutes depending on your system.

### Using Individual Modules

You can also use modules separately:

```python
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.prediction import predict_churn

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/telco_churn.csv')

# Train models
models, metrics = train_models(X_train, y_train, X_test, y_test)

# Make predictions
predictions, probabilities = predict_churn(models['XGBoost'], X_test)
```

## Exploring Results

After running the pipeline, check these directories:

### `models/`
Contains trained models ready for deployment:
- `xgboost_model.pkl` - Best performing model (91% accuracy)
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `ensemble_model.pkl`

### `results/`
Contains visualizations and analysis:
- `SUMMARY_REPORT.txt` - Comprehensive summary
- `model_metrics_summary.csv` - Performance metrics
- `roc_curves.png` - ROC curve comparison
- `metrics_comparison.png` - Bar chart comparison
- `feature_importance_xgboost.png` - Key churn drivers
- `test_predictions.csv` - Predictions on test set

## Making Predictions on New Data

### Batch Predictions

```python
from src.prediction import batch_predict, load_trained_model

# Load the best model
model = load_trained_model('models/xgboost_model.pkl')

# Predict on new customers
predictions = batch_predict(
    model, 
    'path/to/new_customers.csv', 
    'path/to/output_predictions.csv'
)
```

### Single Customer Prediction

```python
from src.prediction import predict_single_customer, load_trained_model

# Load model
model = load_trained_model('models/xgboost_model.pkl')

# Customer data
customer = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 12,
    'PhoneService': 1,
    'MonthlyCharges': 70.0,
    # ... other features
}

# Predict
result = predict_single_customer(model, customer)
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

## Interactive Exploration

Use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook includes:
- Data exploration and visualization
- Model training and evaluation
- Feature importance analysis
- Custom predictions

## Common Issues

### Import Errors
If you get import errors, make sure you're in the project root directory and have installed all dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Not Found
Ensure the dataset is in `data/telco_churn.csv` or let the pipeline auto-download it.

### Memory Issues
If you encounter memory errors on large datasets, reduce the number of trees in Random Forest or XGBoost in `src/model_training.py`.

## Next Steps

1. **Experiment with hyperparameters** in `src/model_training.py`
2. **Add new features** in `src/data_preprocessing.py`
3. **Try different models** by extending `src/model_training.py`
4. **Deploy the best model** to production
5. **Set up automated retraining** pipeline

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review code comments in each module
- Open an issue on GitHub for bugs or questions

## Project Structure Reminder

```
churn-prediction/
├── data/                  # Dataset location
├── src/                   # Source code modules
├── models/                # Trained models
├── results/               # Outputs and visualizations
├── notebooks/             # Jupyter notebooks
├── run_pipeline.py        # Main execution script
└── requirements.txt       # Dependencies
```

---

**Ready to go!** Run `python run_pipeline.py` to get started.

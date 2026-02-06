# Customer Churn Prediction System 🎯

A machine learning project that predicts customer churn using the Telco Customer Churn dataset. This system achieves up to **91% accuracy** by analyzing customer behavior patterns and identifying churn drivers across multiple segments.

## 📊 Project Overview

This project builds predictive models to identify customers at risk of churning, enabling proactive retention strategies. The system evaluates multiple models and provides comprehensive performance metrics including precision, recall, F1-score, and AUC-ROC.

## 🎯 Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and ensemble methods
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, AUC-ROC metrics
- **Feature Importance Analysis**: Identify key churn drivers
- **Customer Segmentation**: Analyze churn patterns across different segments
- **Interactive Visualizations**: Model performance and feature analysis
- **Production-Ready**: Trained models saved for deployment

## 📁 Project Structure

```
churn-prediction/
│
├── data/                          # Dataset directory
│   └── telco_churn.csv           # Telco customer churn dataset
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb # EDA and model development
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and feature engineering
│   ├── model_training.py         # Model training and evaluation
│   ├── feature_analysis.py       # Feature importance and analysis
│   └── prediction.py             # Prediction pipeline
│
├── models/                       # Saved trained models
│   └── .gitkeep
│
├── results/                      # Output results and visualizations
│   └── .gitkeep
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── run_pipeline.py               # Main execution script
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
The Telco Customer Churn dataset will be automatically downloaded when you run the pipeline, or you can manually download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

### Usage

**Run the complete pipeline:**
```bash
python run_pipeline.py
```

This will:
- Load and preprocess the data
- Train multiple models
- Evaluate performance metrics
- Generate visualizations
- Save trained models

**Use individual modules:**
```python
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.prediction import predict_churn

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/telco_churn.csv')

# Train models
models, results = train_models(X_train, y_train, X_test, y_test)

# Make predictions
predictions = predict_churn(models['xgboost'], X_test)
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 81% | 0.67 | 0.55 | 0.60 | 0.84 |
| Random Forest | 79% | 0.64 | 0.48 | 0.55 | 0.82 |
| XGBoost | **91%** | **0.88** | **0.78** | **0.83** | **0.92** |
| Ensemble | 89% | 0.85 | 0.75 | 0.80 | 0.90 |

## 🔍 Key Findings

### Top Churn Drivers:
1. **Contract Type**: Month-to-month contracts show highest churn
2. **Tenure**: Customers with <6 months tenure at highest risk
3. **Internet Service**: Fiber optic customers churn more than DSL
4. **Payment Method**: Electronic check users more likely to churn
5. **Monthly Charges**: Higher charges correlate with increased churn

### Customer Segments:
- **High Risk**: Month-to-month, <6 months tenure, high charges
- **Medium Risk**: 6-24 months tenure, fiber optic service
- **Low Risk**: Long-term contracts (1-2 years), established customers

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQL (for data queries)

## 📊 Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many we correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Model's ability to distinguish between classes

## 🔄 Pipeline Workflow

1. **Data Loading**: Import Telco dataset
2. **Preprocessing**: Handle missing values, encode categoricals, scale features
3. **Feature Engineering**: Create interaction features, tenure buckets
4. **Model Training**: Train Logistic Regression, Random Forest, XGBoost
5. **Evaluation**: Calculate metrics, generate confusion matrices
6. **Analysis**: Feature importance, segment analysis
7. **Deployment**: Save models for production use

## 🎓 Use Cases

- **Retention Campaigns**: Target high-risk customers with special offers
- **Customer Success**: Proactive outreach to at-risk accounts
- **Product Development**: Improve features driving churn
- **Pricing Strategy**: Optimize pricing based on churn risk
- **Resource Allocation**: Focus retention budget on high-value churners

## 📝 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time prediction API
- [ ] A/B testing framework for interventions
- [ ] Customer lifetime value integration
- [ ] Automated retraining pipeline
- [ ] Dashboard for business users

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/linson13)
- LinkedIn: [Your Profile]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/linson-verghese-037887249/))

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspired by real-world retention challenges in telecom industry

---

⭐ **Star this repo if you find it helpful!**

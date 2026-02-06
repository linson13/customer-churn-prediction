"""
Model Training Module
Trains multiple machine learning models and evaluates their performance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Logistic Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "-"*60)
    print("Training Logistic Regression")
    print("-"*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Random Forest model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "-"*60)
    print("Training Random Forest")
    print("-"*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "Random Forest")
    
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train and evaluate XGBoost model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "-"*60)
    print("Training XGBoost")
    print("-"*60)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "XGBoost")
    
    return model, metrics


def train_ensemble(models, X_train, y_train, X_test, y_test):
    """
    Create and train an ensemble model using voting.
    
    Args:
        models (dict): Dictionary of trained models
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "-"*60)
    print("Training Ensemble Model (Voting)")
    print("-"*60)
    
    # Create voting classifier
    estimators = [(name, model) for name, model in models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "Ensemble")
    
    return ensemble, metrics


def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print metrics
    print(f"\n✓ {model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def plot_confusion_matrix(metrics, save_path=None):
    """
    Plot confusion matrix for a model.
    
    Args:
        metrics (dict): Model metrics containing confusion matrix
        save_path (str): Path to save the plot
    """
    cm = metrics['confusion_matrix']
    model_name = metrics['model']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.close()


def plot_roc_curves(all_metrics, y_test, save_path=None):
    """
    Plot ROC curves for all models.
    
    Args:
        all_metrics (dict): Dictionary of all model metrics
        y_test: True labels
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in all_metrics.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        auc = metrics['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved: {save_path}")
    
    plt.close()


def plot_metrics_comparison(all_metrics, save_path=None):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        all_metrics (dict): Dictionary of all model metrics
        save_path (str): Path to save the plot
    """
    # Prepare data
    models = list(all_metrics.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    data = {metric: [all_metrics[model][metric] for model in models] 
            for metric in metrics_names}
    
    # Create plot
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = width * (i - 2)
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), 
               color=color, alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved: {save_path}")
    
    plt.close()


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"✓ Model saved: {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        Trained model
    """
    model = joblib.load(filepath)
    print(f"✓ Model loaded: {filepath}")
    return model


def train_models(X_train, y_train, X_test, y_test, save_models=True):
    """
    Train all models and return results.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        save_models (bool): Whether to save trained models
        
    Returns:
        tuple: (models_dict, metrics_dict)
    """
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    models = {}
    all_metrics = {}
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    models['Logistic Regression'] = lr_model
    all_metrics['Logistic Regression'] = lr_metrics
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models['Random Forest'] = rf_model
    all_metrics['Random Forest'] = rf_metrics
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models['XGBoost'] = xgb_model
    all_metrics['XGBoost'] = xgb_metrics
    
    # Train Ensemble
    ensemble_model, ensemble_metrics = train_ensemble(models, X_train, y_train, X_test, y_test)
    models['Ensemble'] = ensemble_model
    all_metrics['Ensemble'] = ensemble_metrics
    
    # Save models
    if save_models:
        print("\n" + "-"*60)
        print("Saving Models")
        print("-"*60)
        for name, model in models.items():
            filename = name.lower().replace(' ', '_')
            save_model(model, f'models/{filename}_model.pkl')
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60 + "\n")
    
    return models, all_metrics


def create_metrics_summary(all_metrics):
    """
    Create a summary DataFrame of all metrics.
    
    Args:
        all_metrics (dict): Dictionary of all model metrics
        
    Returns:
        pd.DataFrame: Summary of metrics
    """
    summary_data = []
    
    for model_name, metrics in all_metrics.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'AUC-ROC': f"{metrics['roc_auc']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess_data
    
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data('../data/telco_churn.csv')
    models, metrics = train_models(X_train, y_train, X_test, y_test)
    
    # Print summary
    summary = create_metrics_summary(metrics)
    print("\nMetrics Summary:")
    print(summary.to_string(index=False))

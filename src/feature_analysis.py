"""
Feature Analysis Module
Analyzes feature importance and identifies key churn drivers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


def get_feature_importance_tree_based(model, feature_names, top_n=15):
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model (RandomForest, XGBoost)
        feature_names (list): List of feature names
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def get_feature_importance_logistic(model, feature_names, top_n=15):
    """
    Get feature importance from Logistic Regression using coefficients.
    
    Args:
        model: Trained Logistic Regression model
        feature_names (list): List of feature names
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'coef_'):
        # Use absolute values of coefficients
        importances = np.abs(model.coef_[0])
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    else:
        print("Model does not have coef_ attribute")
        return None


def plot_feature_importance(importance_df, model_name, save_path=None):
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df (pd.DataFrame): Feature importance dataframe
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    if importance_df is None or importance_df.empty:
        print("No feature importance data to plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {len(importance_df)} Feature Importance - {model_name}', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved: {save_path}")
    
    plt.close()


def analyze_churn_by_segment(df, segment_col, target_col='Churn'):
    """
    Analyze churn rate by different segments.
    
    Args:
        df (pd.DataFrame): Dataset
        segment_col (str): Column to segment by
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Churn analysis by segment
    """
    if segment_col not in df.columns or target_col not in df.columns:
        print(f"Column {segment_col} or {target_col} not found")
        return None
    
    analysis = df.groupby(segment_col).agg({
        target_col: ['count', 'sum', 'mean']
    }).round(4)
    
    analysis.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
    analysis = analysis.sort_values('Churn_Rate', ascending=False)
    
    return analysis


def plot_churn_by_segment(analysis_df, segment_name, save_path=None):
    """
    Plot churn rate by segment.
    
    Args:
        analysis_df (pd.DataFrame): Segment analysis dataframe
        segment_name (str): Name of the segment
        save_path (str): Path to save the plot
    """
    if analysis_df is None or analysis_df.empty:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(analysis_df))
    width = 0.35
    
    # Plot total customers
    ax.bar([i - width/2 for i in x], analysis_df['Total_Customers'], 
           width, label='Total Customers', color='skyblue', alpha=0.8)
    
    # Plot churned customers
    ax.bar([i + width/2 for i in x], analysis_df['Churned_Customers'], 
           width, label='Churned Customers', color='salmon', alpha=0.8)
    
    # Add churn rate line
    ax2 = ax.twinx()
    ax2.plot(x, analysis_df['Churn_Rate'] * 100, 'ro-', 
             linewidth=2, markersize=8, label='Churn Rate %')
    
    # Formatting
    ax.set_xlabel(segment_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Churn Analysis by {segment_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(analysis_df.index, rotation=45, ha='right')
    
    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Segment analysis plot saved: {save_path}")
    
    plt.close()


def identify_key_churn_drivers(models, feature_names, top_n=10):
    """
    Identify key churn drivers across all models.
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (list): List of feature names
        top_n (int): Number of top features
        
    Returns:
        dict: Feature importance for each model
    """
    print("\n" + "="*60)
    print("IDENTIFYING KEY CHURN DRIVERS")
    print("="*60 + "\n")
    
    all_importance = {}
    
    # XGBoost importance
    if 'XGBoost' in models:
        print("Analyzing XGBoost feature importance...")
        xgb_importance = get_feature_importance_tree_based(
            models['XGBoost'], feature_names, top_n
        )
        all_importance['XGBoost'] = xgb_importance
        
        if xgb_importance is not None:
            print("\nTop 10 Features (XGBoost):")
            for idx, row in xgb_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Random Forest importance
    if 'Random Forest' in models:
        print("\nAnalyzing Random Forest feature importance...")
        rf_importance = get_feature_importance_tree_based(
            models['Random Forest'], feature_names, top_n
        )
        all_importance['Random Forest'] = rf_importance
    
    # Logistic Regression importance
    if 'Logistic Regression' in models:
        print("\nAnalyzing Logistic Regression feature importance...")
        lr_importance = get_feature_importance_logistic(
            models['Logistic Regression'], feature_names, top_n
        )
        all_importance['Logistic Regression'] = lr_importance
    
    print("\n" + "="*60)
    print("CHURN DRIVER ANALYSIS COMPLETE")
    print("="*60 + "\n")
    
    return all_importance


def create_feature_importance_summary(all_importance, top_n=15):
    """
    Create a combined feature importance ranking.
    
    Args:
        all_importance (dict): Feature importance from all models
        top_n (int): Number of top features
        
    Returns:
        pd.DataFrame: Combined feature importance
    """
    # Collect all features and their average rank
    feature_ranks = {}
    
    for model_name, importance_df in all_importance.items():
        if importance_df is not None:
            for idx, row in importance_df.iterrows():
                feature = row['feature']
                if feature not in feature_ranks:
                    feature_ranks[feature] = []
                feature_ranks[feature].append(row['importance'])
    
    # Calculate average importance
    avg_importance = {
        feature: np.mean(importances) 
        for feature, importances in feature_ranks.items()
    }
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Feature': list(avg_importance.keys()),
        'Avg_Importance': list(avg_importance.values())
    }).sort_values('Avg_Importance', ascending=False).head(top_n)
    
    return summary_df


def generate_churn_insights(all_importance, segment_analyses):
    """
    Generate actionable insights from the analysis.
    
    Args:
        all_importance (dict): Feature importance from models
        segment_analyses (dict): Segment-wise churn analysis
        
    Returns:
        list: List of insights
    """
    insights = []
    
    # From feature importance
    if 'XGBoost' in all_importance and all_importance['XGBoost'] is not None:
        top_feature = all_importance['XGBoost'].iloc[0]['feature']
        insights.append(f"🔍 Top churn driver: {top_feature}")
    
    # From segment analysis
    for segment_name, analysis in segment_analyses.items():
        if analysis is not None and not analysis.empty:
            highest_churn_segment = analysis.iloc[0]
            churn_rate = highest_churn_segment['Churn_Rate'] * 100
            insights.append(
                f"⚠️  {segment_name}: '{highest_churn_segment.name}' has highest churn ({churn_rate:.1f}%)"
            )
    
    return insights


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess_data
    from model_training import train_models
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data('../data/telco_churn.csv')
    
    # Train models
    models, metrics = train_models(X_train, y_train, X_test, y_test, save_models=False)
    
    # Analyze feature importance
    feature_names = X_train.columns.tolist()
    all_importance = identify_key_churn_drivers(models, feature_names)
    
    # Create summary
    summary = create_feature_importance_summary(all_importance)
    print("\nFeature Importance Summary:")
    print(summary.to_string(index=False))

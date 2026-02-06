"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        raise


def clean_data(df):
    """
    Clean the dataset by handling missing values and data type issues.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric (handle spaces)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0 (new customers)
    df_clean['TotalCharges'].fillna(0, inplace=True)
    
    # Remove customerID as it's not useful for prediction
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    # Convert binary categorical variables to numeric
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
    
    # Convert gender to numeric
    if 'gender' in df_clean.columns:
        df_clean['gender'] = df_clean['gender'].map({'Male': 1, 'Female': 0})
    
    # Convert target variable
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"✓ Data cleaned: {df_clean.shape[0]} rows remaining")
    print(f"  - Missing values handled")
    print(f"  - Binary variables encoded")
    
    return df_clean


def engineer_features(df):
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df_feat = df.copy()
    
    # Tenure groups
    df_feat['tenure_group'] = pd.cut(df_feat['tenure'], 
                                      bins=[0, 12, 24, 48, 72],
                                      labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
    
    # Average monthly charges
    df_feat['avg_monthly_charges'] = df_feat['TotalCharges'] / (df_feat['tenure'] + 1)
    
    # Service usage score (count of additional services)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_cols:
        if col in df_feat.columns:
            df_feat[col + '_flag'] = df_feat[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    service_flag_cols = [col + '_flag' for col in service_cols if col in df_feat.columns]
    df_feat['service_count'] = df_feat[service_flag_cols].sum(axis=1)
    
    # High value customer flag (top 25% charges)
    df_feat['high_value_customer'] = (df_feat['MonthlyCharges'] > 
                                      df_feat['MonthlyCharges'].quantile(0.75)).astype(int)
    
    # Contract risk score
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    if 'Contract' in df_feat.columns:
        df_feat['contract_risk'] = df_feat['Contract'].map(contract_risk)
    
    print(f"✓ Features engineered:")
    print(f"  - Tenure groups created")
    print(f"  - Service usage metrics added")
    print(f"  - Customer value flags created")
    
    return df_feat


def encode_categorical_features(df, target_col='Churn'):
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df (pd.DataFrame): Dataset with features
        target_col (str): Name of target column
        
    Returns:
        tuple: (X, y) - Features and target
    """
    df_encoded = df.copy()
    
    # Separate features and target
    if target_col in df_encoded.columns:
        y = df_encoded[target_col]
        X = df_encoded.drop(target_col, axis=1)
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Get categorical columns (excluding already encoded ones)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"✓ Categorical encoding complete:")
    print(f"  - Original features: {X.shape[1]}")
    print(f"  - Encoded features: {X_encoded.shape[1]}")
    
    return X_encoded, y


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"✓ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def load_and_preprocess_data(filepath, test_size=0.2, random_state=42, scale=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath (str): Path to dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed
        scale (bool): Whether to scale features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n" + "="*60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Load data
    df = load_data(filepath)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_feat = engineer_features(df_clean)
    
    # Encode categorical features
    X, y = encode_categorical_features(df_feat)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Data split complete:")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Churn rate (train): {y_train.mean():.2%}")
    print(f"  - Churn rate (test): {y_test.mean():.2%}")
    
    # Scale features
    scaler = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60 + "\n")
    
    return X_train, X_test, y_train, y_test, scaler


def get_feature_names(X):
    """
    Get list of feature names.
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        list: Feature names
    """
    return X.columns.tolist()


if __name__ == "__main__":
    # Example usage
    filepath = "../data/telco_churn.csv"
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(filepath)
    print(f"\nFinal feature count: {X_train.shape[1]}")
    print(f"Sample features: {X_train.columns[:5].tolist()}")

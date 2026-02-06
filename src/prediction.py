"""
Prediction Module
Make predictions on new customer data and assess churn risk.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Trained model
    """
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        raise


def predict_churn(model, X, threshold=0.5):
    """
    Predict churn for given customer data.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Customer features
        threshold (float): Classification threshold
        
    Returns:
        tuple: (predictions, probabilities)
    """
    # Get probabilities
    probabilities = model.predict_proba(X)[:, 1]
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities


def predict_churn_risk_level(probabilities):
    """
    Categorize customers into risk levels based on churn probability.
    
    Args:
        probabilities (np.array): Churn probabilities
        
    Returns:
        np.array: Risk level categories
    """
    risk_levels = []
    
    for prob in probabilities:
        if prob < 0.3:
            risk_levels.append('Low Risk')
        elif prob < 0.6:
            risk_levels.append('Medium Risk')
        else:
            risk_levels.append('High Risk')
    
    return np.array(risk_levels)


def create_prediction_report(X, predictions, probabilities, customer_ids=None):
    """
    Create a comprehensive prediction report.
    
    Args:
        X (pd.DataFrame): Customer features
        predictions (np.array): Churn predictions
        probabilities (np.array): Churn probabilities
        customer_ids (list): Customer IDs (optional)
        
    Returns:
        pd.DataFrame: Prediction report
    """
    # Create base report
    report = pd.DataFrame({
        'Churn_Prediction': predictions,
        'Churn_Probability': probabilities,
        'Risk_Level': predict_churn_risk_level(probabilities)
    })
    
    # Add customer IDs if provided
    if customer_ids is not None:
        report.insert(0, 'Customer_ID', customer_ids)
    else:
        report.insert(0, 'Customer_ID', range(1, len(predictions) + 1))
    
    # Sort by probability (highest risk first)
    report = report.sort_values('Churn_Probability', ascending=False)
    
    return report


def identify_high_risk_customers(prediction_report, risk_threshold=0.7, top_n=None):
    """
    Identify high-risk customers for retention campaigns.
    
    Args:
        prediction_report (pd.DataFrame): Prediction report
        risk_threshold (float): Probability threshold for high risk
        top_n (int): Return top N customers (optional)
        
    Returns:
        pd.DataFrame: High-risk customers
    """
    high_risk = prediction_report[
        prediction_report['Churn_Probability'] >= risk_threshold
    ].copy()
    
    if top_n is not None:
        high_risk = high_risk.head(top_n)
    
    return high_risk


def get_retention_recommendations(prediction_report, X, top_features=None):
    """
    Generate retention recommendations for high-risk customers.
    
    Args:
        prediction_report (pd.DataFrame): Prediction report
        X (pd.DataFrame): Customer features
        top_features (list): Most important features for churn
        
    Returns:
        pd.DataFrame: Report with recommendations
    """
    # Focus on high-risk customers
    high_risk = identify_high_risk_customers(prediction_report, risk_threshold=0.6)
    
    recommendations = []
    
    for idx, row in high_risk.iterrows():
        customer_id = row['Customer_ID']
        probability = row['Churn_Probability']
        
        # Get customer features
        if isinstance(X.index[0], int):
            customer_features = X.iloc[idx]
        else:
            customer_features = X.loc[idx]
        
        # Generate recommendation based on features
        recommendation = generate_recommendation(customer_features, probability)
        
        recommendations.append({
            'Customer_ID': customer_id,
            'Churn_Risk': f"{probability:.2%}",
            'Recommendation': recommendation,
            'Priority': 'High' if probability > 0.8 else 'Medium'
        })
    
    return pd.DataFrame(recommendations)


def generate_recommendation(customer_features, probability):
    """
    Generate specific recommendation based on customer profile.
    
    Args:
        customer_features (pd.Series): Customer feature values
        probability (float): Churn probability
        
    Returns:
        str: Recommendation text
    """
    recommendations = []
    
    # Check for month-to-month contract
    if 'Contract_Month-to-month' in customer_features.index:
        if customer_features.get('Contract_Month-to-month', 0) == 1:
            recommendations.append("Offer long-term contract discount")
    
    # Check for high monthly charges
    if 'MonthlyCharges' in customer_features.index:
        if customer_features['MonthlyCharges'] > 70:
            recommendations.append("Provide loyalty discount or bundle offer")
    
    # Check for low tenure
    if 'tenure' in customer_features.index:
        if customer_features['tenure'] < 12:
            recommendations.append("Engage with onboarding support")
    
    # Check for payment method
    if 'PaymentMethod_Electronic check' in customer_features.index:
        if customer_features.get('PaymentMethod_Electronic check', 0) == 1:
            recommendations.append("Encourage automatic payment setup")
    
    # Check for lack of online security
    if 'OnlineSecurity_flag' in customer_features.index:
        if customer_features.get('OnlineSecurity_flag', 0) == 0:
            recommendations.append("Offer free trial of security services")
    
    # Default recommendation
    if not recommendations:
        if probability > 0.8:
            recommendations.append("Immediate retention call required")
        else:
            recommendations.append("Monitor and engage proactively")
    
    return " | ".join(recommendations[:3])  # Limit to top 3


def batch_predict(model, data_path, output_path=None):
    """
    Make predictions on a batch of customers from a CSV file.
    
    Args:
        model: Trained model
        data_path (str): Path to input CSV
        output_path (str): Path to save predictions (optional)
        
    Returns:
        pd.DataFrame: Predictions
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} customers from {data_path}")
    
    # Extract customer IDs if present
    customer_ids = df['customerID'].values if 'customerID' in df.columns else None
    
    # Preprocess (assuming same preprocessing as training)
    from data_preprocessing import clean_data, engineer_features, encode_categorical_features
    
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    
    # Remove target if present
    if 'Churn' in df_feat.columns:
        df_feat = df_feat.drop('Churn', axis=1)
    
    # Make predictions
    predictions, probabilities = predict_churn(model, df_feat)
    
    # Create report
    report = create_prediction_report(df_feat, predictions, probabilities, customer_ids)
    
    # Save if output path provided
    if output_path:
        report.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to {output_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Total Customers: {len(report)}")
    print(f"  Predicted Churners: {predictions.sum()} ({predictions.mean():.2%})")
    print(f"  High Risk (>70%): {(probabilities > 0.7).sum()}")
    print(f"  Medium Risk (30-70%): {((probabilities >= 0.3) & (probabilities <= 0.7)).sum()}")
    print(f"  Low Risk (<30%): {(probabilities < 0.3).sum()}")
    
    return report


def predict_single_customer(model, customer_data):
    """
    Make prediction for a single customer.
    
    Args:
        model: Trained model
        customer_data (dict or pd.Series): Customer features
        
    Returns:
        dict: Prediction result
    """
    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    else:
        customer_df = pd.DataFrame([customer_data])
    
    # Make prediction
    prediction, probability = predict_churn(model, customer_df)
    risk_level = predict_churn_risk_level(probability)[0]
    
    result = {
        'will_churn': bool(prediction[0]),
        'churn_probability': float(probability[0]),
        'risk_level': risk_level,
        'recommendation': generate_recommendation(customer_df.iloc[0], probability[0])
    }
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Prediction Module - Example Usage\n")
    
    # Load model
    model = load_trained_model('models/xgboost_model.pkl')
    
    # Example: Batch prediction
    # predictions = batch_predict(model, 'data/new_customers.csv', 'results/predictions.csv')
    
    print("\nModule ready for predictions!")

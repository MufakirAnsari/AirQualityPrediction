import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import r2_score
import sys
import logging

logger = logging.getLogger(__name__)

def plot_regression_predictions_sample(sample_size=100):
    """
    Plot regression model predictions on random sample like example.jpg
    """
    logger.info("Creating regression prediction plots on random sample")
    
    # Load cleaned data
    try:
        df = pd.read_csv('artifacts/data_cleaned.csv')
    except FileNotFoundError:
        logger.error("artifacts/data_cleaned.csv not found. Please run the main pipeline first.")
        return
    
    # Filter data with PM2.5 values
    df_clean = df.dropna(subset=['PM2.5(µg/m³)'])
    
    if len(df_clean) < sample_size:
        logger.warning(f"Only {len(df_clean)} samples available")
        sample_size = len(df_clean)
    
    # Sample data
    sample_df = df_clean.sample(n=sample_size, random_state=42)
    
    # Model directory
    model_dir = 'artifacts/models/regression'
    
    # Check if regression models exist
    if not os.path.exists(model_dir):
        logger.error(f"Regression model directory {model_dir} not found")
        return
    
    models = ['LinearRegression', 'DecisionTree', 'RandomForest', 
              'XGBoost', 'LightGBM', 'SVM', 'MLP']
    
    available_models = []
    for model_name in models:
        if os.path.exists(f'{model_dir}/{model_name}.joblib'):
            available_models.append(model_name)
    
    if len(available_models) == 0:
        logger.error("No regression models found. Please train regression models first.")
        return
    
    logger.info(f"Found {len(available_models)} models: {available_models}")
    
    # Load preprocessor
    preprocessor_path = f'{model_dir}/preprocessor.joblib'
    if not os.path.exists(preprocessor_path):
        logger.error("Preprocessor not found for regression models")
        return
    
    preprocessor = joblib.load(preprocessor_path)
    
    # Prepare features
    feature_cols = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 
        'PM10(µg/m³)', 'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    available_features = [col for col in feature_cols if col in sample_df.columns]
    X_sample = sample_df[available_features]
    y_true = sample_df['PM2.5(µg/m³)']
    
    # Transform features
    X_processed = preprocessor.transform(X_sample)
    
    # Get predictions from all models
    predictions = {}
    r2_scores = {}
    
    for model_name in available_models:
        try:
            model = joblib.load(f'{model_dir}/{model_name}.joblib')
            y_pred = model.predict(X_processed)
            predictions[model_name] = y_pred
            r2_scores[model_name] = r2_score(y_true, y_pred)
            logger.info(f"{model_name} R²: {r2_scores[model_name]:.3f}")
        except Exception as e:
            logger.error(f"Error loading or predicting with {model_name}: {e}")
            continue
    
    if len(predictions) == 0:
        logger.error("No successful predictions made")
        return
    
    # Create the plot like example.jpg
    n_models = len(predictions)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    plt.figure(figsize=(ncols * 5, nrows * 5))
    
    for idx, (model_name, y_pred) in enumerate(predictions.items(), 1):
        plt.subplot(nrows, ncols, idx)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Labels and title
        plt.xlabel('Actual PM2.5 Values')
        plt.ylabel('Predicted PM2.5 Values')
        plt.title(f'{model_name}\nR² Score: {r2_scores[model_name]:.3f}')
        
        # Grid
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparison of Model Predictions vs. Actual Values\n'
                 f'Random Sample (n={len(y_true)})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    output_path = f'{model_dir}/sampled_predictions_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sampled predictions plot saved to {output_path}")
    
    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 40)
    for model_name in sorted(r2_scores.keys(), key=lambda x: r2_scores[x], reverse=True):
        print(f"{model_name:15}: R² = {r2_scores[model_name]:.4f}")

def plot_classification_predictions_sample(season='Spring', sample_size=100):
    """
    Plot classification model predictions on random sample
    """
    logger.info(f"Creating classification prediction plots for {season}")
    
    # Load cleaned data
    try:
        df = pd.read_csv('artifacts/data_cleaned.csv')
    except FileNotFoundError:
        logger.error("artifacts/data_cleaned.csv not found")
        return
    
    # Filter by season
    df_season = df[df['Season'] == season]
    
    if len(df_season) < sample_size:
        logger.warning(f"Only {len(df_season)} samples available for {season}")
        sample_size = len(df_season)
    
    sample_df = df_season.sample(n=sample_size, random_state=42)
    
    # Model directory
    model_dir = f'artifacts/models/{season.lower()}'
    
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} not found")
        return
    
    # Check available models
    models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM',
              'DecisionTree', 'SVM', 'MLP']
    
    available_models = []
    for model_name in models:
        if os.path.exists(f'{model_dir}/{model_name}.joblib'):
            available_models.append(model_name)
    
    if len(available_models) == 0:
        logger.error(f"No classification models found for {season}")
        return
    
    # Load components
    try:
        preprocessor = joblib.load(f'{model_dir}/preprocessor.joblib')
        label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
    except FileNotFoundError as e:
        logger.error(f"Missing component: {e}")
        return
    
    # Prepare features and target
    feature_cols = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 
        'PM2.5(µg/m³)', 'PM10(µg/m³)', 
        'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    available_features = [col for col in feature_cols if col in sample_df.columns]
    X_sample = sample_df[available_features]
    y_true = sample_df['AQI_Category']
    
    # Transform
    X_processed = preprocessor.transform(X_sample)
    y_true_encoded = label_encoder.transform(y_true)
    
    # Print summary
    print(f"\n{season} Classification Model Comparison:")
    print("-" * 50)
    for model_name in available_models:
        try:
            model = joblib.load(f'{model_dir}/{model_name}.joblib')
            y_pred = model.predict(X_processed)
            accuracy = (y_pred == y_true_encoded).mean()
            print(f"{model_name:15}: Accuracy = {accuracy:.4f}")
        except Exception as e:
            print(f"{model_name:15}: Error - {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'classification':
        season = sys.argv[2] if len(sys.argv) > 2 else 'Spring'
        plot_classification_predictions_sample(season)
    else:
        plot_regression_predictions_sample()
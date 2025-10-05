import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import accuracy_score, r2_score
import logging

logger = logging.getLogger(__name__)

def compare_classification_models(season='Spring', sample_size=100):
    """
    Compare classification models using random samples
    """
    logger.info(f"Comparing classification models for {season}")
    
    # Load data
    df = pd.read_csv('artifacts/data_cleaned.csv')
    df_season = df[df['Season'] == season]
    
    if len(df_season) < sample_size:
        logger.warning(f"Only {len(df_season)} samples available for {season}")
        sample_size = len(df_season)
    
    # Sample data
    sample_df = df_season.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Model directory
    model_dir = f'artifacts/models/{season.lower()}'
    
    # Check if models exist
    models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 
              'DecisionTree', 'SVM', 'MLP']
    
    available_models = []
    for model_name in models:
        if os.path.exists(f'{model_dir}/{model_name}.joblib'):
            available_models.append(model_name)
    
    if len(available_models) < 2:
        logger.error(f"Not enough models available for comparison in {season}")
        return
    
    # Load components
    preprocessor = joblib.load(f'{model_dir}/preprocessor.joblib')
    label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
    
    # Prepare features
    feature_cols = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 
        'PM2.5(µg/m³)', 'PM10(µg/m³)', 
        'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    available_features = [col for col in feature_cols if col in sample_df.columns]
    X_sample = sample_df[available_features]
    y_true = sample_df['AQI_Category']
    
    # Transform features
    X_processed = preprocessor.transform(X_sample)
    y_true_encoded = label_encoder.transform(y_true)
    
    # Get predictions from all models
    predictions = {}
    accuracies = {}
    
    for model_name in available_models:
        model = joblib.load(f'{model_dir}/{model_name}.joblib')
        y_pred = model.predict(X_processed)
        predictions[model_name] = y_pred
        accuracies[model_name] = accuracy_score(y_true_encoded, y_pred)
    
    # Create comparison plot
    create_classification_comparison_plot(y_true_encoded, predictions, accuracies, 
                                        season, model_dir)
    
    logger.info(f"Classification model comparison completed for {season}")

def compare_regression_models(sample_size=100):
    """
    Compare regression models using random samples
    """
    logger.info("Comparing regression models")
    
    # Load data
    df = pd.read_csv('artifacts/data_cleaned.csv')
    df_clean = df.dropna(subset=['PM2.5(µg/m³)'])
    
    if len(df_clean) < sample_size:
        logger.warning(f"Only {len(df_clean)} samples available")
        sample_size = len(df_clean)
    
    # Sample data
    sample_df = df_clean.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Model directory
    model_dir = 'artifacts/models/regression'
    
    # Check if models exist
    models = ['LinearRegression', 'DecisionTree', 'RandomForest', 'XGBoost', 
              'LightGBM', 'SVM', 'MLP']
    
    available_models = []
    for model_name in models:
        if os.path.exists(f'{model_dir}/{model_name}.joblib'):
            available_models.append(model_name)
    
    if len(available_models) < 2:
        logger.error("Not enough regression models available for comparison")
        return
    
    # Load preprocessor
    preprocessor = joblib.load(f'{model_dir}/preprocessor.joblib')
    
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
        model = joblib.load(f'{model_dir}/{model_name}.joblib')
        y_pred = model.predict(X_processed)
        predictions[model_name] = y_pred
        r2_scores[model_name] = r2_score(y_true, y_pred)
    
    # Create comparison plot
    create_regression_comparison_plot(y_true, predictions, r2_scores, model_dir)
    
    logger.info("Regression model comparison completed")

def create_classification_comparison_plot(y_true, predictions, accuracies, season, model_dir):
    """Create comparison plot for classification models"""
    
    n_models = len(predictions)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        
        # Scatter plot of actual vs predicted (encoded values)
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Category (encoded)')
        ax.set_ylabel('Predicted Category (encoded)')
        ax.set_title(f'{model_name}\nAccuracy = {accuracies[model_name]:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle(f'{season} Classification Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_dir}/classification_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_regression_comparison_plot(y_true, predictions, r2_scores, model_dir):
    """Create comparison plot for regression models"""
    
    n_models = len(predictions)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        
        # Scatter plot of actual vs predicted
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual PM2.5 (µg/m³)')
        ax.set_ylabel('Predicted PM2.5 (µg/m³)')
        ax.set_title(f'{model_name}\nR² = {r2_scores[model_name]:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('PM2.5 Regression Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_dir}/sample_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_all_models():
    """Compare models for all seasons and regression"""
    
    # Compare classification models for each season
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    for season in seasons:
        try:
            compare_classification_models(season)
        except Exception as e:
            logger.error(f"Error comparing classification models for {season}: {e}")
    
    # Compare regression models
    try:
        compare_regression_models()
    except Exception as e:
        logger.error(f"Error comparing regression models: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compare_all_models()
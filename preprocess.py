import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, target_col, season):
    """
    Preprocess data for classification (seasonal)
    """
    logger.info(f"Preprocessing data for {season}")
    
    # Feature columns for classification
    feature_cols = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 
        'PM2.5(µg/m³)', 'PM10(µg/m³)', 
        'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    # Select available features
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {available_features}")
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_temp, X_pred, y_temp, y_pred = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_eval_processed = preprocessor.transform(X_eval)
    X_pred_processed = preprocessor.transform(X_pred)
    
    # Save preprocessor
    model_dir = f'artifacts/models/{season.lower()}'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, f'{model_dir}/preprocessor.joblib')
    
    logger.info(f"Training samples: {X_train_processed.shape[0]}")
    logger.info(f"Evaluation samples: {X_eval_processed.shape[0]}")
    logger.info(f"Prediction samples: {X_pred_processed.shape[0]}")
    
    return (X_train_processed, X_eval_processed, X_pred_processed,
            y_train, y_eval, y_pred, preprocessor, available_features)

def preprocess_for_regression(df, target_col='PM2.5(µg/m³)'):
    """
    Preprocess data for regression (PM2.5 prediction)
    """
    logger.info("Preprocessing data for regression")
    
    # Feature columns for regression (exclude PM2.5 as it's the target)
    feature_cols = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 
        'PM10(µg/m³)', 'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    # Select available features
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available regression features: {available_features}")
    
    # Drop rows where target is missing
    df_clean = df.dropna(subset=[target_col])
    
    X = df_clean[available_features].copy()
    y = df_clean[target_col].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    model_dir = 'artifacts/models/regression'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, f'{model_dir}/preprocessor.joblib')
    
    logger.info(f"Regression training samples: {X_train_processed.shape[0]}")
    logger.info(f"Regression test samples: {X_test_processed.shape[0]}")
    
    return (X_train_processed, X_test_processed, y_train, y_test, 
            preprocessor, available_features)
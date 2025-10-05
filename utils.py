import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_directory_structure():
    """Create necessary directory structure for the project"""
    directories = [
        'artifacts',
        'artifacts/data',
        'artifacts/models',
        'artifacts/models/regression',
        'artifacts/evaluation',
        'artifacts/evaluation/combined',
        'artifacts/eda',
        'artifacts/plots'
    ]
    
    # Create seasonal directories
    seasons = ['spring', 'summer', 'autumn', 'winter']
    for season in seasons:
        directories.extend([
            f'artifacts/models/{season}',
            f'artifacts/evaluation/{season}',
            f'artifacts/eda/{season}'
        ])
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Directory structure created successfully")

def calculate_aqi_from_pollutants(row):
    """
    Calculate AQI from pollutant concentrations
    Uses PM2.5 as primary pollutant for simplification
    """
    pm25 = row.get('PM2.5(µg/m³)', np.nan)
    
    if pd.isna(pm25):
        return np.nan
    
    # PM2.5 breakpoints (µg/m³) and corresponding AQI values
    breakpoints = [
        (0, 12, 0, 50),      # Good
        (12.1, 35.4, 51, 100),   # Moderate
        (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200), # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 500, 301, 500)    # Hazardous
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
    
    # If concentration is beyond the highest breakpoint
    return 500

def get_aqi_category_from_value(aqi_value):
    """Convert AQI numerical value to category"""
    if pd.isna(aqi_value):
        return 'Unknown'
    elif aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Moderate'
    elif aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi_value <= 200:
        return 'Unhealthy'
    elif aqi_value <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def validate_data_quality(df, critical_columns=None):
    """
    Validate data quality and report issues
    """
    if critical_columns is None:
        critical_columns = ['Date', 'PM2.5(µg/m³)', 'Season']
    
    logger.info("Validating data quality...")
    
    issues = []
    
    # Check for missing critical columns
    missing_cols = [col for col in critical_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing critical columns: {missing_cols}")
    
    # Check data types
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            issues.append("Date column is not datetime type")
    
    # Check for excessive missing values
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        issues.append(f"Columns with >50% missing values: {high_missing.to_dict()}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for outliers in PM2.5
    if 'PM2.5(µg/m³)' in df.columns:
        pm25_values = df['PM2.5(µg/m³)'].dropna()
        if not pm25_values.empty:
            q99 = pm25_values.quantile(0.99)
            extreme_values = (pm25_values > q99 * 3).sum()
            if extreme_values > 0:
                issues.append(f"Found {extreme_values} extreme PM2.5 values (>3x 99th percentile)")
    
    if issues:
        logger.warning("Data quality issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data quality validation passed")
    
    return issues

def generate_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {}
    
    summary['basic_info'] = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': df.columns.tolist()
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    # Missing values
    missing_values = df.isnull().sum()
    summary['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # Date range if date column exists
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        summary['date_range'] = {
            'start': df['Date'].min(),
            'end': df['Date'].max(),
            'span_days': (df['Date'].max() - df['Date'].min()).days
        }
    
    return summary

def plot_data_overview(df, save_path='artifacts/plots/data_overview.png'):
    """Create overview plots of the dataset"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Missing values heatmap
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data[missing_data > 0].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Missing Values by Column')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
    else:
        axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        axes[0,0].set_title('Missing Values')
    
    # 2. PM2.5 distribution
    if 'PM2.5(µg/m³)' in df.columns:
        df['PM2.5(µg/m³)'].hist(bins=50, ax=axes[0,1], alpha=0.7)
        axes[0,1].set_title('PM2.5 Distribution')
        axes[0,1].set_xlabel('PM2.5 (µg/m³)')
        axes[0,1].set_ylabel('Frequency')
    
    # 3. Seasonal distribution
    if 'Season' in df.columns:
        df['Season'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Seasonal Distribution')
        axes[1,0].set_ylabel('')
    
    # 4. AQI Category distribution
    if 'AQI_Category' in df.columns:
        df['AQI_Category'].value_counts().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('AQI Category Distribution')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Data overview plot saved to {save_path}")

def save_model_metadata(model_info, file_path):
    """Save model training metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info
    }
    
    import json
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def load_model_safely(model_path):
    """Safely load a model with error handling"""
    try:
        import joblib
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def check_model_compatibility(model, feature_names):
    """Check if model is compatible with given features"""
    try:
        # For scikit-learn models
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
            if len(model_features) != len(feature_names):
                return False, f"Feature count mismatch: {len(model_features)} vs {len(feature_names)}"
        
        # Try a dummy prediction
        dummy_input = np.zeros((1, len(feature_names)))
        model.predict(dummy_input)
        return True, "Compatible"
        
    except Exception as e:
        return False, f"Compatibility check failed: {e}"

def get_feature_importance(model, feature_names):
    """Extract feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
            return dict(zip(feature_names, importance))
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to extract feature importance: {e}")
        return None
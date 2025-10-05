import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


# Fix for feature engineering - better handling of missing temporal data

def create_advanced_features(df):
    """
    Create advanced features from the base air quality data with better missing data handling
    """
    logger.info("Creating advanced features...")
    
    df_enhanced = df.copy()
    
    # 1. Interaction features (ratios and products)
    pollutant_cols = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    
    if len(available_pollutants) >= 2:
        # Ratios with better zero handling
        df_enhanced['CO_NO2_ratio'] = df_enhanced['CO (mg/m3)'] / (df_enhanced['NO2(µg/m³)'] + 0.001)
        df_enhanced['PM25_PM10_ratio'] = df_enhanced['PM2.5(µg/m³)'] / (df_enhanced['PM10(µg/m³)'] + 0.001)
        df_enhanced['SO2_NO2_ratio'] = df_enhanced['SO2(µg/m³)'] / (df_enhanced['NO2(µg/m³)'] + 0.001)
        
        # Products (interaction terms)
        df_enhanced['CO_NO2_product'] = df_enhanced['CO (mg/m3)'] * df_enhanced['NO2(µg/m³)']
        df_enhanced['PM_combined'] = df_enhanced['PM2.5(µg/m³)'] + df_enhanced['PM10(µg/m³)']
    
    # 2. Temporal features - CHECK IF DATE COLUMN EXISTS AND HAS VALUES
    if 'Date' in df_enhanced.columns and df_enhanced['Date'].notna().sum() > 0:
        # Only process non-null dates
        valid_dates = df_enhanced['Date'].notna()
        
        df_enhanced['DayOfWeek'] = np.nan
        df_enhanced['Month'] = np.nan
        df_enhanced['Quarter'] = np.nan
        df_enhanced['Month_sin'] = np.nan
        df_enhanced['Month_cos'] = np.nan
        
        # Set values only for valid dates
        df_enhanced.loc[valid_dates, 'DayOfWeek'] = df_enhanced.loc[valid_dates, 'Date'].dt.dayofweek
        df_enhanced.loc[valid_dates, 'Month'] = df_enhanced.loc[valid_dates, 'Date'].dt.month
        df_enhanced.loc[valid_dates, 'Quarter'] = df_enhanced.loc[valid_dates, 'Date'].dt.quarter
        
        # Weekend calculation
        df_enhanced['IsWeekend'] = 0
        df_enhanced.loc[valid_dates, 'IsWeekend'] = (df_enhanced.loc[valid_dates, 'DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding for valid months
        valid_months = df_enhanced['Month'].notna()
        if valid_months.sum() > 0:
            df_enhanced.loc[valid_months, 'Month_sin'] = np.sin(2 * np.pi * df_enhanced.loc[valid_months, 'Month'] / 12)
            df_enhanced.loc[valid_months, 'Month_cos'] = np.cos(2 * np.pi * df_enhanced.loc[valid_months, 'Month'] / 12)
    
    # Handle Hour column similarly
    if 'Hour' in df_enhanced.columns and df_enhanced['Hour'].notna().sum() > 0:
        df_enhanced['Hour_sin'] = np.nan
        df_enhanced['Hour_cos'] = np.nan
        
        valid_hours = df_enhanced['Hour'].notna()
        if valid_hours.sum() > 0:
            df_enhanced.loc[valid_hours, 'Hour_sin'] = np.sin(2 * np.pi * df_enhanced.loc[valid_hours, 'Hour'] / 24)
            df_enhanced.loc[valid_hours, 'Hour_cos'] = np.cos(2 * np.pi * df_enhanced.loc[valid_hours, 'Hour'] / 24)
    
    # 3. Rolling window features - only if we have enough data and sorted dates
    if 'Date' in df_enhanced.columns and len(df_enhanced) > 6 and df_enhanced['Date'].notna().sum() > 6:
        df_enhanced = df_enhanced.sort_values('Date')
        
        # Rolling averages (3-hour and 6-hour windows) - only for available pollutants
        for col in available_pollutants:
            if col in df_enhanced.columns:
                df_enhanced[f'{col}_3h_mean'] = df_enhanced[col].rolling(window=3, min_periods=1).mean()
                df_enhanced[f'{col}_6h_mean'] = df_enhanced[col].rolling(window=6, min_periods=1).mean()
                df_enhanced[f'{col}_3h_std'] = df_enhanced[col].rolling(window=3, min_periods=1).std().fillna(0)
                
        # Lag features (previous 1 and 2 time steps)
        for col in available_pollutants:
            if col in df_enhanced.columns:
                df_enhanced[f'{col}_lag1'] = df_enhanced[col].shift(1)
                df_enhanced[f'{col}_lag2'] = df_enhanced[col].shift(2)
    
    # 4. Meteorological interaction features
    if 'Temp(°C)' in df_enhanced.columns and 'RH (%)' in df_enhanced.columns:
        # Heat index approximation
        df_enhanced['Heat_Index'] = df_enhanced['Temp(°C)'] + 0.5 * (df_enhanced['RH (%)'] / 100)
        # Temperature-humidity interaction
        df_enhanced['Temp_RH_product'] = df_enhanced['Temp(°C)'] * df_enhanced['RH (%)']
    
    # 5. AQI-related features
    if 'PM2.5(µg/m³)' in df_enhanced.columns:
        # PM2.5 severity bins
        df_enhanced['PM25_severity'] = pd.cut(df_enhanced['PM2.5(µg/m³)'], 
                                             bins=[0, 12, 35.4, 55.4, 150.4, float('inf')],
                                             labels=[0, 1, 2, 3, 4])
        df_enhanced['PM25_severity'] = df_enhanced['PM25_severity'].astype(float)
        
        # Moving average difference (trend indicator) - only if we have enough data
        if len(df_enhanced) > 3:
            df_enhanced['PM25_trend'] = (df_enhanced['PM2.5(µg/m³)'] - 
                                       df_enhanced['PM2.5(µg/m³)'].rolling(window=3, min_periods=1).mean())
    
    logger.info(f"Enhanced features created. New shape: {df_enhanced.shape}")
    
    # Log which features have high missing values
    missing_pct = df_enhanced.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        logger.warning(f"Features with >50% missing values: {dict(high_missing)}")
    
    new_features = set(df_enhanced.columns) - set(df.columns)
    logger.info(f"New features added: {new_features}")
    
    return df_enhanced


def get_enhanced_feature_columns():
    """Return list of all possible enhanced feature columns"""
    base_features = [
        'CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)',
        'Temp(°C)', 'RH (%)', 'Hour'
    ]
    
    enhanced_features = [
        'CO_NO2_ratio', 'PM25_PM10_ratio', 'SO2_NO2_ratio',
        'CO_NO2_product', 'PM_combined',
        'DayOfWeek', 'Month', 'IsWeekend', 'Quarter',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
        'Heat_Index', 'Temp_RH_product', 'PM25_severity', 'PM25_trend'
    ]
    
    # Add rolling and lag features
    pollutants = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)']
    for pol in pollutants:
        enhanced_features.extend([
            f'{pol}_3h_mean', f'{pol}_6h_mean', f'{pol}_3h_std',
            f'{pol}_lag1', f'{pol}_lag2'
        ])
    
    return base_features + enhanced_features
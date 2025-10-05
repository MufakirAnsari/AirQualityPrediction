import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """
    Load raw air quality data, clean and prepare it for modeling,
    save cleaned data to artifacts/data_cleaned.csv.
    """
    logger.info("=== Loading and Preparing Data ===")

    try:
        df = pd.read_csv("Airquality9.csv")
        logger.info(f"Raw data loaded with shape: {df.shape}")
    except FileNotFoundError:
        logger.error("Airquality9.csv not found. Please ensure the data file exists.")
        raise

    # Clean column names, strip leading/trailing whitespace
    df.columns = df.columns.str.strip()

    logger.info(f"Columns after stripping: {list(df.columns)}")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract month and define Seasons
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(get_season)

    # Add 'Hour' from 'Time' column - use stripped column name 'Time'
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    else:
        logger.warning("No 'Time' column found after stripping; 'Hour' feature will be missing.")
        df['Hour'] = np.nan

    # Convert pollutant columns to numeric
    pollutant_cols = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)']
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert meteorological columns to numeric
    met_cols = ['Temp(°C)', 'RH (%)']
    for col in met_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create AQI_Category if not present
    if 'AQI_Category' not in df.columns:
        df['AQI_Category'] = df['PM2.5(µg/m³)'].apply(get_aqi_category)
        logger.info("Created AQI_Category from PM2.5 values")

    # Drop rows with missing critical values
    critical_cols = ['PM2.5(µg/m³)', 'Season']
    df = df.dropna(subset=critical_cols)

    logger.info(f"Data shape after cleaning: {df.shape}")

    # Ensure artifacts folder exists and save cleaned data
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/data_cleaned.csv", index=False)
    logger.info("Saved cleaned data to artifacts/data_cleaned.csv")

    return df, 'AQI_Category'

def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    else:
        return 'Autumn'

def get_aqi_category(pm25):
    if pd.isna(pm25):
        return 'Unknown'
    if pm25 <= 12:
        return 'Good'
    elif pm25 <= 35.4:
        return 'Moderate'
    elif pm25 <= 55.4:
        return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:
        return 'Unhealthy'
    elif pm25 <= 250.4:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, target = load_and_prepare_data()
    print(f"Data prepared with target: {target}, shape: {df.shape}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def perform_eda(df, target_col, season):
    """
    Perform exploratory data analysis
    """
    logger.info(f"Performing EDA for {season}")
    
    # Create EDA directory
    eda_dir = f'artifacts/eda/{season.lower()}'
    os.makedirs(eda_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Basic statistics
    stats = {
        'data_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df[target_col].value_counts().to_dict()
    }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Target distribution
    df[target_col].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title(f'{target_col} Distribution - {season}')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. PM2.5 distribution
    if 'PM2.5(µg/m³)' in df.columns:
        axes[0,1].hist(df['PM2.5(µg/m³)'].dropna(), bins=30, alpha=0.7)
        axes[0,1].set_title(f'PM2.5 Distribution - {season}')
        axes[0,1].set_xlabel('PM2.5 (µg/m³)')
    
    # 3. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title(f'Correlation Matrix - {season}')
    
    # 4. Time series plot of PM2.5
    if 'Date' in df.columns and 'PM2.5(µg/m³)' in df.columns:
        df_sorted = df.sort_values('Date')
        axes[1,1].plot(df_sorted['Date'], df_sorted['PM2.5(µg/m³)'], alpha=0.7)
        axes[1,1].set_title(f'PM2.5 Over Time - {season}')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{eda_dir}/eda_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plots
    create_additional_plots(df, eda_dir, season)
    
    logger.info(f"EDA completed for {season}. Plots saved to {eda_dir}")
    return stats

def create_additional_plots(df, eda_dir, season):
    """Create additional EDA plots"""
    
    # Pollutant boxplots by AQI category
    pollutant_cols = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    
    if len(available_pollutants) > 0 and 'AQI_Category' in df.columns:
        fig, axes = plt.subplots(1, len(available_pollutants), figsize=(20, 5))
        if len(available_pollutants) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_pollutants):
            sns.boxplot(data=df, x='AQI_Category', y=col, ax=axes[i])
            axes[i].set_title(f'{col} by AQI Category')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{eda_dir}/pollutants_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Meteorological conditions
    met_cols = ['Temp(°C)', 'RH (%)']
    available_met = [col for col in met_cols if col in df.columns]
    
    if len(available_met) > 0:
        fig, axes = plt.subplots(1, len(available_met), figsize=(12, 5))
        if len(available_met) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_met):
            axes[i].hist(df[col].dropna(), bins=20, alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{eda_dir}/meteorological_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
"""

Comprehensive seasonal EDA with modern seaborn style.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)

def perform_eda(df, target_col, season):
    logger.info(f"Performing EDA for {season}")
    
    save_dir = f'artifacts/eda/{season.lower()}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Target distribution
    plt.figure(figsize=(10, 6))
    target_counts = df[target_col].value_counts()
    colors = sns.color_palette('husl', len(target_counts))
    bars = plt.bar(target_counts.index, target_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    plt.title(f'{season} - AQI Category Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('AQI Category', fontsize=12)
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature distributions by AQI category
    numeric_cols = ['CO (mg/m3)', 'SO2(µg/m³)', 'NO2(µg/m³)', 'PM2.5(µg/m³)', 'PM10(µg/m³)', 'Temp(°C)', 'RH (%)']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.violinplot(data=df, x=target_col, y=col, ax=axes[i], palette='husl')
            axes[i].set_title(f'{col} by AQI Category', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    if len(numeric_cols) < len(axes):
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle(f'{season} - Feature Distributions by AQI Category', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Temporal patterns (if DateTime column available)
    if 'DateTime' in df.columns:
        df_temp = df.copy()
        df_temp['Hour'] = df_temp['DateTime'].dt.hour
        df_temp['DayOfWeek'] = df_temp['DateTime'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Hourly patterns
        if 'AQI' in df.columns:
            hourly_aqi = df_temp.groupby('Hour')['AQI'].mean()
            axes[0].plot(hourly_aqi.index, hourly_aqi.values, marker='o', linewidth=2, markersize=6)
            axes[0].set_title(f'{season} - Average AQI by Hour', fontweight='bold')
            axes[0].set_xlabel('Hour of Day')
            axes[0].set_ylabel('Average AQI')
            axes[0].grid(True, alpha=0.3)
            
            # Weekly patterns
            full_week = pd.Series(index=range(7), dtype=float)
            weekly_aqi = df_temp.groupby('DayOfWeek')['AQI'].mean()
            weekly_aqi_full = full_week.add(weekly_aqi, fill_value=np.nan)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            axes[1].bar(range(7), weekly_aqi_full.values, color=sns.color_palette('husl', 7), alpha=0.8)
            axes[1].set_title(f'{season} - Average AQI by Day of Week', fontweight='bold')
            axes[1].set_xlabel('Day of Week')
            axes[1].set_ylabel('Average AQI')
            axes[1].set_xticks(range(7))
            axes[1].set_xticklabels(days)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/temporal_patterns.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. PM2.5 distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['PM2.5(µg/m³)'].dropna(), bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title(f'{season} - PM2.5 Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('PM2.5 (µg/m³)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pm25_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"EDA completed for {season}. Plots saved to {save_dir}")
    
    return {
        'season': season,
        'sample_count': len(df),
        'target_distribution': df[target_col].value_counts().to_dict(),
        'eda_dir': save_dir,
        'data_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict()
    }


if __name__ == "__main__":
    from load_data import load_and_prepare_data
    df, target = load_and_prepare_data()
    
    for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
        season_data = df[df['Season'] == season]
        if len(season_data) > 0:
            perform_eda(season_data, target, season)
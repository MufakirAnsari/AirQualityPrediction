import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           r2_score, mean_absolute_error, mean_squared_error,
                           classification_report, confusion_matrix)
import os
import logging

logger = logging.getLogger(__name__)

def calculate_extended_metrics(y_true, y_pred, task_type='classification'):
    """
    Calculate extended metrics for both classification and regression
    """
    metrics = {}
    
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        unique_classes = np.unique(y_true)
        for i, cls in enumerate(unique_classes):
            cls_f1 = f1_score(y_true, y_pred, labels=[cls], average=None)
            if len(cls_f1) > 0:
                metrics[f'f1_class_{cls}'] = cls_f1[0]
        
    elif task_type == 'regression':
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Additional regression metrics
        if np.mean(y_true) != 0:
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            metrics['mape'] = np.inf
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = pd.Series(residuals).skew()
        
        # Explained variance
        metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_true))
    
    return metrics

def create_confusion_matrix_plot(y_true, y_pred, labels, save_path):
    """Create and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plots(y_true, y_pred, save_path):
    """Create residual plots for regression analysis"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Residuals vs Predicted
    axes[0,0].scatter(y_pred, residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Predicted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[0,1].hist(residuals, bins=30, alpha=0.7)
    axes[0,1].set_xlabel('Residuals')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Residuals')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Actual
    axes[1,1].scatter(y_true, residuals, alpha=0.6)
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    axes[1,1].set_xlabel('Actual Values')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_error_analysis(y_true, y_pred, X_test, feature_names, save_dir, task_type='classification'):
    """
    Perform detailed error analysis
    """
    logger.info("Performing error analysis...")
    
    if task_type == 'classification':
        # Find misclassified samples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        if len(error_indices) > 0:
            # Analyze error patterns
            error_data = pd.DataFrame(X_test[error_indices], columns=feature_names)
            error_data['true_label'] = y_true[error_indices]
            error_data['pred_label'] = y_pred[error_indices]
            
            # Save error samples
            error_data.to_csv(f'{save_dir}/error_analysis.csv', index=False)
            
            # Create error distribution plot
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            pd.Series(y_true[error_indices]).value_counts().plot(kind='bar')
            plt.title('True Labels of Misclassified Samples')
            plt.xlabel('True Label')
            plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            pd.Series(y_pred[error_indices]).value_counts().plot(kind='bar')
            plt.title('Predicted Labels of Misclassified Samples')
            plt.xlabel('Predicted Label')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    elif task_type == 'regression':
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        
        # Find samples with largest errors
        error_threshold = np.percentile(abs_errors, 90)  # Top 10% errors
        high_error_indices = np.where(abs_errors >= error_threshold)[0]
        
        if len(high_error_indices) > 0:
            # Analyze high-error samples
            error_data = pd.DataFrame(X_test[high_error_indices], columns=feature_names)
            error_data['true_value'] = y_true[high_error_indices]
            error_data['pred_value'] = y_pred[high_error_indices]
            error_data['residual'] = residuals[high_error_indices]
            error_data['abs_error'] = abs_errors[high_error_indices]
            
            # Save high-error samples
            error_data.to_csv(f'{save_dir}/high_error_analysis.csv', index=False)
            
            # Create error analysis plots
            create_residual_plots(y_true, y_pred, f'{save_dir}/residual_analysis.png')

def generate_comprehensive_report(results, eda_results, target_col, season, model_type='classification'):
    """Generate comprehensive evaluation report"""
    
    logger.info(f"Generating comprehensive report for {season} {model_type}")
    
    eval_dir = f'artifacts/evaluation/{season.lower()}'
    os.makedirs(eval_dir, exist_ok=True)
    
    report_path = f'{eval_dir}/comprehensive_report.md'
    
    with open(report_path, 'w') as file:
        file.write(f"# Comprehensive {season} {model_type.title()} Report\n\n")
        
        # Dataset summary
        file.write("## Dataset Summary\n\n")
        file.write(f"- **Shape**: {eda_results.get('data_shape', 'N/A')}\n")
        file.write(f"- **Target**: {target_col}\n")
        
        if 'target_distribution' in eda_results:
            file.write("- **Target Distribution**:\n")
            for category, count in eda_results['target_distribution'].items():
                file.write(f"  - {category}: {count}\n")
        
        # Model performance
        file.write("\n## Model Performance\n\n")
        
        if 'eval_summary' in results:
            # Create performance table
            df_results = pd.DataFrame(results['eval_summary'])
            
            if model_type == 'classification':
                df_results = df_results.sort_values('Eval F1 Macro', ascending=False)
                file.write("| Model | F1 Macro | Accuracy | Precision | Recall |\n")
                file.write("|-------|----------|----------|-----------|--------|\n")
                
                for _, row in df_results.iterrows():
                    file.write(f"| {row.get('Model', 'N/A')} | "
                             f"{row.get('Eval F1 Macro', 0):.4f} | "
                             f"{row.get('Eval Accuracy', 0):.4f} | "
                             f"{row.get('Eval Precision', 0):.4f} | "
                             f"{row.get('Eval Recall', 0):.4f} |\n")
                             
            elif model_type == 'regression':
                df_results = df_results.sort_values('R2', ascending=False)
                file.write("| Model | R² | RMSE | MAE | MAPE |\n")
                file.write("|-------|----|----- |----- |------|\n")
                
                for _, row in df_results.iterrows():
                    file.write(f"| {row.get('Model', 'N/A')} | "
                             f"{row.get('R2', 0):.4f} | "
                             f"{row.get('RMSE', 0):.4f} | "
                             f"{row.get('MAE', 0):.4f} | "
                             f"{row.get('MAPE', 0):.4f} |\n")
        
        # Best model details
        if 'best_model_name' in results:
            file.write(f"\n## Best Model: {results['best_model_name']}\n\n")
            
            if 'best_model_metrics' in results:
                metrics = results['best_model_metrics']
                file.write("### Detailed Metrics\n\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        file.write(f"- **{metric}**: {value:.4f}\n")
                    else:
                        file.write(f"- **{metric}**: {value}\n")
        
        # Data quality issues
        file.write("\n## Data Quality\n\n")
        if 'missing_values' in eda_results:
            missing_data = eda_results['missing_values']
            if any(count > 0 for count in missing_data.values()):
                file.write("### Missing Values\n\n")
                for col, count in missing_data.items():
                    if count > 0:
                        file.write(f"- **{col}**: {count} missing values\n")
            else:
                file.write("- No missing values detected\n")
        
        # Feature importance (if available)
        if 'feature_importance' in results:
            file.write("\n## Feature Importance\n\n")
            importance = results['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            file.write("| Feature | Importance |\n")
            file.write("|---------|------------|\n")
            for feature, imp in sorted_features[:10]:  # Top 10
                file.write(f"| {feature} | {imp:.4f} |\n")
        
        # Files generated
        file.write(f"\n## Generated Files\n\n")
        file.write(f"- **Models**: `artifacts/models/{season.lower()}/`\n")
        file.write(f"- **Plots**: `artifacts/evaluation/{season.lower()}/`\n")
        file.write(f"- **Data**: `artifacts/eda/{season.lower()}/`\n")
        
        file.write(f"\n## Recommendations\n\n")
        
        if model_type == 'classification':
            # Add classification-specific recommendations
            if 'eval_summary' in results:
                best_f1 = max([r.get('Eval F1 Macro', 0) for r in results['eval_summary']])
                if best_f1 < 0.7:
                    file.write("- **Model Performance**: Consider feature engineering or ensemble methods to improve F1 score\n")
                elif best_f1 > 0.95:
                    file.write("- **Model Performance**: Excellent performance, check for potential overfitting\n")
        
        elif model_type == 'regression':
            # Add regression-specific recommendations
            if 'eval_summary' in results:
                best_r2 = max([r.get('R2', 0) for r in results['eval_summary']])
                if best_r2 < 0.7:
                    file.write("- **Model Performance**: Consider advanced feature engineering or ensemble methods\n")
                elif best_r2 > 0.95:
                    file.write("- **Model Performance**: Excellent performance, validate on out-of-time data\n")
    
    logger.info(f"Comprehensive report saved to {report_path}")

def compare_seasonal_models_advanced():
    """Advanced comparison of seasonal models with statistical tests"""
    logger.info("Performing advanced seasonal model comparison")
    
    seasons = ['spring', 'summer', 'autumn', 'winter']
    all_results = []
    
    for season in seasons:
        results_file = f'artifacts/evaluation/{season}/model_comparison.csv'
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df['Season'] = season.title()
            all_results.append(df)
    
    if not all_results:
        logger.warning("No seasonal results found for comparison")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Statistical comparison
    from scipy import stats
    
    # Compare model performance across seasons
    comparison_results = {}
    models = combined_df['Model'].unique()
    
    for model in models:
        model_data = combined_df[combined_df['Model'] == model]
        if len(model_data) > 1:
            # ANOVA test for significant differences across seasons
            season_groups = [group['Eval F1 Macro'].values for name, group in model_data.groupby('Season')]
            if len(season_groups) > 1 and all(len(group) > 0 for group in season_groups):
                f_stat, p_value = stats.f_oneway(*season_groups)
                comparison_results[model] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    # Save advanced comparison
    os.makedirs('artifacts/evaluation/advanced', exist_ok=True)
    
    # Save statistical comparison results
    pd.DataFrame(comparison_results).T.to_csv('artifacts/evaluation/advanced/statistical_comparison.csv')
    
    # Create advanced visualization
    plt.figure(figsize=(15, 10))
    
    # Performance across seasons
    plt.subplot(2, 2, 1)
    sns.boxplot(data=combined_df, x='Season', y='Eval F1 Macro')
    plt.title('Model Performance Distribution by Season')
    plt.xticks(rotation=45)
    
    # Model comparison
    plt.subplot(2, 2, 2)
    model_mean_performance = combined_df.groupby('Model')['Eval F1 Macro'].mean().sort_values(ascending=True)
    model_mean_performance.plot(kind='barh')
    plt.title('Average Model Performance Across Seasons')
    plt.xlabel('Mean F1 Score')
    
    # Heatmap of performance
    plt.subplot(2, 1, 2)
    pivot_table = combined_df.pivot_table(values='Eval F1 Macro', index='Model', columns='Season', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', cbar_kws={'label': 'F1 Score'})
    plt.title('Model Performance Heatmap by Season')
    
    plt.tight_layout()
    plt.savefig('artifacts/evaluation/advanced/advanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Advanced seasonal comparison completed")
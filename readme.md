# Air Quality Prediction Pipeline - Enhanced Setup & Usage Guide

## Overview

This is a comprehensive, production-ready machine learning pipeline for air quality prediction with the following capabilities:

### ✅ **Core Features**
- **Dual Modeling**: Seasonal AQI classification + PM2.5 regression
- **Advanced Feature Engineering**: Interaction terms, temporal features, rolling windows
- **Hyperparameter Tuning**: Grid/Random search with cross-validation
- **Ensemble Methods**: Voting and stacking classifiers/regressors
- **Experiment Tracking**: MLflow integration for experiment management
- **Comprehensive Evaluation**: Extended metrics, error analysis, residual plots
- **Configuration Management**: YAML-based centralized configuration
- **Unit Testing**: Automated testing for all components
- **Data Quality Analysis**: Outlier detection and validation

## Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd air-quality-prediction

# Create virtual environment
conda create -n airquality python=3.9
conda activate airquality

# Install dependencies
pip install -r requirements_enhanced.txt
```

### 2. Basic Usage

```bash
# Run complete pipeline with defaults
python run_all_enhanced.py

# Run with hyperparameter tuning
python run_all_enhanced.py --tune-hyperparams

# Run with ensemble models
python run_all_enhanced.py --use-ensembles

# Run with experiment tracking
python run_all_enhanced.py --enable-tracking

# Run specific seasons only
python run_all_enhanced.py --seasons Spring Summer

# Skip certain components
python run_all_enhanced.py --skip-visualization --skip-classification
```

### 3. View Results

```bash
# View experiment tracking (if enabled)
mlflow ui

# Check artifacts
ls artifacts/
```

## Advanced Configuration

### Configuration File (`config.yaml`)

The pipeline is fully configurable through `config.yaml`:

```yaml
# Core settings
data:
  raw_file: "Airquality9.csv"
  target_regression: "PM2.5(µg/m³)"

# Enable advanced features
features:
  use_feature_engineering: true
  rolling_windows: [3, 6, 12]
  lag_features: [1, 2, 3]

# Model settings
training:
  hyperparameter_tuning:
    enabled: true
    method: "randomized"
    n_iter: 50
  
# Experiment tracking
experiment_tracking:
  enabled: true
  backend: "mlflow"
```

### Command Line Options

```bash
python run_all_enhanced.py --help
```

**Available Options:**
- `--config`: Configuration file path
- `--tune-hyperparams`: Enable hyperparameter tuning
- `--use-ensembles`: Enable ensemble models
- `--enable-tracking`: Enable experiment tracking
- `--skip-classification`: Skip seasonal classification
- `--skip-regression`: Skip PM2.5 regression
- `--skip-visualization`: Skip visualization generation
- `--seasons`: Specify which seasons to process
- `--verbose`: Enable detailed logging

## File Structure

```
air-quality-prediction/
├── config.yaml                    # Main configuration
├── run_all_enhanced.py            # Enhanced main pipeline
├── 
├── Core Modules:
├── load_data.py                   # Data loading and cleaning
├── feature_engineering.py        # Advanced feature creation
├── preprocess.py                  # Data preprocessing
├── models.py                      # Model training (enhanced)
├── evaluate_enhanced.py          # Comprehensive evaluation
├── predict.py                     # Prediction utilities
├── 
├── Enhancement Modules:
├── config_loader.py              # Configuration management
├── experiment_tracking.py        # MLflow integration
├── hyperparameter_tuning.py      # Grid/Random search
├── ensemble_models.py            # Voting/Stacking ensembles
├── utils.py                       # Utility functions
├── 
├── Analysis & Visualization:
├── eda.py                        # Exploratory data analysis
├── compare_model_predictions.py   # Model comparison
├── plot_sampled_predictions.py   # Visualization utilities
├── 
├── Testing:
├── tests/
│   └── test_pipeline.py          # Unit tests
├── 
├── Dependencies:
├── requirements_enhanced.txt      # All dependencies
└── setup-instructions.md         # This file
```

## Output Structure

```
artifacts/
├── data_cleaned.csv              # Processed dataset
├── models/                       # Trained models
│   ├── regression/              # PM2.5 regression models
│   ├── spring/                  # Spring classification models
│   ├── summer/                  # Summer classification models
│   ├── autumn/                  # Autumn classification models
│   └── winter/                  # Winter classification models
├── evaluation/                   # Evaluation results
│   ├── combined/                # Cross-seasonal comparisons
│   ├── advanced/                # Advanced statistical analysis
│   └── [season]/                # Per-season evaluations
├── eda/                         # Exploratory data analysis
└── plots/                       # Generated visualizations

mlruns/                          # MLflow experiment tracking
logs/                            # Pipeline logs (if enabled)
```

## Key Enhancements Explained

### 1. **Feature Engineering**
- **Interaction Features**: Ratios and products of pollutants
- **Temporal Features**: Cyclical encoding of time, day of week, seasonality
- **Rolling Windows**: Moving averages and standard deviations
- **Lag Features**: Previous time step values for temporal dependencies

### 2. **Hyperparameter Tuning**
- **Grid Search**: Exhaustive search over parameter grids
- **Random Search**: Efficient random sampling for large parameter spaces
- **Cross-Validation**: Robust model selection with k-fold CV
- **Early Stopping**: Prevents overfitting in boosting algorithms

### 3. **Ensemble Methods**
- **Voting Classifiers**: Combine multiple models via majority voting
- **Voting Regressors**: Average predictions from multiple models
- **Stacking**: Use meta-learner to combine base model predictions
- **Automatic Selection**: Choose best ensemble method via CV

### 4. **Experiment Tracking**
- **MLflow Integration**: Track experiments, parameters, metrics
- **Model Versioning**: Automatic model versioning and comparison
- **Artifact Storage**: Store models, plots, and data
- **Web UI**: View and compare experiments via `mlflow ui`

### 5. **Advanced Evaluation**
- **Extended Metrics**: MAE, RMSE, MAPE for regression; precision, recall for classification
- **Error Analysis**: Identify patterns in model errors
- **Residual Analysis**: Comprehensive residual plots and statistics
- **Statistical Comparison**: ANOVA tests across seasons/models

### 6. **Configuration Management**
- **YAML Configuration**: Centralized parameter management
- **Environment-Specific**: Different configs for dev/prod
- **Dynamic Loading**: Runtime configuration updates
- **Validation**: Automatic config validation and defaults

### 7. **Data Quality Analysis**
- **Outlier Detection**: Isolation Forest for anomaly detection
- **Missing Value Analysis**: Comprehensive missing data reports
- **Data Drift Monitoring**: Track changes in data distribution
- **Quality Metrics**: Automated data quality scoring

## Usage Examples

### Example 1: Quick Development Run
```bash
# Fast run for development/testing
python run_all_enhanced.py --seasons Spring --skip-visualization
```

### Example 2: Full Production Run
```bash
# Complete production run with all features
python run_all_enhanced.py \
  --tune-hyperparams \
  --use-ensembles \
  --enable-tracking \
  --config production_config.yaml
```

### Example 3: Experiment Comparison
```bash
# Run multiple experiments
python run_all_enhanced.py --enable-tracking --config experiment1.yaml
python run_all_enhanced.py --enable-tracking --config experiment2.yaml

# View results
mlflow ui
```

### Example 4: Making Predictions
```bash
# Classification predictions
python predict.py new_data.csv --type classification --season spring --model RandomForest

# Regression predictions
python predict.py new_data.csv --type regression --model LightGBM
```

## Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_pipeline.py -v
```

### Adding New Tests
Create test files in `tests/` following the pattern:
```python
import pytest
from your_module import your_function

class TestYourModule:
    def test_your_function(self):
        assert your_function(input) == expected_output
```

## Performance Optimization

### Memory Optimization
```yaml
# In config.yaml
advanced:
  memory_efficient: true
  parallel_training: true
  save_intermediate: false
```

### Speed Optimization
```yaml
training:
  hyperparameter_tuning:
    method: "randomized"  # Faster than grid search
    n_iter: 20           # Fewer iterations
    n_jobs: -1           # Use all CPU cores
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

2. **Memory Issues**
   - Reduce sample sizes in config
   - Enable memory-efficient mode
   - Close other applications

3. **MLflow Issues**
   ```bash
   # Reset MLflow
   rm -rf mlruns/
   python run_all_enhanced.py --enable-tracking
   ```

4. **Configuration Errors**
   ```bash
   # Validate config
   python -c "from config_loader import Config; Config('config.yaml')"
   ```

### Getting Help

1. Check logs in `logs/pipeline.log`
2. Run with `--verbose` for detailed output
3. Check MLflow UI for experiment details
4. Review test results for component issues

## Extension Guide

### Adding New Models

1. **Add to `models.py`**:
```python
models['NewModel'] = YourNewModel(parameters...)
```

2. **Add to hyperparameter grids**:
```python
# In hyperparameter_tuning.py
param_grids['NewModel'] = {
    'param1': [value1, value2],
    'param2': [value3, value4]
}
```

### Adding New Features

1. **Extend `feature_engineering.py`**:
```python
def create_new_features(df):
    df['new_feature'] = some_calculation(df)
    return df
```

2. **Update configuration**:
```yaml
features:
  use_new_features: true
```

### Adding New Metrics

1. **Extend `evaluate_enhanced.py`**:
```python
def calculate_new_metric(y_true, y_pred):
    return your_metric_calculation
```

## Best Practices

### Development Workflow
1. Start with small dataset for development
2. Use `--verbose` for debugging
3. Test individual components before full pipeline
4. Use version control for configurations

### Production Deployment
1. Use configuration files for environment-specific settings
2. Enable experiment tracking for reproducibility
3. Set up automated testing in CI/CD
4. Monitor data quality in production

### Model Management
1. Use MLflow for model versioning
2. Document model performance and business impact
3. Set up automated retraining schedules
4. Monitor for model drift

## Support and Maintenance

### Regular Maintenance
- Update dependencies regularly
- Monitor model performance
- Review and update configurations
- Clean up old experiment data

### Performance Monitoring
- Track prediction accuracy over time
- Monitor data quality metrics
- Set up alerts for anomalies
- Regular model retraining

This enhanced pipeline provides a solid foundation for air quality prediction with room for further customization and scaling.

python run_all_enhanced.py --tune-hyperparams --use-ensembles --enable-tracking

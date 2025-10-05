# Air Quality Prediction Pipeline

## Overview

This repository offers a robust, modular, and fully automated Python pipeline for multi-season air quality forecasting and benchmarking using modern machine learning. Built for reproducibility and extensibility, it provides everything you need for end-to-end data science—from raw data ingestion through engineering, model training, visual benchmarking, and reporting—all in one place.

## Features

- **Multi-season Stratification:** Models trained and evaluated separately for Spring, Summer, Autumn, Winter.
- **Flexible Modeling:** Supports classification (AQI category) and regression (PM2.5).
- **Model Zoo:** Includes LightGBM, XGBoost, RandomForest, SVM, MLP, Logistic/Linear Regression, Decision Tree.
- **Feature Engineering:** Rolling means, lags, temporal encodings, interaction features, meteorological indexes.
- **Automated EDA:** Generates seasonal plots—category distributions, feature violins, temporal trends, PM2.5 histograms (correlation matrix omitted per config).
- **Data Quality Checks:** Missingness warnings, outlier detection, automatic outlier reporting.
- **Visual Benchmarking:** Plots include:
  - Model performance by season (boxplots)
  - Mean model performance (bar plots)
  - Advanced heatmap (model vs. season, see below)
  - Prediction scatter/comparison curves (regression)
- **Artifact Management & Logging:** All data, plots, and models saved under `artifacts/` with detailed workflow logs.
  
## Example Results

![Advanced Comparison](



## Directory Structure

```
├── Airquality9.csv                  # Example dataset (replace with your own for production)
├── run_all_enhanced.py              # Main pipeline entrypoint
├── load_data.py | feature_engineering.py | preprocess.py | models.py | eda.py | evaluate_enhanced.py
├── artifacts/
│   ├── data_cleaned.csv             # Cleaned, feature-engineered data
│   ├── eda/                         # EDA plots (per season)
│   ├── models/
│   │   ├── [season]/                # Models/artifacts for each season
│   │   └── regression/              # Regression models and plots
│   ├── evaluation/                  # Comparison reports, advanced comparison visuals
│   └── data_quality/                # Outlier analysis, quality reports
├── requirements.txt                 # All dependencies listed
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/air-quality-pipeline.git
cd air-quality-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Usage

**To run the full pipeline:**
```bash
python run_all_enhanced.py --verbose
```
Options:
- `--tune-hyperparams` for hyperparameter search
- `--use-ensembles` for ensemble benchmarking
- `--enable-tracking` for MLflow experiment tracking
- `--skip-classification` or `--skip-regression` to bypass parts of the workflow

### 3. Output & Artifacts

- All plots are saved in `artifacts/eda/{season}/` (EDA), `artifacts/models/regression/` (regression scatter/comparison), and `artifacts/evaluation/advanced/` (heatmaps and summary visuals).
- Cleaned data and feature logs in `artifacts/data_cleaned.csv`.
- Outliers saved at `artifacts/data_quality/outlier_analysis.csv`.

### 4. Customization

- **Configure** features, models, and workflow via `config.yaml`.
- **Plug and play** new models by editing `models.py` and updating `requirements.txt` as needed.
- **Update feature engineering** logic in `feature_engineering.py`.

## Technologies

- **Python 3.12+**
- **scikit-learn, pandas, numpy**
- **xgboost, lightgbm**
- **matplotlib, seaborn**
- **joblib**
- Optional: TensorFlow, PyTorch, MLflow (for advanced/experimental setups)

## Citing & Acknowledgement

If you use or adapt this pipeline for research or production, please consider citing or referencing this repository in your publications or documentation.

## Contributors

- [Your name(s) here]

## License

[MIT] or [Specify your license]

***

*For questions, support, or improvements: please open an Issue or Pull Request on GitHub.*

***

**This project equips you to benchmark, optimize, and visualize air quality prediction with confidence and clarity.**

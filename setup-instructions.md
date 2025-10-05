# Air Quality Index ML Pipeline - Setup Instructions

## 📁 File Structure

Your project should have the following structure:
```
project_folder/
├── project_details.docx     # Your project requirements (provided)
├── Airquality9.csv         # Your dataset (provided)
├── requirements.txt        # Python dependencies
├── run_all.py             # Main pipeline runner
├── load_data.py           # Data loading and preparation
├── eda.py                 # Exploratory data analysis  
├── preprocess.py          # Data preprocessing
├── models.py              # Model training and evaluation
├── evaluate.py            # Advanced evaluation and reporting
├── utils.py               # Utility functions
├── predict.py             # Standalone prediction script
└── artifacts/             # Generated outputs (created automatically)
    ├── eda/               # EDA plots and visualizations
    ├── models/            # Trained models and preprocessors
    ├── evaluation/        # Model evaluation results
    ├── data_cleaned.csv   # Processed dataset
    ├── report.md          # Final analysis report
    └── pipeline.log       # Execution logs
```

## 🚀 Quick Start (For Spyder)

### Step 1: Install Dependencies
Open Spyder console and run:
```python
!pip install -r requirements.txt
```

### Step 2: Set Working Directory
In Spyder:
1. Go to the folder containing all your .py files
2. Right-click in the file explorer → "Set as current directory"
3. Or use: `os.chdir(r'path/to/your/project/folder')`

### Step 3: Run the Pipeline
Open `run_all.py` in Spyder and press F5, or run:
```python
exec(open('run_all.py').read())
```

## 📋 What Each Module Does

- **`run_all.py`**: Orchestrates the entire pipeline from start to finish
- **`load_data.py`**: Loads CSV data, creates AQI target variable, handles datetime features
- **`eda.py`**: Comprehensive exploratory data analysis with visualizations  
- **`preprocess.py`**: Data cleaning, encoding, scaling, train/test split
- **`models.py`**: Trains 15+ ML models with hyperparameter tuning and cross-validation
- **`evaluate.py`**: Advanced model evaluation, feature importance, SHAP analysis
- **`utils.py`**: Helper functions, logging, directory management
- **`predict.py`**: Standalone script for making predictions on new data

## 🎯 Expected Outputs

After successful execution, you'll have:

### 📊 Visualizations (artifacts/eda/)
- Target variable distribution plots
- Numeric feature distributions and correlations
- Categorical feature analysis
- Missing value patterns
- Feature-target relationships

### 🤖 Models (artifacts/models/)
- `best_model.joblib` - Top performing model
- `preprocessor.joblib` - Data preprocessing pipeline
- `all_results.joblib` - Complete results from all models

### 📈 Evaluation (artifacts/evaluation/)
- `model_comparison.csv` - Detailed performance metrics for all models
- `model_comparison.png` - Performance visualization
- `confusion_matrices.png` - Confusion matrices for top models
- `feature_importance.png` - Feature importance plots
- `classification_report.csv` - Detailed per-class metrics

### 📝 Reports
- `artifacts/report.md` - Comprehensive analysis report
- `artifacts/pipeline.log` - Execution logs with timing
- `artifacts/data_cleaned.csv` - Cleaned dataset

## 🔍 Key Features

### Automatic Problem Detection
The pipeline automatically detects this is a **classification problem** by:
- Reading project requirements from project_details.docx
- Calculating AQI values using Indian AQI standards
- Creating 6-class target: Good, Satisfactory, Moderate, Poor, Very Poor, Severe

### Comprehensive ML Pipeline
- **15+ Models**: Baselines, Linear, Tree-based, SVM, Neural Networks, Ensembles
- **Hyperparameter Tuning**: RandomizedSearchCV with 20 iterations per model
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Class Imbalance Handling**: SMOTE oversampling for imbalanced target
- **Feature Engineering**: Datetime decomposition, categorical encoding, scaling

### Advanced Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Feature Importance**: Tree-based importance, permutation importance
- **Model Interpretability**: SHAP analysis for top models
- **Comprehensive Reporting**: Automated markdown reports

## ⚡ Performance Notes

- **Execution Time**: ~5-15 minutes depending on system
- **Memory Usage**: ~500MB-1GB RAM 
- **Parallel Processing**: Uses all CPU cores for model training
- **Progress Logging**: Real-time progress updates in console

## 🛠️ Troubleshooting

### Common Issues:
1. **Import Error**: Ensure all .py files are in the same directory
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **File Not Found**: Check that `Airquality9.csv` and `project_details.docx` are present
4. **Memory Error**: Reduce dataset size if over 100k samples

### If Pipeline Fails:
1. Check `artifacts/pipeline.log` for detailed error messages
2. Run individual modules separately to isolate issues
3. Verify working directory is set correctly in Spyder

## 📊 Making Predictions on New Data

After training, use the prediction script:
```python
from predict import predict_aqi
predictions = predict_aqi('new_data.csv')
```

The new data CSV should have the same columns as the training data (excluding the target).

## 🎯 Expected Results

Based on the project requirements and data analysis:
- **Best Model**: Likely Random Forest or XGBoost
- **Expected Accuracy**: 85-95% (depends on data quality)
- **Key Features**: PM2.5, PM10, Temperature, Humidity most important
- **Challenge**: Class imbalance (few Severe/Very Poor samples)

---
**Ready to run! Open Spyder, set your working directory, and execute run_all.py** 🚀
import os
import time
import sys
import warnings
import logging
import argparse
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import project modules
try:
    from load_data import load_and_prepare_data
    from feature_engineering import create_advanced_features
    from eda import perform_eda
    from preprocess import preprocess_data, preprocess_for_regression
    from models import train_and_evaluate_classification_models, train_and_evaluate_regression_models
    from evaluate_enhanced import generate_comprehensive_report, compare_seasonal_models_advanced
    from utils import setup_logging, create_directory_structure, validate_data_quality
    from plot_sampled_predictions import plot_regression_predictions_sample
    from compare_model_predictions import compare_all_models
    from config_loader import get_config
    from experiment_tracking import init_experiment_tracking, get_experiment_tracker
    from hyperparameter_tuning import tune_classification_models, tune_regression_models
    from ensemble_models import create_ensemble_models
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the current directory")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Air Quality Prediction Pipeline')
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--use-ensembles', action='store_true', help='Enable ensemble models')
    parser.add_argument('--enable-tracking', action='store_true', help='Enable experiment tracking')
    parser.add_argument('--skip-classification', action='store_true', help='Skip classification pipeline')
    parser.add_argument('--skip-regression', action='store_true', help='Skip regression pipeline')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization pipeline')
    parser.add_argument('--seasons', nargs='+', choices=['Spring', 'Summer', 'Autumn', 'Winter'], 
                       default=['Spring', 'Summer', 'Autumn', 'Winter'], help='Seasons to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def run_classification_pipeline(df, target_col, config, args):
    """Run the seasonal classification pipeline with enhancements"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Enhanced Classification Pipeline ===")
    
    results_all = {}
    experiment_tracker = get_experiment_tracker() if args.enable_tracking else None
    
    for season in args.seasons:
        logger.info(f"\n--- Processing {season} ---")
        
        # Filter data for season
        df_season = df[df['Season'] == season]
        
        if len(df_season) < config.get('training.min_samples_per_season', 100):
            logger.warning(f"Insufficient data for {season}: {len(df_season)} samples. Skipping.")
            continue
        
        logger.info(f"{season} data shape: {df_season.shape}")
        
        # Perform enhanced EDA
        try:
            eda_results = perform_eda(df_season, target_col, season)
        except Exception as e:
            logger.error(f"EDA failed for {season}: {e}")
            eda_results = {'data_shape': df_season.shape, 'missing_values': {}, 'target_distribution': {}}
        
        # Preprocess data
        try:
            (X_train, X_eval, X_pred, y_train, y_eval, y_pred, 
             preprocessor, feature_names) = preprocess_data(df_season, target_col, season)
        except Exception as e:
            logger.error(f"Preprocessing failed for {season}: {e}")
            continue
        
        # Train and evaluate models
        try:
            # Start experiment tracking for this season
            if experiment_tracker:
                experiment_tracker.start_run(f"classification_{season.lower()}", 
                                            tags={'season': season, 'task': 'classification'})
            
            results = train_and_evaluate_classification_models(
                X_train, X_eval, y_train, y_eval, X_pred, y_pred,
                feature_names, preprocessor, target_col, season,
                tune_hyperparams=args.tune_hyperparams,
                use_ensembles=args.use_ensembles,
                experiment_tracker=experiment_tracker
            )
            results_all[season] = results
            
            # End experiment tracking
            if experiment_tracker:
                experiment_tracker.end_run()
                
        except Exception as e:
            logger.error(f"Model training failed for {season}: {e}")
            if experiment_tracker:
                experiment_tracker.end_run()
            continue
        
        # Generate enhanced report
        try:
            generate_comprehensive_report(results, eda_results, target_col, season, 'classification')
        except Exception as e:
            logger.error(f"Report generation failed for {season}: {e}")
    
    logger.info(f"Classification pipeline completed for {len(results_all)} seasons")
    return results_all

def run_regression_pipeline(df, config, args):
    """Run the PM2.5 regression pipeline with enhancements"""
    
    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Enhanced Regression Pipeline ===")
    
    experiment_tracker = get_experiment_tracker() if args.enable_tracking else None
    
    try:
        # Preprocess for regression
        (X_train, X_test, y_train, y_test, 
         preprocessor, feature_names) = preprocess_for_regression(df, config.get('data.target_regression'))
        
        # Start experiment tracking
        if experiment_tracker:
            experiment_tracker.start_run("pm25_regression", tags={'task': 'regression'})
        
        # Train regression models
        model_dir = config.get('paths.models_dir') + '/regression'
        results = train_and_evaluate_regression_models(
            X_train, X_test, y_train, y_test, model_dir,
            tune_hyperparams=args.tune_hyperparams,
            use_ensembles=args.use_ensembles,
            experiment_tracker=experiment_tracker
        )
        
        # Generate enhanced regression report
        from evaluate_enhanced import generate_comprehensive_report
        eda_results = {'data_shape': df.shape, 'missing_values': {}, 'target_distribution': {}}
        generate_comprehensive_report(results, eda_results, config.get('data.target_regression'), 
                                    'Regression', 'regression')
        
        # End experiment tracking
        if experiment_tracker:
            experiment_tracker.end_run()
        
        logger.info("Enhanced regression pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Regression pipeline failed: {e}")
        if experiment_tracker:
            experiment_tracker.end_run()
        return {}

def run_visualization_pipeline(config, args):
    """Run enhanced visualization and comparison scripts"""
    
    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Enhanced Visualization Pipeline ===")
    
    try:
        # Generate regression prediction plots
        plot_regression_predictions_sample(sample_size=config.get('plotting.sample_size', 100))
        
        # Compare all models
        compare_all_models()
        
        # Advanced seasonal model comparison
        compare_seasonal_models_advanced()
        
        logger.info("Enhanced visualization pipeline completed")
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {e}")

# Fixed run_data_quality_analysis function

def run_data_quality_analysis(df, config):
    """Run comprehensive data quality analysis"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Running Data Quality Analysis ===")
    
    try:
        # CREATE MISSING DIRECTORY
        os.makedirs('artifacts/data_quality', exist_ok=True)
        
        # Validate data quality
        critical_columns = [
            config.get('data.target_regression'),
            config.get('data.target_classification'),
            'Season', 'Date'
        ]
        issues = validate_data_quality(df, critical_columns)
        
        # Advanced outlier detection
        if config.get('advanced.outlier_detection', True):
            from sklearn.ensemble import IsolationForest
            
            # Detect outliers in PM2.5
            pm25_col = config.get('data.target_regression')
            if pm25_col in df.columns:
                pm25_data = df[pm25_col].dropna().values.reshape(-1, 1)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(pm25_data)
                
                n_outliers = (outliers == -1).sum()
                logger.info(f"Detected {n_outliers} outliers in {pm25_col}")
                
                # Save outlier analysis - DIRECTORY NOW EXISTS
                outlier_df = df[df[pm25_col].notna()].copy()
                outlier_df['is_outlier'] = outliers == -1
                outlier_df.to_csv('artifacts/data_quality/outlier_analysis.csv', index=False)
        
        return issues
        
    except Exception as e:
        logger.error(f"Data quality analysis failed: {e}")
        return []

def main():
    """Main pipeline execution with all enhancements"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    from config_loader import reload_config
    config = reload_config(args.config)
    
    # Setup logging from config
    config.setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    
    # Change to script directory
    os.chdir(script_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Create directory structure from config
    config.create_directories()
    
    # Initialize experiment tracking if enabled
    if args.enable_tracking:
        experiment_name = config.get('experiment_tracking.experiment_name', 'air_quality_prediction')
        init_experiment_tracking(experiment_name)
        logger.info("Experiment tracking initialized")
    
    start_time = time.time()
    logger.info(f"Enhanced pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load and prepare data
        logger.info("=== Loading and Preparing Data ===")
        df, target_col = load_and_prepare_data()
        
        # Apply feature engineering if enabled
        if config.get('features.use_feature_engineering', True):
            logger.info("Applying feature engineering...")
            df = create_advanced_features(df)
        
        # Run data quality analysis
        data_quality_issues = run_data_quality_analysis(df, config)
        if data_quality_issues:
            logger.warning("Data quality issues detected but continuing...")
        
        # Run classification pipeline
        classification_results = {}
        if not args.skip_classification:
            classification_results = run_classification_pipeline(df, target_col, config, args)
        
        # Run regression pipeline
        regression_results = {}
        if not args.skip_regression:
            regression_results = run_regression_pipeline(df, config, args)
        
        # Run visualization pipeline
        if not args.skip_visualization:
            run_visualization_pipeline(config, args)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n=== Enhanced Pipeline Summary ===")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Configuration used: {args.config}")
        logger.info(f"Classification models trained for: {list(classification_results.keys())}")
        logger.info(f"Regression models trained: {len(regression_results)} models")
        logger.info(f"Hyperparameter tuning: {'Enabled' if args.tune_hyperparams else 'Disabled'}")
        logger.info(f"Ensemble models: {'Enabled' if args.use_ensembles else 'Disabled'}")
        logger.info(f"Experiment tracking: {'Enabled' if args.enable_tracking else 'Disabled'}")
        logger.info(f"Artifacts saved to: {os.path.abspath(config.get('paths.artifacts_dir'))}")
        
        print(f"\n" + "="*70)
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results saved in: {os.path.abspath(config.get('paths.artifacts_dir'))}")
        
        if args.enable_tracking:
            print("View experiment results with: mlflow ui")
        
        print("="*70)
        
        # Run tests if requested
        if config.get('advanced.run_tests', False):
            logger.info("Running unit tests...")
            import subprocess
            try:
                result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("All tests passed!")
                else:
                    logger.warning("Some tests failed. Check test output.")
            except Exception as e:
                logger.error(f"Failed to run tests: {e}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
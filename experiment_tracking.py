import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """MLflow experiment tracking wrapper"""
    
    def __init__(self, experiment_name="air_quality_prediction", tracking_uri="mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment '{self.experiment_name}' setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            self.enabled = False
        else:
            self.enabled = True
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run"""
        if not self.enabled:
            return None
            
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if tags is None:
            tags = {}
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name}")
        return run
    
    def log_params(self, params):
        """Log parameters to MLflow"""
        if not self.enabled:
            return
            
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        if not self.enabled:
            return
            
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model(self, model, model_name, signature=None, input_example=None):
        """Log model to MLflow"""
        if not self.enabled:
            return
            
        try:
            # Determine model type and log accordingly
            model_type = type(model).__name__
            
            if 'XGB' in model_type:
                mlflow.xgboost.log_model(model, model_name, signature=signature, input_example=input_example)
            elif 'LGBM' in model_type or 'LightGBM' in model_type:
                mlflow.lightgbm.log_model(model, model_name, signature=signature, input_example=input_example)
            else:
                mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)
            
            logger.info(f"Model '{model_name}' logged to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log artifact to MLflow"""
        if not self.enabled:
            return
            
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
                logger.info(f"Artifact '{local_path}' logged to MLflow")
            else:
                logger.warning(f"Artifact '{local_path}' does not exist")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_dataframe(self, df, name):
        """Log dataframe as CSV artifact"""
        if not self.enabled:
            return
            
        try:
            temp_path = f"temp_{name}.csv"
            df.to_csv(temp_path, index=False)
            self.log_artifact(temp_path, f"data/{name}.csv")
            os.remove(temp_path)  # Clean up
        except Exception as e:
            logger.error(f"Failed to log dataframe: {e}")
    
    def end_run(self):
        """End current MLflow run"""
        if not self.enabled:
            return
            
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def log_classification_run(self, model, model_name, X_train, X_test, y_train, y_test, 
                             metrics, params=None, season=None):
        """Log complete classification run"""
        if not self.enabled:
            return
            
        tags = {
            'model_type': 'classification',
            'model_name': model_name,
            'task': 'seasonal_aqi_prediction'
        }
        
        if season:
            tags['season'] = season
        
        with mlflow.start_run(tags=tags) as run:
            # Log parameters
            if params:
                self.log_params(params)
            
            # Log dataset info
            self.log_params({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'n_classes': len(np.unique(y_train))
            })
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            try:
                # Create input example
                input_example = X_test[:5] if hasattr(X_test, '__getitem__') else None
                self.log_model(model, model_name, input_example=input_example)
            except Exception as e:
                logger.error(f"Failed to log model in classification run: {e}")
            
            return run.info.run_id
    
    def log_regression_run(self, model, model_name, X_train, X_test, y_train, y_test, 
                          metrics, params=None):
        """Log complete regression run"""
        if not self.enabled:
            return
            
        tags = {
            'model_type': 'regression',
            'model_name': model_name,
            'task': 'pm25_prediction'
        }
        
        with mlflow.start_run(tags=tags) as run:
            # Log parameters
            if params:
                self.log_params(params)
            
            # Log dataset info
            self.log_params({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'target_mean': np.mean(y_train),
                'target_std': np.std(y_train)
            })
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            try:
                # Create input example
                input_example = X_test[:5] if hasattr(X_test, '__getitem__') else None
                self.log_model(model, model_name, input_example=input_example)
            except Exception as e:
                logger.error(f"Failed to log model in regression run: {e}")
            
            return run.info.run_id
    
    def compare_runs(self, run_ids, metric_name):
        """Compare multiple runs by a specific metric"""
        if not self.enabled:
            return None
            
        try:
            runs_data = []
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                metric_value = run.data.metrics.get(metric_name, None)
                runs_data.append({
                    'run_id': run_id,
                    'run_name': run.info.run_name,
                    metric_name: metric_value,
                    'model_name': run.data.tags.get('model_name', 'Unknown')
                })
            
            comparison_df = pd.DataFrame(runs_data)
            comparison_df = comparison_df.sort_values(metric_name, ascending=False)
            
            logger.info(f"Run comparison by {metric_name}:")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return None

# Global tracker instance
_tracker = None

def get_experiment_tracker():
    """Get global experiment tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker

def init_experiment_tracking(experiment_name="air_quality_prediction", tracking_uri="mlruns"):
    """Initialize experiment tracking"""
    global _tracker
    _tracker = ExperimentTracker(experiment_name, tracking_uri)
    return _tracker
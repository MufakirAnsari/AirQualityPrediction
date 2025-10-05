import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)

def get_classification_param_grids():
    """Return parameter grids for classification models"""
    return {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        },
        'DecisionTree': {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }

def get_regression_param_grids():
    """Return parameter grids for regression models"""
    return {
        'LinearRegression': {},  # No hyperparameters to tune
        'DecisionTree': {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }

def tune_classification_models(X_train, y_train, model_dict, cv=3, n_jobs=-1):
    """
    Tune hyperparameters for classification models
    """
    logger.info("Starting hyperparameter tuning for classification models...")
    
    param_grids = get_classification_param_grids()
    tuned_models = {}
    best_params = {}
    
    for name, base_model in model_dict.items():
        if name not in param_grids:
            logger.warning(f"No parameter grid found for {name}, using default parameters")
            tuned_models[name] = base_model
            continue
            
        param_grid = param_grids[name]
        
        if len(param_grid) == 0:  # No parameters to tune
            tuned_models[name] = base_model
            continue
        
        logger.info(f"Tuning {name}...")
        
        try:
            # Use RandomizedSearchCV for faster tuning on large parameter spaces
            if len(param_grid) > 4:
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=cv, n_jobs=n_jobs,
                    scoring='f1_macro', n_iter=20, random_state=42
                )
            else:
                search = GridSearchCV(
                    base_model, param_grid, cv=cv, n_jobs=n_jobs,
                    scoring='f1_macro'
                )
            
            search.fit(X_train, y_train)
            
            tuned_models[name] = search.best_estimator_
            best_params[name] = search.best_params_
            
            logger.info(f"{name} best score: {search.best_score_:.4f}")
            logger.info(f"{name} best params: {search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error tuning {name}: {e}")
            tuned_models[name] = base_model
    
    return tuned_models, best_params

def tune_regression_models(X_train, y_train, model_dict, cv=3, n_jobs=-1):
    """
    Tune hyperparameters for regression models
    """
    logger.info("Starting hyperparameter tuning for regression models...")
    
    param_grids = get_regression_param_grids()
    tuned_models = {}
    best_params = {}
    
    for name, base_model in model_dict.items():
        if name not in param_grids:
            logger.warning(f"No parameter grid found for {name}, using default parameters")
            tuned_models[name] = base_model
            continue
            
        param_grid = param_grids[name]
        
        if len(param_grid) == 0:  # No parameters to tune
            tuned_models[name] = base_model
            continue
        
        logger.info(f"Tuning {name}...")
        
        try:
            # Use RandomizedSearchCV for faster tuning on large parameter spaces
            if len(param_grid) > 4:
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=cv, n_jobs=n_jobs,
                    scoring='r2', n_iter=20, random_state=42
                )
            else:
                search = GridSearchCV(
                    base_model, param_grid, cv=cv, n_jobs=n_jobs,
                    scoring='r2'
                )
            
            search.fit(X_train, y_train)
            
            tuned_models[name] = search.best_estimator_
            best_params[name] = search.best_params_
            
            logger.info(f"{name} best score: {search.best_score_:.4f}")
            logger.info(f"{name} best params: {search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error tuning {name}: {e}")
            tuned_models[name] = base_model
    
    return tuned_models, best_params
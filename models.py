import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate_classification_models(
    X_train, X_eval, y_train, y_eval, X_pred, y_pred,
    feature_names, preprocessor, target_col, season,
    tune_hyperparams=False, use_ensembles=False, experiment_tracker=None):

    logger.info(f"Training classification models for {season}")

    model_dir = f'artifacts/models/{season.lower()}'
    eval_dir = f'artifacts/evaluation/{season.lower()}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_eval_enc = le.transform(y_eval)
    y_pred_enc = le.transform(y_pred)

    joblib.dump(le, f'{model_dir}/label_encoder.joblib')

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            num_leaves=31,
            min_child_samples=20,
            min_data_in_leaf=10,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1
        ),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=500, random_state=42)
    }

    if tune_hyperparams:
        try:
            from hyperparameter_tuning import tune_classification_models
            tuned_models, best_params = tune_classification_models(X_train, y_train_enc, models)
            models.update(tuned_models)
            joblib.dump(best_params, f'{model_dir}/best_hyperparams.joblib')
            logger.info("Hyperparameter tuning completed")
        except ImportError:
            logger.warning("Hyperparameter tuning module not found, using default parameters")

    if use_ensembles:
        try:
            from ensemble_models import create_ensemble_models
            ensemble_models = create_ensemble_models(models, X_train, y_train_enc, 'classification')
            models.update(ensemble_models)
            logger.info("Ensemble models created")
        except ImportError:
            logger.warning("Ensemble models module not found, skipping ensembles")

    eval_results = []
    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        logger.info(f"Training {name}...")

        try:
            model.fit(X_train, y_train_enc)
            y_eval_pred = model.predict(X_eval)

            f1 = f1_score(y_eval_enc, y_eval_pred, average='macro')
            acc = accuracy_score(y_eval_enc, y_eval_pred)
            prec = precision_score(y_eval_enc, y_eval_pred, average='macro', zero_division=0)
            rec = recall_score(y_eval_enc, y_eval_pred, average='macro', zero_division=0)

            eval_results.append({
                'Model': name,
                'Eval F1 Macro': f1,
                'Eval Accuracy': acc,
                'Eval Precision': prec,
                'Eval Recall': rec
            })

            joblib.dump(model, f'{model_dir}/{name}.joblib')

            if experiment_tracker:
                experiment_tracker.log_params({'model_name': name, 'season': season})
                experiment_tracker.log_metrics({'f1_macro': f1, 'accuracy': acc, 'precision': prec, 'recall': rec})

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

            logger.info(f"{name} - F1: {f1:.4f}, Accuracy: {acc:.4f}")

        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            continue

    results_df = pd.DataFrame(eval_results).sort_values('Eval F1 Macro', ascending=False)
    results_df.to_csv(f'{eval_dir}/model_comparison.csv', index=False)

    logger.info(f"Best model for {season}: {best_name} with F1 {best_f1:.4f}")

    return {
        'best_model': best_model,
        'best_name': best_name,
        'eval_results': eval_results,
        'label_encoder': le
    }


def train_and_evaluate_regression_models(
    X_train, X_test, y_train, y_test,
    model_dir,
    tune_hyperparams=False,
    use_ensembles=False,
    experiment_tracker=None):

    logger.info("Training regression models")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(model_dir), 'evaluation'), exist_ok=True)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
        "LightGBM": lgb.LGBMRegressor(
            random_state=42,
            verbose=-1,
            num_leaves=31,
            min_child_samples=20,
            min_data_in_leaf=10,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1
        ),
        "SVM": SVR(),
        "MLP": MLPRegressor(max_iter=1000, random_state=42)
    }

    if tune_hyperparams:
        try:
            from hyperparameter_tuning import tune_regression_models
            tuned_models, best_params = tune_regression_models(X_train, y_train, models)
            models.update(tuned_models)
            joblib.dump(best_params, f'{model_dir}/best_hyperparams.joblib')
            logger.info("Hyperparameter tuning completed")
        except ImportError:
            logger.warning("Hyperparameter tuning module not found, using default parameters")

    if use_ensembles:
        try:
            from ensemble_models import create_ensemble_models
            ensemble_models = create_ensemble_models(models, X_train, y_train, 'regression')
            models.update(ensemble_models)
            logger.info("Ensemble models created")
        except ImportError:
            logger.warning("Ensemble models module not found, skipping ensembles")

    eval_results = []
    best_model = None
    best_name = None
    best_r2 = float('-inf')
    predictions = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))

            eval_results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
            predictions[name] = y_pred

            joblib.dump(model, f'{model_dir}/{name}.joblib')

            if experiment_tracker:
                experiment_tracker.log_params({'model_name': name})
                experiment_tracker.log_metrics({'r2': r2, 'rmse': rmse, 'mae': mae})

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

            logger.info(f"{name} - R2: {r2:.4f}, RMSE: {rmse:.4f}")

        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            continue

    # Create comparison plot
    n_models = len(predictions)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = [axes]
    axes_flat = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        ax.scatter(y_test, y_pred, alpha=0.6)
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name} (R2={eval_results[idx]["R2"]:.3f})')
        ax.grid(True)

    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{model_dir}/regression_comparison.png')
    plt.close()

    # Save evaluation results
    results_df = pd.DataFrame(eval_results).sort_values('R2', ascending=False)
    results_df.to_csv(f'{model_dir}/regression_results.csv', index=False)

    logger.info(f"Best regression model: {best_name} with R2 {best_r2:.4f}")

    return {
        'best_model': best_model,
        'best_name': best_name,
        'eval_results': eval_results
    }

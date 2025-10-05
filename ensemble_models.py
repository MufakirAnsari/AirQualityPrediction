import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)

def create_voting_classifier(base_models, X_train, y_train):
    """
    Create voting classifier from base models
    """
    logger.info("Creating voting classifier...")
    
    # Prepare estimators for voting
    estimators = [(name, model) for name, model in base_models.items() 
                 if hasattr(model, 'predict_proba')]
    
    if len(estimators) < 2:
        logger.warning("Not enough models with predict_proba for voting classifier")
        return None
    
    # Create both hard and soft voting classifiers
    hard_voting = VotingClassifier(estimators=estimators, voting='hard')
    soft_voting = VotingClassifier(estimators=estimators, voting='soft')
    
    # Fit and evaluate both
    hard_voting.fit(X_train, y_train)
    soft_voting.fit(X_train, y_train)
    
    # Use cross-validation to select best voting method
    hard_scores = cross_val_score(hard_voting, X_train, y_train, cv=3, scoring='f1_macro')
    soft_scores = cross_val_score(soft_voting, X_train, y_train, cv=3, scoring='f1_macro')
    
    if np.mean(soft_scores) > np.mean(hard_scores):
        logger.info(f"Soft voting selected (score: {np.mean(soft_scores):.4f})")
        return soft_voting
    else:
        logger.info(f"Hard voting selected (score: {np.mean(hard_scores):.4f})")
        return hard_voting

def create_voting_regressor(base_models, X_train, y_train):
    """
    Create voting regressor from base models
    """
    logger.info("Creating voting regressor...")
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    if len(estimators) < 2:
        logger.warning("Not enough models for voting regressor")
        return None
    
    voting_regressor = VotingRegressor(estimators=estimators)
    voting_regressor.fit(X_train, y_train)
    
    # Evaluate performance
    score = cross_val_score(voting_regressor, X_train, y_train, cv=3, scoring='r2')
    logger.info(f"Voting regressor R² score: {np.mean(score):.4f}")
    
    return voting_regressor

def create_stacking_classifier(base_models, X_train, y_train):
    """
    Create stacking classifier from base models
    """
    logger.info("Creating stacking classifier...")
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    if len(estimators) < 2:
        logger.warning("Not enough models for stacking classifier")
        return None
    
    # Use logistic regression as meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3,
        stack_method='predict_proba'
    )
    
    stacking_classifier.fit(X_train, y_train)
    
    # Evaluate performance
    score = cross_val_score(stacking_classifier, X_train, y_train, cv=3, scoring='f1_macro')
    logger.info(f"Stacking classifier F1 score: {np.mean(score):.4f}")
    
    return stacking_classifier

def create_stacking_regressor(base_models, X_train, y_train):
    """
    Create stacking regressor from base models
    """
    logger.info("Creating stacking regressor...")
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    if len(estimators) < 2:
        logger.warning("Not enough models for stacking regressor")
        return None
    
    # Use linear regression as meta-learner
    meta_learner = LinearRegression()
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3
    )
    
    stacking_regressor.fit(X_train, y_train)
    
    # Evaluate performance
    score = cross_val_score(stacking_regressor, X_train, y_train, cv=3, scoring='r2')
    logger.info(f"Stacking regressor R² score: {np.mean(score):.4f}")
    
    return stacking_regressor

def create_ensemble_models(base_models, X_train, y_train, model_type='classification'):
    """
    Create all ensemble models from base models
    """
    ensemble_models = {}
    
    if model_type == 'classification':
        # Create voting classifier
        voting_clf = create_voting_classifier(base_models, X_train, y_train)
        if voting_clf is not None:
            ensemble_models['VotingClassifier'] = voting_clf
        
        # Create stacking classifier
        stacking_clf = create_stacking_classifier(base_models, X_train, y_train)
        if stacking_clf is not None:
            ensemble_models['StackingClassifier'] = stacking_clf
            
    elif model_type == 'regression':
        # Create voting regressor
        voting_reg = create_voting_regressor(base_models, X_train, y_train)
        if voting_reg is not None:
            ensemble_models['VotingRegressor'] = voting_reg
        
        # Create stacking regressor
        stacking_reg = create_stacking_regressor(base_models, X_train, y_train)
        if stacking_reg is not None:
            ensemble_models['StackingRegressor'] = stacking_reg
    
    logger.info(f"Created {len(ensemble_models)} ensemble models")
    return ensemble_models
"""
Train Logistic Regression, Random Forest, and XGBoost on the BRFSS dataset.
Build a weighted consensus ensemble from the two best performers.
Compute evaluation metrics and serialize all models.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_store')


def train_all():
    splits = joblib.load(os.path.join(MODELS_DIR, 'splits.joblib'))
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']

    models = {}

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr.fit(X_train, y_train)
    models['lr'] = lr

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    models['xgb'] = xgb

    print("\n--- Validation Metrics ---")
    val_metrics = {}
    for name, model in models.items():
        metrics = evaluate(model, X_val, y_val)
        val_metrics[name] = metrics
        print(f"{name}: {metrics}")

    print("\n--- Test Metrics ---")
    test_metrics = {}
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test)
        test_metrics[name] = metrics
        print(f"{name}: {metrics}")

    val_aucs = {name: val_metrics[name]['auc_roc'] for name in models}
    sorted_models = sorted(val_aucs.items(), key=lambda x: x[1], reverse=True)
    top_two = [sorted_models[0][0], sorted_models[1][0]]
    print(f"\nTop 2 models by validation AUC: {top_two}")

    w1 = val_aucs[top_two[0]]
    w2 = val_aucs[top_two[1]]
    total = w1 + w2
    weights = {top_two[0]: w1 / total, top_two[1]: w2 / total}
    print(f"Ensemble weights: {weights}")

    ensemble_info = {
        'models': top_two,
        'weights': weights,
    }

    ensemble_proba_val = (
        weights[top_two[0]] * models[top_two[0]].predict_proba(X_val)[:, 1] +
        weights[top_two[1]] * models[top_two[1]].predict_proba(X_val)[:, 1]
    )
    ensemble_pred_val = (ensemble_proba_val >= 0.5).astype(int)
    ens_val_metrics = {
        'accuracy': round(accuracy_score(y_val, ensemble_pred_val), 4),
        'f1': round(f1_score(y_val, ensemble_pred_val), 4),
        'auc_roc': round(roc_auc_score(y_val, ensemble_proba_val), 4),
        'precision': round(precision_score(y_val, ensemble_pred_val), 4),
        'recall': round(recall_score(y_val, ensemble_pred_val), 4),
    }
    print(f"\nEnsemble validation: {ens_val_metrics}")

    ensemble_proba_test = (
        weights[top_two[0]] * models[top_two[0]].predict_proba(X_test)[:, 1] +
        weights[top_two[1]] * models[top_two[1]].predict_proba(X_test)[:, 1]
    )
    ensemble_pred_test = (ensemble_proba_test >= 0.5).astype(int)
    ens_test_metrics = {
        'accuracy': round(accuracy_score(y_test, ensemble_pred_test), 4),
        'f1': round(f1_score(y_test, ensemble_pred_test), 4),
        'auc_roc': round(roc_auc_score(y_test, ensemble_proba_test), 4),
        'precision': round(precision_score(y_test, ensemble_pred_test), 4),
        'recall': round(recall_score(y_test, ensemble_pred_test), 4),
    }
    print(f"Ensemble test: {ens_test_metrics}")

    frac_pos_val, mean_pred_val = calibration_curve(y_val, ensemble_proba_val, n_bins=10)
    calibration_data = {
        'fraction_of_positives': frac_pos_val.tolist(),
        'mean_predicted_value': mean_pred_val.tolist()
    }

    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f'{name}_model.joblib')
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")

    joblib.dump(ensemble_info, os.path.join(MODELS_DIR, 'ensemble_info.joblib'))

    all_metrics = {
        'validation': {**val_metrics, 'ensemble': ens_val_metrics},
        'test': {**test_metrics, 'ensemble': ens_test_metrics},
        'ensemble_models': top_two,
        'ensemble_weights': weights,
        'calibration': calibration_data,
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("\nAll models and metrics saved.")


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        'accuracy': round(accuracy_score(y, y_pred), 4),
        'f1': round(f1_score(y, y_pred), 4),
        'auc_roc': round(roc_auc_score(y, y_proba), 4),
        'precision': round(precision_score(y, y_pred), 4),
        'recall': round(recall_score(y, y_pred), 4),
    }


if __name__ == '__main__':
    train_all()

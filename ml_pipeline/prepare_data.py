"""
Download and preprocess the BRFSS 2022 Heart Disease Health Indicators dataset.
Produces train/val/test splits with SMOTE applied to training set only.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_store')

FEATURE_COLS = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
    'HeartDiseaseorAttack'
]

RENAME_MAP = {
    'HeartDiseaseorAttack': 'target'
}


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, 'brfss2022.csv')
    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}")
        return csv_path

    try:
        import kagglehub
        path = kagglehub.dataset_download("pytlak/brfss-2022-heart-disease-health-indicators")
        import glob
        csvs = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
        if csvs:
            import shutil
            shutil.copy(csvs[0], csv_path)
            print(f"Downloaded to {csv_path}")
            return csv_path
    except Exception as e:
        print(f"Kaggle download failed: {e}")

    print("Attempting direct download...")
    import urllib.request
    url = "https://raw.githubusercontent.com/pytlak/brfss-2022-heart-disease-health-indicators/main/heart_disease_health_indicators_BRFSS2022.csv"
    try:
        urllib.request.urlretrieve(url, csv_path)
        print(f"Downloaded to {csv_path}")
        return csv_path
    except Exception as e2:
        print(f"Direct download also failed: {e2}")
        raise RuntimeError("Could not download dataset. Please place the CSV manually at data/brfss2022.csv")


def preprocess(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")

    available = [c for c in FEATURE_COLS if c in df.columns]
    if 'HeartDiseaseorAttack' not in df.columns:
        for alt in ['HeartDiseaseOrAttack', 'heartdiseaseorattack', 'Target', 'target']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'HeartDiseaseorAttack'})
                break

    df = df[available].copy()
    df = df.rename(columns=RENAME_MAP)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Clean shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

    X = df.drop('target', axis=1)
    y = df['target'].astype(int)

    feature_names = list(X.columns)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_res.shape}")
    print(f"Resampled distribution:\n{pd.Series(y_train_res).value_counts(normalize=True)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    splits = {
        'X_train': X_train_res, 'y_train': y_train_res,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
    joblib.dump(splits, os.path.join(MODELS_DIR, 'splits.joblib'))
    print("Splits saved.")
    return splits


if __name__ == '__main__':
    csv_path = download_dataset()
    preprocess(csv_path)

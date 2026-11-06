"""
src/preprocess.py
Data preprocessing pipeline for Crop Yield Prediction.

Tasks performed:
1. Loads Karnataka multi-crop dataset (120 rows).
2. Encodes categorical variables (crop, district, season, soil_type) with LabelEncoder.
   Saves encoders dict to results/encoders.pkl.
3. Scales 12 input features using MinMaxScaler.
   Saves fitted scaler to results/scaler.pkl.
4. Reshapes feature matrix X to 3D tensor (n_samples, n_features, 1) for Conv1D.
5. Provides get_data() for 80/20 train-test split stratified by crop.
6. Provides get_cv_splits() for reproducible 5-fold Stratified cross-validation.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib

# Global Reproducibility Constant
SEED = 42
np.random.seed(SEED)

# Feature & Target specifications
CATEGORICAL_COLS = ["crop", "district", "season", "soil_type"]

FEATURE_COLS = [
    "crop",
    "year",
    "district",
    "season",
    "soil_type",
    "nitrogen",
    "phosphorus",
    "potassium",
    "ph",
    "temperature",
    "rainfall",
    "humidity",
]

TARGET_COL = "yield_tonnes_per_hectare"


def resolve_project_paths(
    csv_path: Optional[str] = None, results_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Resolves dataset path and results output directory across common execution contexts
    (project root, crop_yield_project subdir, or from within src/).
    """
    # Candidate locations for CSV
    candidates = []
    if csv_path:
        candidates.append(csv_path)

    # Standard default paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates.extend(
        [
            Path.cwd() / "data" / "karnataka_multi_crop_master.csv",
            project_root / "data" / "karnataka_multi_crop_master.csv",
            Path.cwd() / "crop_yield_project" / "data" / "karnataka_multi_crop_master.csv",
            Path("/data/karnataka_multi_crop_master.csv"),
            Path("/public/assets/aistudio/karnataka_multi_crop_master_e9e443cf.csv"),
        ]
    )

    resolved_csv = None
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.is_file():
            resolved_csv = str(cand_path.resolve())
            break

    if resolved_csv is None:
        # Fall back to default string if not yet created on disk
        resolved_csv = str(candidates[0]) if candidates else "data/karnataka_multi_crop_master.csv"

    # Candidate locations for results
    if results_dir:
        resolved_results = Path(results_dir)
    elif (project_root / "results").is_dir() or project_root.exists():
        resolved_results = project_root / "results"
    else:
        resolved_results = Path.cwd() / "results"

    resolved_results.mkdir(parents=True, exist_ok=True)

    return resolved_csv, str(resolved_results.resolve())


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads dataset from CSV and verifies expected columns.
    """
    resolved_csv, _ = resolve_project_paths(csv_path=csv_path)
    if not os.path.exists(resolved_csv):
        raise FileNotFoundError(
            f"Dataset not found at '{resolved_csv}'. Please ensure "
            f"'data/karnataka_multi_crop_master.csv' exists."
        )

    df = pd.read_csv(resolved_csv)

    # Sanity checks
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COL}'")

    return df


def encode_categoricals(
    df: pd.DataFrame, results_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encodes categorical features using LabelEncoder and saves encoders to results/encoders.pkl.
    """
    _, resolved_results = resolve_project_paths(results_dir=results_dir)
    df_encoded = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    encoders_path = os.path.join(resolved_results, "encoders.pkl")
    joblib.dump(encoders, encoders_path)

    return df_encoded, encoders


def get_data(
    csv_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    fit_scaler_on_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads raw CSV, encodes categorical variables, performs an 80/20 train/test split
    stratified by crop, fits/applies MinMaxScaler, and reshapes feature matrices to 3D
    tensors (n_samples, 12, 1) for Conv1D input.

    Returns:
        X_train: np.ndarray of shape (n_train, 12, 1)
        X_test:  np.ndarray of shape (n_test, 12, 1)
        y_train: np.ndarray of shape (n_train,)
        y_test:  np.ndarray of shape (n_test,)
    """
    resolved_csv, resolved_results = resolve_project_paths(csv_path, results_dir)

    # 1. Load data
    df = load_dataset(resolved_csv)

    # 2. Encode categorical variables
    df_encoded, _ = encode_categoricals(df, results_dir=resolved_results)

    # 3. Extract features & target
    X_raw = df_encoded[FEATURE_COLS].values.astype(np.float32)
    y_raw = df_encoded[TARGET_COL].values.astype(np.float32)

    # Stratification label: original crop ID
    crop_stratify = df_encoded["crop"].values

    # 4. Train-test split (80/20, stratified by crop, random_state=SEED)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=SEED,
        stratify=crop_stratify,
    )

    # 5. Feature scaling using MinMaxScaler
    scaler = MinMaxScaler()
    if fit_scaler_on_train:
        # Prevent data leakage: fit on train, transform train and test
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
    else:
        scaler.fit(X_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

    scaler_path = os.path.join(resolved_results, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 6. Reshape for Conv1D input: (n_samples, n_features, 1)
    X_train = X_train_scaled.reshape((X_train_scaled.shape[0], len(FEATURE_COLS), 1))
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0], len(FEATURE_COLS), 1))

    return X_train, X_test, y_train, y_test


def get_full_data(
    csv_path: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns full processed features X (scaled, reshaped), target y,
    and crop category array for cross-validation stratification.

    Returns:
        X: np.ndarray of shape (n_samples, 12, 1)
        y: np.ndarray of shape (n_samples,)
        crop_stratify: np.ndarray of shape (n_samples,)
    """
    resolved_csv, resolved_results = resolve_project_paths(csv_path, results_dir)
    df = load_dataset(resolved_csv)
    df_encoded, _ = encode_categoricals(df, results_dir=resolved_results)

    X_raw = df_encoded[FEATURE_COLS].values.astype(np.float32)
    y = df_encoded[TARGET_COL].values.astype(np.float32)
    crop_stratify = df_encoded["crop"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Save scaler if not already saved
    scaler_path = os.path.join(resolved_results, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    X = X_scaled.reshape((X_scaled.shape[0], len(FEATURE_COLS), 1))
    return X, y, crop_stratify


def get_cv_splits(n_splits: int = 5) -> StratifiedKFold:
    """
    Returns a StratifiedKFold splitter object with 5 splits, shuffle=True, random_state=42.

    Usage with continuous target y and categorical crop:
        cv = get_cv_splits()
        for train_idx, val_idx in cv.split(X, crop_stratify):
            ...
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PREPROCESSING VERIFICATION (src/preprocess.py)")
    print("=" * 60)

    # 1. Run get_data()
    X_train, X_test, y_train, y_test = get_data()

    n_total = len(X_train) + len(X_test)
    print(f"[OK] Random Seed: {SEED}")
    print(f"[OK] Total records loaded: {n_total}")
    print(f"[OK] X_train shape: {X_train.shape}  (80% split -> {len(X_train)} samples, 12 features, 1 channel)")
    print(f"[OK] X_test  shape: {X_test.shape}   (20% split -> {len(X_test)} samples, 12 features, 1 channel)")
    print(f"[OK] y_train shape: {y_train.shape}")
    print(f"[OK] y_test  shape: {y_test.shape}")

    # 2. Check value ranges
    print(f"[OK] Feature min in X_train: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"[OK] Yield min in y_train:   {y_train.min():.4f}, max: {y_train.max():.4f}")

    # 3. Print sample row
    print("\n--- Sample Scaled Feature Vector (X_train[0]) ---")
    for feat_name, val in zip(FEATURE_COLS, X_train[0].flatten()):
        print(f"  {feat_name:<14}: {val:.4f}")
    print(f"  --> yield (tonnes/ha): {y_train[0]:.2f}")

    # 4. Verify 5-fold CV splitter
    cv = get_cv_splits()
    print(f"\n[OK] 5-Fold Stratified Splitter: {cv}")

    _, results_path = resolve_project_paths()
    print(f"\n[OK] Encoders saved to: {os.path.join(results_path, 'encoders.pkl')}")
    print(f"[OK] Scaler saved to:   {os.path.join(results_path, 'scaler.pkl')}")
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

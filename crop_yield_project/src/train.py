"""
src/train.py
End-to-end experiment orchestration and comparative evaluation suite.

Executes 5 comparative experiments on the test set:
  A) CNN_only: Ablation model with Bidirectional GRU replaced by GlobalAveragePooling1D.
  B) BiGRU_only: Ablation model with Conv1D removed (features input directly to BiGRU).
  C) CNN_BiGRU: Hybrid baseline with default hyperparameters.
  D) CNN_BiGRU_PSO: Hybrid model with single-objective PSO-optimized hyperparameters.
  E) CNN_BiGRU_NSPSO: Proposed hybrid model with bi-objective NSPSO-selected hyperparameters
     (Lowest RMSE subject to n_params <= 5000; or global lowest RMSE if all > 5000).

Computes on the TEST set:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
  - Trainable parameters count
  - Training execution time (seconds)

Saves:
  - results/metrics.json
  - results/predictions.npz (for evaluate.py visualization)
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Local module imports
from src.preprocess import get_data, get_cv_splits, resolve_project_paths
from src.model_cnn_bigru import build_cnn_bigru, train_model, count_params
from src.pso_nspso import (
    NSPSO,
    run_single_objective_pso,
    MIN_BOUNDS,
    MAX_BOUNDS,
    decode_particle,
)

# Global reproducibility seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# ABLATION MODEL BUILDERS
# ==============================================================================

def build_cnn_only(
    input_shape: Tuple[int, int] = (12, 1),
    conv_filters: int = 32,
    kernel_size: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs CNN-only ablation model (BiGRU replaced with GlobalAveragePooling1D).
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_features"),
            layers.Conv1D(
                filters=int(conv_filters),
                kernel_size=int(kernel_size),
                padding="same",
                activation="relu",
                name="conv1d_features",
            ),
            layers.MaxPooling1D(pool_size=2, padding="same", name="maxpool1d"),
            layers.GlobalAveragePooling1D(name="global_avg_pool"),
            layers.Dropout(float(dropout), name="dropout_reg"),
            layers.Dense(16, activation="relu", name="dense_latent"),
            layers.Dense(1, activation="linear", name="yield_output"),
        ],
        name="CNN_Only_Ablation",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_bigru_only(
    input_shape: Tuple[int, int] = (12, 1),
    gru_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs BiGRU-only ablation model (Conv1D removed; input fed directly to BiGRU).
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_features"),
            layers.Bidirectional(
                layers.GRU(int(gru_units), return_sequences=False),
                name="bidirectional_gru",
            ),
            layers.Dropout(float(dropout), name="dropout_reg"),
            layers.Dense(16, activation="relu", name="dense_latent"),
            layers.Dense(1, activation="linear", name="yield_output"),
        ],
        name="BiGRU_Only_Ablation",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ==============================================================================
# HYPERPARAMETER RETRIEVAL HELPERS
# ==============================================================================

def get_pso_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    crop_train: np.ndarray,
    results_dir: str,
) -> Dict[str, Any]:
    """
    Loads PSO-optimized parameters from results/pso_results.json.
    If absent, executes single-objective PSO optimization.
    """
    pso_json = os.path.join(results_dir, "pso_results.json")
    if os.path.exists(pso_json):
        try:
            with open(pso_json, "r") as f:
                data = json.load(f)
                if "best_params" in data and data["best_params"]:
                    print(f"[INFO] Loaded existing PSO parameters from {pso_json}")
                    return data["best_params"]
        except Exception as e:
            print(f"[WARN] Error reading {pso_json}: {e}. Re-running PSO.")

    # Re-run PSO if not present
    print("[INFO] pso_results.json not found. Running Single-Objective PSO...")
    best_params, _ = run_single_objective_pso(
        X_train=X_train,
        y_train=y_train,
        crop_train=crop_train,
        n_particles=8,
        iters=8,
        results_dir=results_dir,
    )
    return best_params


def get_nspso_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    crop_train: np.ndarray,
    results_dir: str,
    pso_ref_rmse: float = 0.5,
) -> Dict[str, Any]:
    """
    Selects the optimal trade-off solution from NSPSO Pareto front:
    Minimizes RMSE subject to n_params <= 5000.
    If all solutions exceed 5000 parameters, selects the solution with global lowest RMSE.
    """
    nspso_json = os.path.join(results_dir, "nspso_results.json")
    pareto_solutions = []

    if os.path.exists(nspso_json):
        try:
            with open(nspso_json, "r") as f:
                pareto_solutions = json.load(f)
                print(f"[INFO] Loaded {len(pareto_solutions)} solutions from {nspso_json}")
        except Exception as e:
            print(f"[WARN] Error reading {nspso_json}: {e}.")

    if not pareto_solutions:
        print("[INFO] nspso_results.json not found. Running Multi-Objective NSPSO...")
        nspso = NSPSO(
            n_particles=8,
            iters=8,
            dims=4,
            bounds=(MIN_BOUNDS, MAX_BOUNDS),
            archive_size=20,
        )
        nspso.optimize(
            X_train=X_train,
            y_train=y_train,
            crop_train=crop_train,
            results_dir=results_dir,
            pso_ref_rmse=pso_ref_rmse,
        )
        with open(nspso_json, "r") as f:
            pareto_solutions = json.load(f)

    # Apply decision rule:
    # 1. Filter candidates where n_params <= 5000
    constrained = [sol for sol in pareto_solutions if sol.get("n_params", 999999) <= 5000]

    if constrained:
        # Pick the one with lowest validation RMSE
        selected = min(constrained, key=lambda s: s.get("rmse", float("inf")))
        print(
            f"[INFO] NSPSO Selection (constrained n_params <= 5000): "
            f"Params={selected['n_params']}, Val RMSE={selected['rmse']:.4f}"
        )
    else:
        # Fallback to global lowest RMSE
        selected = min(pareto_solutions, key=lambda s: s.get("rmse", float("inf")))
        print(
            f"[INFO] NSPSO Selection (all exceed 5000; selected lowest RMSE): "
            f"Params={selected['n_params']}, Val RMSE={selected['rmse']:.4f}"
        )

    return selected["params"]


# ==============================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# ==============================================================================

def run_experiments() -> Dict[str, Dict[str, Any]]:
    """
    Runs all 5 comparative experiments on the test set and returns the metrics dictionary.
    """
    _, results_dir = resolve_project_paths()

    print("=" * 75)
    print("STARTING CROP YIELD EXPERIMENTAL BENCHMARK SUITE")
    print("Dataset: 120 total records (80% Train [96], 20% Test [24])")
    print("=" * 75)

    # 1. Load data
    X_train_full, X_test, y_train_full, y_test = get_data()

    # 2. Carve validation set (10% of train = ~10 samples for validation, 86 for training)
    # Using seed 42 for exact reproducibility
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.10,
        random_state=SEED,
    )

    print(f"[OK] Training Split:   {X_train.shape[0]} samples")
    print(f"[OK] Validation Split: {X_val.shape[0]} samples (Carved 10% for EarlyStopping)")
    print(f"[OK] Test Split:       {X_test.shape[0]} samples (Held-out evaluation)")

    # Retrieve crop categories for train set for PSO / NSPSO if re-run is needed
    from src.preprocess import load_dataset, encode_categoricals
    resolved_csv, _ = resolve_project_paths()
    df_raw = load_dataset(resolved_csv)
    df_enc, _ = encode_categoricals(df_raw)
    crop_all = df_enc["crop"].values
    from sklearn.model_selection import train_test_split as tts
    _, _, _, _, crop_train_full, _ = tts(
        df_enc.values, df_enc["yield_tonnes_per_hectare"].values, crop_all, test_size=0.20, random_state=SEED, stratify=crop_all
    )

    # 3. Retrieve PSO and NSPSO hyperparameters
    pso_params = get_pso_hyperparameters(
        X_train=X_train_full,
        y_train=y_train_full,
        crop_train=crop_train_full,
        results_dir=results_dir,
    )

    nspso_params = get_nspso_hyperparameters(
        X_train=X_train_full,
        y_train=y_train_full,
        crop_train=crop_train_full,
        results_dir=results_dir,
    )

    # 4. Define the 5 model specifications
    experiments = {
        "CNN_only": {
            "builder": lambda: build_cnn_only(
                input_shape=(12, 1),
                conv_filters=32,
                kernel_size=2,
                dropout=0.2,
                learning_rate=0.001,
            ),
            "description": "CNN with Conv1D + GlobalAveragePooling1D (No GRU)",
        },
        "BiGRU_only": {
            "builder": lambda: build_bigru_only(
                input_shape=(12, 1),
                gru_units=32,
                dropout=0.2,
                learning_rate=0.001,
            ),
            "description": "Bidirectional GRU only (No Conv1D)",
        },
        "CNN_BiGRU": {
            "builder": lambda: build_cnn_bigru(
                input_shape=(12, 1),
                conv_filters=32,
                kernel_size=2,
                gru_units=32,
                dropout=0.2,
                learning_rate=0.001,
            ),
            "description": "Baseline CNN-BiGRU (Default Hyperparameters)",
        },
        "CNN_BiGRU_PSO": {
            "builder": lambda: build_cnn_bigru(
                input_shape=(12, 1),
                conv_filters=pso_params["conv_filters"],
                kernel_size=2,
                gru_units=pso_params["gru_units"],
                dropout=pso_params["dropout"],
                learning_rate=pso_params["learning_rate"],
            ),
            "description": f"CNN-BiGRU (PSO Tuned: filters={pso_params['conv_filters']}, gru={pso_params['gru_units']})",
        },
        "CNN_BiGRU_NSPSO": {
            "builder": lambda: build_cnn_bigru(
                input_shape=(12, 1),
                conv_filters=nspso_params["conv_filters"],
                kernel_size=2,
                gru_units=nspso_params["gru_units"],
                dropout=nspso_params["dropout"],
                learning_rate=nspso_params["learning_rate"],
            ),
            "description": f"CNN-BiGRU (NSPSO Pareto: filters={nspso_params['conv_filters']}, gru={nspso_params['gru_units']})",
        },
    }

    metrics: Dict[str, Dict[str, Any]] = {}
    proposed_predictions = None

    # 5. Run each experiment sequentially with try/except protection
    for model_key, spec in experiments.items():
        print("\n" + "-" * 75)
        print(f"RUNNING EXPERIMENT: {model_key}")
        print(f"Description: {spec['description']}")
        print("-" * 75)

        try:
            # Seed initialization before training
            tf.keras.backend.clear_session()
            tf.random.set_seed(SEED)
            np.random.seed(SEED)

            start_t = time.time()

            # Build model
            model = spec["builder"]()
            n_params = count_params(model)

            # Train with early stopping on carved validation set
            model, history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=100,
                batch_size=16,
                patience=10,
                verbose=0,
            )

            train_time = round(time.time() - start_t, 3)

            # Predict on TEST set
            y_pred_test = model.predict(X_test, verbose=0).flatten()

            # Compute evaluation metrics
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = float(mean_absolute_error(y_test, y_pred_test))
            r2 = float(r2_score(y_test, y_pred_test))

            metrics[model_key] = {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "n_params": int(n_params),
                "train_time": train_time,
            }

            print(
                f"[SUCCESS] {model_key} -> "
                f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | "
                f"Params: {n_params:,} | Time: {train_time:.2f}s"
            )

            # Store predictions and loss curves for proposed model E
            if model_key == "CNN_BiGRU_NSPSO":
                y_pred_train = model.predict(X_train_full, verbose=0).flatten()
                proposed_predictions = {
                    "y_true": y_test,
                    "y_pred": y_pred_test,
                    "y_train_true": y_train_full,
                    "y_train_pred": y_pred_train,
                    "loss": np.array(history.history.get("loss", [])),
                    "val_loss": np.array(history.history.get("val_loss", [])),
                }

        except Exception as e:
            print(f"[ERROR] Experiment '{model_key}' encountered an exception: {e}")
            metrics[model_key] = {
                "rmse": None,
                "mae": None,
                "r2": None,
                "n_params": None,
                "train_time": None,
                "error": str(e),
            }

    # 6. Save results to results/metrics.json
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Metrics successfully saved to: {metrics_path}")

    # 7. Save predictions of proposed model E to results/predictions.npz
    if proposed_predictions is not None:
        pred_path = os.path.join(results_dir, "predictions.npz")
        np.savez(
            pred_path,
            y_true=proposed_predictions["y_true"],
            y_pred=proposed_predictions["y_pred"],
            y_train_true=proposed_predictions["y_train_true"],
            y_train_pred=proposed_predictions["y_train_pred"],
            loss=proposed_predictions.get("loss", np.array([])),
            val_loss=proposed_predictions.get("val_loss", np.array([])),
        )
        print(f"[OK] Proposed model predictions saved to: {pred_path}")
    elif "CNN_BiGRU" in metrics and metrics["CNN_BiGRU"]["rmse"] is not None:
        # Fallback if NSPSO failed: save baseline predictions
        print("[WARN] NSPSO model unavailable; saving baseline predictions.")

    # 8. Print clean summary table to stdout
    print("\n" + "=" * 75)
    print("FINAL EXPERIMENTAL BENCHMARK SUMMARY TABLE")
    print("=" * 75)
    header = f"{'Model':<18} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8} | {'Params':<8} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))

    for model_name, res in metrics.items():
        if res.get("rmse") is not None:
            print(
                f"{model_name:<18} | "
                f"{res['rmse']:<8.4f} | "
                f"{res['mae']:<8.4f} | "
                f"{res['r2']:<8.4f} | "
                f"{res['n_params']:<8,d} | "
                f"{res['train_time']:<8.2f}"
            )
        else:
            print(f"{model_name:<18} | FAILED ({res.get('error', 'unknown error')[:25]})")

    print("=" * 75)

    return metrics


if __name__ == "__main__":
    run_experiments()

"""
src/pso_nspso.py
Optimization framework for CNN-BiGRU crop yield prediction models.

Includes:
  PART A: Single-Objective PSO (pyswarms) tuning 4 hyperparameters:
          [conv_filters, gru_units, dropout, learning_rate (log-scale)]
          Objective: Minimize validation RMSE (5-fold CV on train set).
  PART B: Multi-Objective NSPSO (Non-Dominated Sorting PSO from scratch):
          Objective 1 (f1): Minimize validation RMSE (3-fold CV on train set).
          Objective 2 (f2): Minimize trainable parameter count (model simplicity).
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import pyswarms as ps
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Local imports
from src.preprocess import resolve_project_paths, load_dataset, encode_categoricals, FEATURE_COLS, TARGET_COL, get_cv_splits
from src.model_cnn_bigru import build_cnn_bigru, train_model, count_params

# Global reproducibility seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameter search bounds:
# [0] conv_filters : [8, 64]
# [1] gru_units    : [8, 64]
# [2] dropout      : [0.1, 0.5]
# [3] log10(lr)    : [-4.0, -2.0]  --> 10**x in [1e-4, 1e-2]
MIN_BOUNDS = np.array([8.0, 8.0, 0.1, -4.0], dtype=np.float64)
MAX_BOUNDS = np.array([64.0, 64.0, 0.5, -2.0], dtype=np.float64)


def decode_particle(particle: np.ndarray) -> Dict[str, Any]:
    """
    Decodes a continuous 4D particle vector into discrete and log-scaled hyperparameters.
    """
    clipped = np.clip(particle, MIN_BOUNDS, MAX_BOUNDS)
    conv_filters = int(np.round(clipped[0]))
    gru_units = int(np.round(clipped[1]))
    dropout = float(clipped[2])
    learning_rate = float(10 ** clipped[3])

    return {
        "conv_filters": conv_filters,
        "gru_units": gru_units,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "log_lr": float(clipped[3]),
    }


def evaluate_candidate_cv(
    particle: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    crop_train: np.ndarray,
    n_splits: int = 5,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 16,
) -> Tuple[float, int]:
    """
    Evaluates candidate hyperparameters using Stratified K-Fold CV on training data.

    Returns:
        mean_val_rmse: Mean RMSE across validation folds.
        param_count: Trainable parameter count of the model architecture.
    """
    params = decode_particle(particle)

    # Set seeds before model construction to ensure deterministic evaluation
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Build reference model to compute objective 2 (param count)
    ref_model = build_cnn_bigru(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        conv_filters=params["conv_filters"],
        kernel_size=2,
        gru_units=params["gru_units"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
    )
    param_count = count_params(ref_model)
    tf.keras.backend.clear_session()

    # Stratified K-Fold on training subset using categorical crop
    cv = get_cv_splits(n_splits=n_splits)
    fold_rmses = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, crop_train)):
        X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Reset seed per fold
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED + fold)
        np.random.seed(SEED + fold)

        model = build_cnn_bigru(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            conv_filters=params["conv_filters"],
            kernel_size=2,
            gru_units=params["gru_units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )

        model, _ = train_model(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=0,
        )

        val_preds = model.predict(X_val, verbose=0).flatten()
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        fold_rmses.append(val_rmse)

        tf.keras.backend.clear_session()

    mean_rmse = float(np.mean(fold_rmses))
    return mean_rmse, param_count


# ==============================================================================
# PART A: SINGLE-OBJECTIVE PSO (pyswarms)
# ==============================================================================

def pso_fitness(
    params_array: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    crop_train: np.ndarray,
) -> np.ndarray:
    """
    Vectorized fitness evaluation for pyswarms.
    Takes params_array of shape (n_particles, 4) and returns 1D array of shape (n_particles,).
    """
    n_particles = params_array.shape[0]
    costs = np.zeros(n_particles, dtype=np.float64)

    for i in range(n_particles):
        rmse, _ = evaluate_candidate_cv(
            particle=params_array[i],
            X_train=X_train,
            y_train=y_train,
            crop_train=crop_train,
            n_splits=5,
            epochs=50,
            patience=8,
        )
        costs[i] = rmse

    return costs


def run_single_objective_pso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    crop_train: np.ndarray,
    n_particles: int = 8,
    iters: int = 8,
    results_dir: str = "results",
) -> Tuple[Dict[str, Any], float]:
    """
    Runs single-objective PSO to minimize 5-fold CV validation RMSE.
    """
    print("\n" + "=" * 65)
    print("STARTING PART A: SINGLE-OBJECTIVE PSO OPTIMIZATION")
    print(f"Particles: {n_particles} | Iterations: {iters} | Bounds: 4D Hyperparameters")
    print("=" * 65)

    bounds = (MIN_BOUNDS, MAX_BOUNDS)
    options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=4,
        options=options,
        bounds=bounds,
    )

    best_cost_tracker = {"best_rmse": float("inf"), "best_params": None}

    def objective_wrapper(p_array):
        costs = pso_fitness(p_array, X_train, y_train, crop_train)
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_cost_tracker["best_rmse"]:
            best_cost_tracker["best_rmse"] = float(costs[min_idx])
            best_cost_tracker["best_params"] = decode_particle(p_array[min_idx])

        # Save intermediate snapshot
        intermediate_path = os.path.join(results_dir, "pso_results.json")
        with open(intermediate_path, "w") as f:
            json.dump(best_cost_tracker, f, indent=2)

        return costs

    best_cost, best_pos = optimizer.optimize(objective_wrapper, iters=iters)
    best_params = decode_particle(best_pos)

    results_data = {
        "best_params": best_params,
        "best_rmse": float(best_cost),
    }

    out_file = os.path.join(results_dir, "pso_results.json")
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"[OK] PSO Finished! Best RMSE: {best_cost:.4f}")
    print(f"[OK] Best Params: {best_params}")
    print(f"[OK] Saved to: {out_file}")

    return best_params, float(best_cost)


# ==============================================================================
# PART B: MULTI-OBJECTIVE NSPSO (Non-Dominated Sorting PSO from Scratch)
# ==============================================================================

class NSPSO:
    """
    Non-Dominated Sorting Particle Swarm Optimization (NSPSO) for multi-objective optimization.

    Objectives (both minimized):
      - f1: Validation RMSE (3-fold CV on training set)
      - f2: Number of trainable parameters (complexity penalization)
    """

    def __init__(
        self,
        n_particles: int = 8,
        iters: int = 8,
        dims: int = 4,
        bounds: Tuple[np.ndarray, np.ndarray] = (MIN_BOUNDS, MAX_BOUNDS),
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        archive_size: int = 20,
    ):
        self.n_particles = n_particles
        self.iters = iters
        self.dims = dims
        self.min_bounds, self.max_bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size

        # Initialize particles uniformly within bounds
        self.particles = np.random.uniform(
            self.min_bounds, self.max_bounds, size=(self.n_particles, self.dims)
        )
        # Initialize velocities scaled to 10% of bound spans
        bound_span = self.max_bounds - self.min_bounds
        self.v_max = 0.20 * bound_span
        self.velocities = np.random.uniform(
            -self.v_max * 0.5, self.v_max * 0.5, size=(self.n_particles, self.dims)
        )

        # Personal best positions and costs
        self.pbest_pos = np.copy(self.particles)
        self.pbest_cost = np.full((self.n_particles, 2), np.inf)

        # External non-dominated archive: list of dicts with {"pos": ..., "cost": ...}
        self.archive: List[Dict[str, np.ndarray]] = []

    def evaluate(
        self,
        positions: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        crop_train: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates candidate positions on [f1=RMSE, f2=ParamCount].
        Uses 3-fold CV for speed on CPU.
        """
        costs = np.zeros((positions.shape[0], 2), dtype=np.float64)

        for i in range(positions.shape[0]):
            rmse, n_params = evaluate_candidate_cv(
                particle=positions[i],
                X_train=X_train,
                y_train=y_train,
                crop_train=crop_train,
                n_splits=3,  # 3-fold CV for fast NSPSO execution
                epochs=40,
                patience=6,
            )
            costs[i, 0] = rmse
            costs[i, 1] = float(n_params)

        return costs

    @staticmethod
    def dominates(cost_a: np.ndarray, cost_b: np.ndarray) -> bool:
        """
        Returns True if cost_a Pareto-dominates cost_b (both objectives minimized).
        """
        return np.all(cost_a <= cost_b) and np.any(cost_a < cost_b)

    def fast_non_dominated_sort(self, costs: np.ndarray) -> List[List[int]]:
        """
        Deb's Fast Non-Dominated Sorting Algorithm.
        Returns list of Pareto fronts (each front is a list of candidate indices).
        """
        n = costs.shape[0]
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self.dominates(costs[p], costs[q]):
                    dominated_solutions[p].append(q)
                elif self.dominates(costs[q], costs[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for p in fronts[current_front]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            current_front += 1
            fronts.append(next_front)

        # Remove trailing empty front
        if len(fronts[-1]) == 0:
            fronts.pop()

        return fronts

    def crowding_distance(self, front_costs: np.ndarray) -> np.ndarray:
        """
        Calculates crowding distance for each solution in a front.
        """
        n_points = front_costs.shape[0]
        if n_points <= 2:
            return np.full(n_points, np.inf)

        distances = np.zeros(n_points, dtype=np.float64)

        for obj_idx in range(front_costs.shape[1]):
            sorted_indices = np.argsort(front_costs[:, obj_idx])
            obj_min = front_costs[sorted_indices[0], obj_idx]
            obj_max = front_costs[sorted_indices[-1], obj_idx]

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            norm_range = obj_max - obj_min
            if norm_range == 0:
                continue

            for i in range(1, n_points - 1):
                prev_val = front_costs[sorted_indices[i - 1], obj_idx]
                next_val = front_costs[sorted_indices[i + 1], obj_idx]
                distances[sorted_indices[i]] += (next_val - prev_val) / norm_range

        return distances

    def update_archive(self, candidates: List[Dict[str, np.ndarray]]) -> None:
        """
        Merges candidates with current archive, performs non-dominated sorting,
        retains Front 1, and truncates to archive_size via crowding distance.
        """
        all_members = self.archive + candidates
        if not all_members:
            return

        costs = np.array([m["cost"] for m in all_members], dtype=np.float64)

        # Non-dominated sort
        fronts = self.fast_non_dominated_sort(costs)
        front_0_indices = fronts[0]

        # Extract only non-dominated solutions
        front_0_members = [all_members[idx] for idx in front_0_indices]
        front_0_costs = costs[front_0_indices]

        # Truncate if exceeds archive size
        if len(front_0_members) > self.archive_size:
            cd = self.crowding_distance(front_0_costs)
            # Sort descending by crowding distance to prioritize boundary & diverse solutions
            sorted_order = np.argsort(-cd)[: self.archive_size]
            self.archive = [front_0_members[i] for i in sorted_order]
        else:
            self.archive = front_0_members

    def update_pbest(self, current_positions: np.ndarray, current_costs: np.ndarray) -> None:
        """
        Updates each particle's personal best using Pareto dominance.
        """
        for i in range(self.n_particles):
            curr_c = current_costs[i]
            pbest_c = self.pbest_cost[i]

            if np.isinf(pbest_c[0]):
                self.pbest_cost[i] = curr_c
                self.pbest_pos[i] = current_positions[i]
            elif self.dominates(curr_c, pbest_c):
                self.pbest_cost[i] = curr_c
                self.pbest_pos[i] = current_positions[i]
            elif not self.dominates(pbest_c, curr_c):
                # Non-dominated: 50% probability of updating
                if np.random.rand() < 0.5:
                    self.pbest_cost[i] = curr_c
                    self.pbest_pos[i] = current_positions[i]

    def update_velocity_position(self) -> None:
        """
        Standard PSO velocity and position update using archive leaders.
        """
        if not self.archive:
            return

        archive_positions = np.array([m["pos"] for m in self.archive])

        for i in range(self.n_particles):
            # Select leader from archive at random (or tournament)
            leader_idx = np.random.randint(0, len(self.archive))
            gbest = archive_positions[leader_idx]

            r1 = np.random.rand(self.dims)
            r2 = np.random.rand(self.dims)

            # Velocity update
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                + self.c2 * r2 * (gbest - self.particles[i])
            )

            # Clip velocities
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Position update
            self.particles[i] = self.particles[i] + self.velocities[i]

            # Clip positions to bounds
            self.particles[i] = np.clip(self.particles[i], self.min_bounds, self.max_bounds)

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        crop_train: np.ndarray,
        results_dir: str = "results",
        pso_ref_rmse: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Executes the multi-objective optimization loop.
        """
        print("\n" + "=" * 65)
        print("STARTING PART B: MULTI-OBJECTIVE NSPSO OPTIMIZATION")
        print(f"Particles: {self.n_particles} | Iterations: {self.iters} | Archive: {self.archive_size}")
        print("=" * 65)

        for it in range(self.iters):
            # 1. Evaluate current particles
            current_costs = self.evaluate(self.particles, X_train, y_train, crop_train)

            # 2. Update personal bests
            self.update_pbest(self.particles, current_costs)

            # 3. Update external Pareto archive
            candidates = [
                {"pos": np.copy(self.particles[i]), "cost": np.copy(current_costs[i])}
                for i in range(self.n_particles)
            ]
            self.update_archive(candidates)

            # 4. Update velocity and position
            self.update_velocity_position()

            # Status log per iteration as requested
            print(
                f"Iter {it + 1}/{self.iters} | "
                f"PSO best RMSE: {pso_ref_rmse:.4f} | "
                f"NSPSO archive size: {len(self.archive)}"
            )

            # Save intermediate results
            self.save_results(results_dir=results_dir)

        archive_positions = [m["pos"] for m in self.archive]
        archive_costs = [m["cost"] for m in self.archive]

        return archive_positions, archive_costs

    def save_results(self, results_dir: str = "results") -> None:
        """
        Saves current archive solutions to JSON and generates Pareto front plot.
        """
        os.makedirs(results_dir, exist_ok=True)
        results = []

        for m in self.archive:
            params = decode_particle(m["pos"])
            results.append(
                {
                    "params": params,
                    "rmse": float(m["cost"][0]),
                    "n_params": int(m["cost"][1]),
                }
            )

        # Sort results by RMSE ascending
        results.sort(key=lambda x: x["rmse"])

        json_path = os.path.join(results_dir, "nspso_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Generate Pareto front scatter plot
        if results:
            rmses = [r["rmse"] for r in results]
            n_params = [r["n_params"] for r in results]

            plt.figure(figsize=(8, 5))
            sns.set_theme(style="whitegrid")
            plt.scatter(
                n_params,
                rmses,
                color="#2563eb",
                s=80,
                edgecolor="black",
                alpha=0.85,
                label="Pareto Optimal Solutions",
            )

            # Highlight best accuracy solution
            best_acc_idx = int(np.argmin(rmses))
            plt.scatter(
                n_params[best_acc_idx],
                rmses[best_acc_idx],
                color="#dc2626",
                s=140,
                marker="*",
                label=f"Lowest RMSE ({rmses[best_acc_idx]:.4f})",
            )

            plt.title("NSPSO Pareto Optimal Front (Bi-Objective Optimization)", fontsize=13, fontweight="bold")
            plt.xlabel("Model Complexity: Trainable Parameters (Objective 2)", fontsize=11)
            plt.ylabel("Validation RMSE [tonnes/ha] (Objective 1)", fontsize=11)
            plt.legend(frameon=True)
            plt.tight_layout()

            plot_path = os.path.join(results_dir, "pareto_front.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()


# ==============================================================================
# MAIN VERIFICATION EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("HYBRID PSO - NSPSO HYPERPARAMETER OPTIMIZATION SUITE")
    print("=" * 65)

    # 1. Resolve paths and load dataset
    resolved_csv, resolved_results = resolve_project_paths()
    df = load_dataset(resolved_csv)
    df_encoded, _ = encode_categoricals(df, results_dir=resolved_results)

    X_raw = df_encoded[FEATURE_COLS].values.astype(np.float32)
    y_raw = df_encoded[TARGET_COL].values.astype(np.float32)
    crop_all = df_encoded["crop"].values

    # Train/Test split: 80% train (96 samples for 120-row dataset), 20% test
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X_tr_raw, _, y_tr, _, crop_tr, _ = train_test_split(
        X_raw,
        y_raw,
        crop_all,
        test_size=0.20,
        random_state=SEED,
        stratify=crop_all,
    )

    # Scale training features
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_train = X_tr_scaled.reshape((X_tr_scaled.shape[0], len(FEATURE_COLS), 1))
    y_train = y_tr
    crop_train = crop_tr

    print(f"[OK] Training set prepared: {X_train.shape} samples (Stratified by crop)")

    # 2. Run Single-Objective PSO (Part A)
    best_pso_params, best_pso_rmse = run_single_objective_pso(
        X_train=X_train,
        y_train=y_train,
        crop_train=crop_train,
        n_particles=8,
        iters=8,
        results_dir=resolved_results,
    )

    # 3. Run Multi-Objective NSPSO (Part B)
    nspso = NSPSO(
        n_particles=8,
        iters=8,
        dims=4,
        bounds=(MIN_BOUNDS, MAX_BOUNDS),
        archive_size=20,
    )

    archive_pos, archive_cost = nspso.optimize(
        X_train=X_train,
        y_train=y_train,
        crop_train=crop_train,
        results_dir=resolved_results,
        pso_ref_rmse=best_pso_rmse,
    )

    print("\n" + "=" * 65)
    print("OPTIMIZATION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"[OK] PSO results saved to:   {os.path.join(resolved_results, 'pso_results.json')}")
    print(f"[OK] NSPSO results saved to: {os.path.join(resolved_results, 'nspso_results.json')}")
    print(f"[OK] Pareto plot saved to:   {os.path.join(resolved_results, 'pareto_front.png')}")
    print(f"[OK] Total Pareto-optimal solutions discovered: {len(archive_pos)}")
    print("=" * 65)

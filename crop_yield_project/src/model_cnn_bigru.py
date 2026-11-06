"""
src/model_cnn_bigru.py
CNN-BiGRU hybrid deep learning architecture for Crop Yield Prediction.

Architecture overview:
  Input: (12, 1) -> 12 agricultural/climatic features treated as 1D sequence
  -> Conv1D: Feature extraction across local feature combinations
  -> MaxPooling1D: Downsampling temporal representation
  -> Bidirectional(GRU): Bidirectional sequential temporal dependency modeling
  -> Dropout: Regularization to prevent overfitting on small datasets
  -> Dense(16, relu): Non-linear dense feature combination
  -> Dense(1, linear): Continuous crop yield prediction (tonnes/ha)
"""

import os
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def count_params(model: tf.keras.Model) -> int:
    """
    Returns the total number of trainable parameters in the Keras model.
    Used as the complexity objective for multi-objective optimization (NSPSO).
    """
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))


def build_cnn_bigru(
    input_shape: Tuple[int, int] = (12, 1),
    conv_filters: int = 32,
    kernel_size: int = 2,
    gru_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Constructs and compiles the hybrid CNN-BiGRU regression model.

    Args:
        input_shape: Tuple of (n_features, n_channels), default (12, 1).
        conv_filters: Number of filters in the Conv1D layer.
        kernel_size: Convolution kernel length.
        gru_units: Number of hidden units in forward and backward GRU cells.
        dropout: Dropout rate for regularization.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model instance.
    """
    # Cast hyperparameters if passed as float or np.int64 from PSO particles
    conv_filters = int(conv_filters)
    kernel_size = int(kernel_size)
    gru_units = int(gru_units)
    dropout = float(dropout)
    learning_rate = float(learning_rate)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_features"),
            layers.Conv1D(
                filters=conv_filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name="conv1d_features",
            ),
            layers.MaxPooling1D(
                pool_size=2,
                padding="same",
                name="maxpool1d",
            ),
            layers.Bidirectional(
                layers.GRU(
                    gru_units,
                    return_sequences=False,
                ),
                name="bidirectional_gru",
            ),
            layers.Dropout(dropout, name="dropout_reg"),
            layers.Dense(16, activation="relu", name="dense_latent"),
            layers.Dense(1, activation="linear", name="yield_output"),
        ],
        name="CNN_BiGRU_Yield_Predictor",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    # Print summary as requested for parameter verification
    model.summary()

    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 10,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains the CNN-BiGRU model with early stopping on validation loss.

    Args:
        model: Compiled Keras model.
        X_train: Training features, shape (n_train, 12, 1).
        y_train: Training targets, shape (n_train,).
        X_val: Validation features, shape (n_val, 12, 1).
        y_val: Validation targets, shape (n_val,).
        epochs: Maximum training epochs (default 100).
        batch_size: Mini-batch size (default 16).
        patience: Epochs with no improvement before early termination (default 10).
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).

    Returns:
        Tuple of (fitted_model, history_object).
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1 if verbose > 0 else 0,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    return model, history


if __name__ == "__main__":
    print("=" * 65)
    print("BUILDING CNN-BiGRU MODEL WITH DEFAULT HYPERPARAMETERS")
    print("=" * 65)

    default_model = build_cnn_bigru(
        input_shape=(12, 1),
        conv_filters=32,
        kernel_size=2,
        gru_units=32,
        dropout=0.2,
        learning_rate=0.001,
    )

    trainable_p = count_params(default_model)
    total_p = default_model.count_params()

    print("\n" + "=" * 65)
    print("MODEL PARAMETER VERIFICATION")
    print("=" * 65)
    print(f"[OK] Total parameters:     {total_p:,}")
    print(f"[OK] Trainable parameters: {trainable_p:,}")
    print("=" * 65)

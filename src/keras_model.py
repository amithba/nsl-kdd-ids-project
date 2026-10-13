"""
keras_model.py (OPTIONAL)
--------------------------
Drop-in Keras/TensorFlow version of model.py, for environments that have
TensorFlow installed (e.g. Google Colab). Mirrors the same 128 -> 64 -> 5
architecture as the scikit-learn MLP used by default in this project.

Usage:
    python train.py --backend keras

Requires: pip install tensorflow
"""
import numpy as np


def build_keras_model(input_dim: int, n_classes: int = 5):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_keras_model(X_train, y_train, X_test, y_test, n_classes=5, epochs=30, batch_size=256):
    from tensorflow import keras

    model = build_keras_model(X_train.shape[1], n_classes)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return model, history, y_pred

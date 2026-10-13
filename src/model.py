"""
model.py
--------
Defines the Artificial Neural Network (Multi-Layer Perceptron) used for
multi-class attack classification: Normal / DoS / Probe / R2L / U2R.

Implemented with scikit-learn's MLPClassifier so the project runs with no
extra GPU/deep-learning framework. If TensorFlow/Keras is available in your
environment, an equivalent Keras architecture is given in keras_model.py as
a drop-in alternative (same layer sizes, same results-reporting code).
"""
from sklearn.neural_network import MLPClassifier


def build_model(random_state: int = 42) -> MLPClassifier:
    """
    Architecture: input -> 128 (ReLU) -> 64 (ReLU) -> 5 (softmax, via
    sklearn's multinomial output). Adam optimizer, L2 regularization,
    early stopping on a held-out validation split.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,            # L2 regularization
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=random_state,
        verbose=False,
    )

"""
train.py
--------
End-to-end pipeline: load data -> preprocess -> train ANN -> evaluate ->
save plots, confusion matrix, and a metrics summary to ../outputs/.

Usage:
    python train.py                         # uses sklearn MLP (default)
    python train.py --backend keras         # uses TensorFlow/Keras (if installed)
    python train.py --train data/KDDTrain+.txt --test data/KDDTest+.txt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

from preprocess import prepare_datasets
from model import build_model
from schema import CLASSES

ROOT = Path(__file__).resolve().parent.parent


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(history_dict, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_dict["loss"], label="training loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "KDDTrain+_sample.txt"))
    parser.add_argument("--test", default=str(ROOT / "data" / "KDDTest+_sample.txt"))
    parser.add_argument("--backend", choices=["sklearn", "keras"], default="sklearn")
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Loading data:\n  train = {args.train}\n  test  = {args.test}")
    X_train, X_test, y_train, y_test, feature_names, le = prepare_datasets(
        Path(args.train), Path(args.test)
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    if args.backend == "sklearn":
        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        history_dict = {"loss": model.loss_curve_}
        train_acc = accuracy_score(y_train, model.predict(X_train))
    else:
        from keras_model import train_keras_model
        model, history, y_pred = train_keras_model(
            X_train, y_train, X_test, y_test, n_classes=len(CLASSES)
        )
        history_dict = {"loss": history.history["loss"]}
        train_acc = max(history.history["accuracy"])

    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))

    # Binary (Normal vs Attack) view -- mirrors the "is there an attack at
    # all" framing used in the reference report.
    y_test_bin = (y_test != CLASSES.index("Normal")).astype(int)
    y_pred_bin = (y_pred != CLASSES.index("Normal")).astype(int)
    binary_acc = accuracy_score(y_test_bin, y_pred_bin)

    print("\n=== Results ===")
    print(f"Train accuracy (multi-class): {train_acc:.4f}")
    print(f"Test accuracy  (multi-class): {test_acc:.4f}")
    print(f"Test macro F1-score:          {macro_f1:.4f}")
    print(f"Test accuracy (Normal/Attack binary view): {binary_acc:.4f}")
    print("\nClassification report:\n", report)

    plot_confusion_matrix(cm, CLASSES, outdir / "confusion_matrix.png")
    plot_training_curve(history_dict, outdir / "training_loss.png")

    metrics = {
        "backend": args.backend,
        "train_accuracy": train_acc,
        "test_accuracy_multiclass": test_acc,
        "test_accuracy_binary": binary_acc,
        "test_macro_f1": macro_f1,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "classes": CLASSES,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nSaved plots + metrics to {outdir}/")


if __name__ == "__main__":
    main()

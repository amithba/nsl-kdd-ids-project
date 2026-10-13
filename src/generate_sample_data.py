"""
generate_sample_data.py
------------------------
Generates a small SYNTHETIC dataset that mimics the structure and class
proportions of NSL-KDD (41 features + label). This is NOT the real dataset.

Purpose: let you run and sanity-check the whole pipeline (preprocessing,
training, evaluation, plotting) instantly, without downloading anything.

For your real submission you MUST replace this with the actual NSL-KDD
files (KDDTrain+.txt / KDDTest+.txt) -- see README.md, Step 2, for the
download links. Re-run train.py after swapping the data in data/.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from schema import COLUMN_NAMES, CLASSES

# A few representative fine-grained attack names per category, so the
# synthetic file looks like a real NSL-KDD file (which stores specific
# attack names, e.g. "neptune", not the coarse category "DoS").
ATTACK_NAMES_BY_CLASS = {
    "Normal": ["normal"],
    "DoS": ["neptune", "smurf", "back", "teardrop", "pod"],
    "Probe": ["ipsweep", "portsweep", "nmap", "satan"],
    "R2L": ["guess_passwd", "warezmaster", "ftp_write", "imap"],
    "U2R": ["buffer_overflow", "rootkit", "loadmodule", "perl"],
}

RNG = np.random.default_rng(42)

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "telnet", "private", "domain_u", "finger", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH"]

# Roughly mirrors the class imbalance reported for NSL-KDD (Table 3 of the
# reference report): Normal and DoS dominate, R2L/U2R are rare.
CLASS_WEIGHTS_TRAIN = {"Normal": 0.53, "DoS": 0.36, "Probe": 0.09, "R2L": 0.009, "U2R": 0.001}
CLASS_WEIGHTS_TEST = {"Normal": 0.43, "DoS": 0.33, "Probe": 0.11, "R2L": 0.12, "U2R": 0.01}


def _make_split(n_rows: int, class_weights: dict) -> pd.DataFrame:
    keys = list(class_weights.keys())
    probs = np.array(list(class_weights.values()), dtype=float)
    probs = probs / probs.sum()
    labels = RNG.choice(keys, size=n_rows, p=probs)

    # Give each class a distinct underlying signal so a model can actually
    # learn separable patterns (this is what makes the synthetic run
    # meaningful for pipeline-testing purposes).
    class_shift = {"Normal": 0.0, "DoS": 2.5, "Probe": 1.2, "R2L": -1.5, "U2R": -3.0}

    n_numeric = len(COLUMN_NAMES) - 3 - 2  # minus categorical, label, difficulty
    rows = []
    for lbl in labels:
        shift = class_shift[lbl]
        base = RNG.normal(loc=shift, scale=1.0, size=n_numeric)
        base = np.abs(base)  # NSL-KDD numeric features are non-negative
        rows.append(base)
    numeric_data = np.vstack(rows)

    df = pd.DataFrame(numeric_data, columns=[c for c in COLUMN_NAMES
                                              if c not in ("protocol_type", "service", "flag",
                                                            "label", "difficulty")])

    # Rates should look like rates (0-1)
    rate_cols = [c for c in df.columns if "rate" in c]
    for c in rate_cols:
        df[c] = (df[c] / (df[c].max() + 1e-6)).clip(0, 1)

    df["protocol_type"] = RNG.choice(PROTOCOLS, size=n_rows, p=[0.6, 0.3, 0.1])
    df["service"] = RNG.choice(SERVICES, size=n_rows)
    df["flag"] = RNG.choice(FLAGS, size=n_rows, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05])
    # Store a specific fine-grained attack name per row (as real NSL-KDD
    # files do), drawn from the row's coarse category.
    df["label"] = [RNG.choice(ATTACK_NAMES_BY_CLASS[lbl]) for lbl in labels]
    df["difficulty"] = RNG.integers(1, 21, size=n_rows)

    return df[COLUMN_NAMES]


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    train_df = _make_split(12000, CLASS_WEIGHTS_TRAIN)
    test_df = _make_split(3000, CLASS_WEIGHTS_TEST)

    train_path = out_dir / "KDDTrain+_sample.txt"
    test_path = out_dir / "KDDTest+_sample.txt"
    train_df.to_csv(train_path, header=False, index=False)
    test_df.to_csv(test_path, header=False, index=False)

    print(f"Wrote {len(train_df)} synthetic training rows -> {train_path}")
    print(f"Wrote {len(test_df)} synthetic test rows -> {test_path}")
    print("\nClass distribution (train):")
    print(train_df["label"].value_counts())
    print("\nClass distribution (test):")
    print(test_df["label"].value_counts())


if __name__ == "__main__":
    main()

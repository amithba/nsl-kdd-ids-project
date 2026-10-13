
# Network Intrusion / Security Attack Detection using a Neural Network (NSL-KDD)

**Course:** UE24CS352A — Machine Learning, Mini-Project
**Team:** Team No.4,AMITH B A - PES1UG24AM032,ASHWIN ANAND - PES1UG24AM057

A multi-class Artificial Neural Network (ANN) that classifies network
connection records from the **NSL-KDD** dataset as `Normal`, `DoS`, `Probe`,
`R2L`, or `U2R`, following the same approach as the reference CS229 paper
*"Using Neural Network for Security Attacks Detection"* (data pipeline,
41→122 feature one-hot expansion, feed-forward NN, train/test accuracy
comparison), adapted to run end-to-end offline with scikit-learn.


## 1. Project Structure

.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # put KDDTrain+.txt / KDDTest+.txt here
│   ├── KDDTrain+_sample.txt      # small synthetic file for a quick smoke test
│   └── KDDTest+_sample.txt
├── src/
│   ├── schema.py                 # 41-feature column names + attack→category map
│   ├── generate_sample_data.py   # makes the synthetic *_sample.txt files
│   ├── preprocess.py             # load, label-map, one-hot encode, scale
│   ├── model.py                  # ANN (sklearn MLPClassifier) architecture
│   ├── keras_model.py            # optional TensorFlow/Keras version
│   └── train.py                  # full train + evaluate + plot pipeline
├── outputs/                      # generated: confusion matrix, loss curve, metrics.json
└── report/
    └── Mini_Project_Report.pdf   # 2-page write-up for submission

## 2. Getting the Real Dataset

The repo ships only a small **synthetic** sample (`data/*_sample.txt`) so you
can test the pipeline instantly with no downloads. For your actual
submission, download the real NSL-KDD files and drop them into `data/`:

- Official source (University of New Brunswick): https://www.unb.ca/cic/datasets/nsl.html
- Kaggle mirror: https://www.kaggle.com/datasets/hassan06/nslkdd
- Common GitHub mirror (plain `KDDTrain+.txt` / `KDDTest+.txt`): search
  "NSL-KDD KDDTrain+.txt github" — several research repos mirror the exact
  files, e.g. `defcom17/NSL_KDD`.

You need `KDDTrain+.txt` and `KDDTest+.txt` (the full, non-21%, non-20-percent
versions are fine too — just point `--train`/`--test` at whichever files you
use).

## 3. Setup

bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## 4. Running the Pipeline

bash
cd src

# (a) Quick smoke test on the bundled synthetic sample data:
python generate_sample_data.py     # only needed once, already included
python train.py

# (b) Real run, once you've placed the real files in ../data/:
python train.py --train ../data/KDDTrain+.txt --test ../data/KDDTest+.txt


This prints train/test accuracy, macro F1, and a full classification report
to the console, and writes to `../outputs/`:
- `confusion_matrix.png`
- `training_loss.png`
- `metrics.json`
- `classification_report.txt`

Re-run with the real data and **use those numbers** (not the synthetic-sample
numbers) in your final report/slides.

### Optional: Keras/TensorFlow backend
If you have TensorFlow installed (e.g. in Google Colab):
bash
pip install tensorflow
python train.py --backend keras --train ../data/KDDTrain+.txt --test ../data/KDDTest+.txt


## 5. Adding This Project to GitHub

From inside the project folder:

bash
# 1. Initialize git (skip if already a repo)
git init

# 2. Stage and commit everything
git add .
git commit -m "Initial commit: NSL-KDD intrusion detection ANN"

# 3. Create a new PRIVATE repository on GitHub
#    (via the website: github.com/new -> set Visibility = Private)
#    or via GitHub CLI:
gh repo create your-team-nsl-kdd-ids --private --source=. --remote=origin

# 4. If you created the repo on the website instead, link it manually:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main

# 5. Push
git push -u origin main

**Share the repo with faculty/TAs (required by the assignment guidelines):**
1. Open the repo on GitHub → **Settings** → **Collaborators and teams**.
2. Click **Add people**, enter the faculty/TA's GitHub username or email.
3. They'll get an invite email — this satisfies "repository must be shared
   with the faculty and TAs."

**Keep it updated:** every time you make changes,
bash
git add .
git commit -m "describe what changed"
git push

## 6. Deliverables Checklist (per assignment guidelines)

- [x] Private GitHub repo, shared with faculty/TAs, with this README
- [x] Source code (`src/`)
- [ ] Two-page PDF write-up → see `report/Mini_Project_Report.pdf`
      (fill in your team number, names, SRNs, and **re-run on the real
      dataset** to replace the example numbers before submitting)
- [ ] Slide deck for the review session (not included here — ask if you'd
      like one built from this same content)
- [ ] Live demo of `python train.py` during the review

## 7. Notes on the Approach

- **Features:** all 41 NSL-KDD features; the 3 categorical ones
  (`protocol_type`, `service`, `flag`) are one-hot encoded, numeric features
  are standardized (zero mean, unit variance) — same preprocessing idea as
  the reference report.
- **Labels:** the ~39 fine-grained attack names are mapped to 5 classes
  (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`) using the standard mapping table
  from Tavallaee et al. (2009).
- **Model:** feed-forward ANN, 2 hidden layers (128, 64 units, ReLU), Adam
  optimizer, L2 regularization, early stopping on a validation split —
  functionally equivalent to the single-hidden-layer network in the
  reference paper but with modern defaults (Adam instead of plain
  mini-batch gradient ascent, early stopping instead of a fixed epoch count).
- **Why accuracy alone isn't enough:** NSL-KDD's classes are heavily
  imbalanced (R2L/U2R are rare) and the test set contains attack subtypes
  never seen in training, so `train.py` also reports macro F1-score and a
  full confusion matrix, not just accuracy.

# nsl-kdd-ids-project
https://github.com/amithba/nsl-kdd-ids-project

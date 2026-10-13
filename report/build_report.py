"""
build_report.py
----------------
Generates the two-page PDF write-up required by the assignment guidelines
(problem statement, dataset details, approach, implementation overview,
conclusions) from a reportlab template.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib import colors

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], fontSize=16, spaceAfter=4))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=10,
                           alignment=TA_CENTER, textColor=colors.grey, spaceAfter=12))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, leading=13))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.grey))

doc = SimpleDocTemplate(
    "Mini_Project_Report.pdf", pagesize=letter,
    topMargin=0.55 * inch, bottomMargin=0.55 * inch,
    leftMargin=0.65 * inch, rightMargin=0.65 * inch,
)

story = []

story.append(Paragraph("Network Intrusion Detection using a Neural Network", styles["TitleCenter"]))
story.append(Paragraph("UE24CS352A — Machine Learning Mini-Project Write-up &nbsp;|&nbsp; "
                        "Team &lt;No.&gt; &nbsp;|&nbsp; &lt;Names, SRNs&gt;", styles["SubTitle"]))

story.append(Paragraph("1. Problem Statement", styles["H2"]))
story.append(Paragraph(
    "Network intrusion detection systems (IDS) must distinguish normal traffic from an "
    "increasingly diverse set of cyberattacks, while generalizing to attack variants not "
    "seen during training. This project builds a supervised Artificial Neural Network (ANN) "
    "that classifies each network connection record into one of five categories — "
    "<b>Normal, DoS, Probe, R2L (Remote-to-Local), or U2R (User-to-Root)</b> — based on 41 "
    "traffic/connection features, and evaluates how well the model generalizes to a test set "
    "that contains attack subtypes absent from training.", styles["Body"]))

story.append(Paragraph("2. Dataset", styles["H2"]))
story.append(Paragraph(
    "We use the <b>NSL-KDD</b> dataset (Tavallaee et al., 2009), a cleaned, non-redundant "
    "revision of the original KDD Cup 1999 dataset widely used as an IDS benchmark. Each "
    "record has 41 features: basic TCP/IP connection attributes (e.g. duration, protocol_type, "
    "service, flag, src_bytes, dst_bytes), content features (e.g. num_failed_logins, root_shell), "
    "and traffic features computed over 2-second windows (e.g. count, serror_rate, "
    "same_srv_rate). Each record is labelled with one of ~39 specific attack names, which we "
    "map to four standard attack categories plus Normal, following the mapping table published "
    "with the dataset. The training set is dominated by Normal and DoS records, with R2L and "
    "U2R making up less than 1% of records combined — a severe class imbalance that is the "
    "central challenge of this task. The test set additionally contains several attack subtypes "
    "that never appear in training, testing the model's ability to generalize.", styles["Body"]))

story.append(Paragraph("3. Approach", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Preprocessing:</b> the three categorical features "
                        "(protocol_type, service, flag) are one-hot encoded; all numeric "
                        "features are standardized to zero mean / unit variance using "
                        "statistics from the training set only.", styles["Body"])),
    ListItem(Paragraph("<b>Label mapping:</b> ~39 fine-grained attack names → "
                        "{Normal, DoS, Probe, R2L, U2R} using the standard NSL-KDD category "
                        "table.", styles["Body"])),
    ListItem(Paragraph("<b>Model:</b> a feed-forward ANN with two hidden layers "
                        "(128 and 64 ReLU units), trained with the Adam optimizer, L2 "
                        "weight regularization, and early stopping on a held-out validation "
                        "split to prevent overfitting.", styles["Body"])),
    ListItem(Paragraph("<b>Evaluation:</b> multi-class accuracy and macro-averaged F1-score "
                        "(which, unlike raw accuracy, is not dominated by the majority "
                        "Normal/DoS classes), a full confusion matrix, and a binary "
                        "Normal-vs-Attack view since that is the operationally critical "
                        "decision for an IDS.", styles["Body"])),
], bulletType="bullet", leftIndent=12))

story.append(Paragraph("4. Implementation Overview", styles["H2"]))
story.append(Paragraph(
    "The pipeline is implemented in Python (pandas, scikit-learn, matplotlib) as a small, "
    "reproducible repository: <font face='Courier'>schema.py</font> defines the 41-feature "
    "column layout and attack-category mapping; <font face='Courier'>preprocess.py</font> "
    "loads the raw files, applies label mapping, one-hot encoding and scaling; "
    "<font face='Courier'>model.py</font> defines the ANN "
    "(<font face='Courier'>MLPClassifier</font>, 128→64→5); and "
    "<font face='Courier'>train.py</font> orchestrates training, evaluation and plot "
    "generation. An optional Keras/TensorFlow implementation "
    "(<font face='Courier'>keras_model.py</font>) with the same architecture is included for "
    "environments where TensorFlow is available. After one-hot encoding, the 41 raw features "
    "expand to 55 input dimensions on the data used for this run.", styles["Body"]))

story.append(Paragraph("5. Results", styles["H2"]))
story.append(Paragraph(
    "Figures below are from an end-to-end run of this pipeline (see table for sample sizes). "
    "<i>Note: re-run <font face='Courier'>train.py</font> on the full real NSL-KDD files before "
    "final submission and update these numbers accordingly — see the project README.</i>",
    styles["Body"]))
story.append(Spacer(1, 4))

table_data = [
    ["Metric", "Value"],
    ["Train samples / Test samples", "12,000 / 3,000"],
    ["Input features (after one-hot encoding)", "55"],
    ["Train accuracy (multi-class)", "98.5%"],
    ["Test accuracy (multi-class)", "86.6%"],
    ["Test macro F1-score", "0.54"],
    ["Test accuracy (Normal vs Attack, binary view)", "97.2%"],
]
t = Table(table_data, colWidths=[3.4 * inch, 1.6 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
story.append(Spacer(1, 8))

img_row = Table([[
    Image("confusion_matrix.png", width=2.7 * inch, height=2.25 * inch),
    Image("training_loss.png", width=2.7 * inch, height=1.8 * inch),
]], colWidths=[3.1 * inch, 3.1 * inch])
img_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story.append(img_row)

story.append(Paragraph(
    "As expected from the dataset's class imbalance, Normal and DoS are classified with "
    "very high precision/recall, Probe is reasonable, while R2L and U2R — which make up "
    "well under 1% of the training set — are recalled poorly despite high precision when "
    "predicted. This mirrors the finding in prior work on this dataset (e.g. the CS229 "
    "report this project follows) that raw multi-class accuracy can look strong while "
    "minority attack classes remain hard to detect, which is why macro F1 and the binary "
    "Normal-vs-Attack view are reported alongside accuracy.", styles["Body"]))

story.append(Paragraph("6. Conclusions", styles["H2"]))
story.append(Paragraph(
    "A feed-forward neural network trained on NSL-KDD's 41 engineered features can "
    "reliably separate Normal traffic from the dominant DoS/Probe attack categories, but "
    "struggles with the rare R2L and U2R classes without additional handling (e.g. class "
    "weighting, oversampling, or threshold tuning) — consistent with published results on "
    "this benchmark. For an operational IDS, the binary Normal-vs-Attack accuracy is the "
    "most relevant number, since even a misclassified attack category still correctly "
    "triggers an alert. Future work could explore class-weighted loss, SMOTE-style "
    "oversampling of R2L/U2R, or ensembling with signature-based detection for the rarest "
    "attack types.", styles["Body"]))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "Code, data instructions, and reproduction steps: see the accompanying GitHub repository "
    "README.", styles["Small"]))

doc.build(story)
print("Wrote Mini_Project_Report.pdf")

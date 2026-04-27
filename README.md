# Fraud Detection using Machine Learning

End-to-end machine learning project focused on detecting fraudulent credit card transactions in a highly imbalanced dataset.

The goal is not only to train a classifier, but to show a realistic fraud-prevention workflow: data understanding, preprocessing, imbalance handling, model comparison, metric selection, and reproducible experimentation.

## Business Context

Fraud detection problems are usually highly imbalanced: fraudulent cases are rare, but the business impact of missing them can be very high. For that reason, this project prioritizes metrics such as recall, precision, F1-score, PR-AUC and ROC-AUC instead of relying only on accuracy.

## Objectives

- Detect fraudulent transactions using supervised machine learning.
- Handle severe class imbalance.
- Compare baseline and advanced models.
- Evaluate models using fraud-oriented metrics.
- Build a clean and reproducible project structure.
- Prepare the codebase for future API deployment or batch scoring.

## Dataset

This project is designed to work with the public Kaggle dataset:

**Credit Card Fraud Detection**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains transactions made by European cardholders. Features are anonymized PCA components, plus `Time`, `Amount`, and `Class`, where:

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

> The dataset is not included in this repository due to size and licensing. Download `creditcard.csv` from Kaggle and place it inside the `data/` folder.

## Project Structure

```text
fraud-detection-ml/
├── data/                         # Local dataset folder, ignored by Git
├── models/                       # Trained models, ignored by Git
├── notebooks/
│   └── fraud_detection_experiment.ipynb
├── reports/
│   └── figures/                  # Generated plots, ignored by Git
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Modeling Approach

The project includes:

1. Data loading and validation.
2. Stratified train/test split.
3. Preprocessing pipeline with feature scaling.
4. Baseline model: Logistic Regression.
5. Advanced model: Random Forest.
6. Optional imbalance handling using class weights.
7. Fraud-oriented evaluation with:
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC / Average Precision
   - Confusion matrix

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/olcesefacundo97/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `creditcard.csv` from Kaggle and place it here:

```text
data/creditcard.csv
```

### 5. Train and evaluate models

```bash
python -m src.train --data-path data/creditcard.csv
```

## Example Output

The training script prints a comparison between models using fraud-oriented metrics.

Example metrics table:

```text
model                 precision    recall    f1_score    roc_auc    pr_auc
Logistic Regression  0.86         0.62      0.72        0.96       0.75
Random Forest        0.94         0.79      0.86        0.98       0.84
```

> The exact values may vary depending on environment, split and configuration.

## Why These Metrics Matter

In fraud detection, accuracy can be misleading because most transactions are legitimate. A model can achieve very high accuracy by predicting almost every transaction as non-fraudulent, while still missing the most important cases.

For that reason:

- **Recall** helps measure how many fraudulent cases are detected.
- **Precision** helps measure how many predicted fraud cases are actually fraud.
- **PR-AUC** is especially useful for imbalanced classification problems.
- **ROC-AUC** helps evaluate ranking quality across thresholds.

## Future Improvements

- Add SMOTE or other resampling strategies.
- Add XGBoost or LightGBM.
- Add threshold optimization based on business cost.
- Save trained models with joblib.
- Build a FastAPI service for real-time scoring.
- Add MLflow for experiment tracking.
- Add Docker support.
- Add CI checks with GitHub Actions.

## CV Description

**Fraud Detection using Machine Learning**

- Built an end-to-end fraud detection project using Python and Scikit-Learn.
- Trained and compared classification models on a highly imbalanced dataset.
- Applied preprocessing pipelines, class weighting and fraud-oriented evaluation metrics.
- Prioritized recall, precision and PR-AUC to reflect real fraud-prevention trade-offs.
- Structured the repository for reproducibility and future production deployment.

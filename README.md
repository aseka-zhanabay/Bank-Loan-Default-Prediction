# Bank Loan Default Prediction Using Machine Learning

## Overview

This project explores the application of supervised machine learning to bank loan default prediction. Using a loan dataset described in the notebook as the East Midland Bank (EMB) dataset, it investigates borrower and loan characteristics, prepares data for classification, and compares several algorithms for distinguishing fully paid loans from charged-off loans.

The notebook was developed as part of an assessment for **BFIN3003 — Artificial Intelligence and Financial Application**.

## Objectives

- Explore loan characteristics and the distribution of default outcomes.
- Clean the dataset and prepare numerical features for modelling.
- Train and compare classification algorithms.
- Tune Random Forest hyperparameters using cross-validation.
- Evaluate predictions using multiple classification metrics and visualizations.

## Methods

**Exploratory analysis:** missing-value checks, duplicate removal, default distribution, feature distributions, and a correlation heatmap.

**Models implemented:**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The notebook also imports LightGBM, but does not include it in the final model comparison.

**Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, and ROC curves. Random Forest hyperparameters are explored using `GridSearchCV` with three-fold cross-validation.

## Tech Stack

Python, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, and LightGBM (imported).

## Repository Contents

```text
Bank-Loan-Default-Prediction/
├── Assesment2_2_2.ipynb   # Analysis and modelling notebook
├── README.md              # Project documentation
└── east_midlands.csv      # Required input data (add only if sharing is permitted)
```

## Getting Started

1. Clone the repository and open the project folder.
2. Create a Python environment and install the dependencies:

   ```bash
   pip install notebook pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
   ```
3. Place `east_midlands.csv` in the same directory as the notebook, if you have permission to use and share the data.
4. Launch Jupyter Notebook and open `Assesment2_2_2.ipynb`.

   ```bash
   jupyter notebook
   ```

**Current reproducibility note:** The uploaded notebook needs some fixes before it can be run reliably from top to bottom. In particular, the EDA encodes categorical columns (including `loan_status`) before a later step filters `loan_status` by its original string labels. The ROC curve section also uses `roc_curve` without explicitly importing it. Resolve these issues and rerun the notebook before presenting numerical results.

## Outputs

The notebook includes code to generate:
- `default_distribution.png`
- `correlation_heatmap.png`
- `confusion_matrices.png`
- `roc_curves.png`

It also constructs a comparison table for model Accuracy and ROC-AUC. This README does not claim a winning model or numerical performance until the complete pipeline is rerun and verified.

## Limitations and Future Improvements

- Fix notebook execution order and ensure reproducible preprocessing.
- Fit preprocessing transformations on training data only to avoid leakage.
- Consider stratified train/test splitting and class imbalance.
- Compare threshold-dependent metrics alongside ROC-AUC, particularly recall for defaults.
- Add clear results, interpretation, and model limitations after validation.

## Attribution

The notebook currently identifies **Assem Zhanabay** as the student author. Preserve accurate attribution and confirm permission before publishing or presenting the work as your own.

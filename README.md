# Online Shoppers Purchasing Intention — predicting a purchase

A machine learning project: predicting whether a website visitor will complete a purchase (binary classification), based on their behavior during the session.

## Dataset

**[Online Shoppers Purchasing Intention Dataset (UCI)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)**
The file `online_shoppers_intention.csv` is included in this repository.

- 12,330 user sessions, 17 features + target variable `Revenue` (bool — whether a purchase was made)
- Features: number and duration of visits to different page types (Administrative / Informational / ProductRelated), `BounceRates`, `ExitRates`, `PageValues`, `Month`, `VisitorType`, `Weekend`, technical fields (`OperatingSystems`, `Browser`, `Region`, `TrafficType`), etc.
- No missing values

**Why this dataset:** real tabular data with a mix of numerical and categorical features, a clear class imbalance (~85% / 15%), and potential multicollinearity (`BounceRates` / `ExitRates`) — in other words, the data itself sets up meaningful hypotheses to test, not just a generic "does the model work" question.

## Hypotheses (formed before the analysis)

1. Random Forest will beat logistic regression on ROC-AUC — the relationship between features and a purchase may turn out to be nonlinear.
2. An unrestricted decision tree will overfit: train F1 macro will keep rising to 1.0, while test F1 macro will be lower than for a depth-limited tree.
3. `PageValues` will turn out to be the most important feature by random forest feature importance.
4. Dropping one of two strongly correlated features (`BounceRates`, correlation with `ExitRates` > 0.9) won't meaningfully hurt logistic regression's AUC.
5. Ridge (L2) regularization will beat unregularized logistic regression on test AUC.
6. F1 macro will be noticeably lower than accuracy because of the class imbalance.
7. Selecting the top-10 features via SelectKBest won't meaningfully hurt quality compared to the full feature set (~30 features after one-hot encoding).

## Results (summary)

| # | Hypothesis | Result |
|---|----------|-----------|
| 1 | RF > LogReg on AUC | ✅ Confirmed (AUC ~0.92 vs ~0.89) |
| 2 | Unrestricted tree overfits (F1 macro) | ✅ Confirmed (train F1 macro = 1.0, test F1 macro ≈0.73 vs ≈0.78 at depth 6-8) |
| 3 | PageValues is the top feature | ✅ Confirmed (importance ~0.36, several times higher than any other feature) |
| 4 | Dropping BounceRates won't hurt AUC | ✅ Confirmed (difference ~0.0003) |
| 5 | Ridge beats unregularized LogReg | ⚠️ More likely not confirmed (difference is within noise; GridSearch actually picked a weak-regularization `C=100` as optimal) |
| 6 | F1 macro is lower than accuracy | ✅ Confirmed (10-18 point gap across all models) |
| 7 | Top-10 features don't hurt quality | ✅ Confirmed (AUC drops by only ~0.002-0.003) |

Detailed reasoning for each hypothesis is in `research.ipynb`, section "Conclusions per hypothesis."

**Best model:** Random Forest (200 trees, tuned via GridSearchCV) — accuracy ≈ 0.90, F1 macro ≈ 0.79, ROC-AUC ≈ 0.92 on the test set.

## What's in `research.ipynb`

- EDA: distributions, correlations, class imbalance, categorical features, and an explicit check for **nonlinearity** between features and the target (purchase rate per bin)
- Preprocessing: `StandardScaler` for numerical features, `OneHotEncoder` for categorical, stratified train/test split
- Models: logistic regression (unregularized, L1, L2), decision tree, random forest
- Metrics: accuracy, F1 macro, ROC-AUC (not just accuracy — due to the class imbalance)
- Cross-validation (`StratifiedKFold`, 5 folds)
- Hyperparameter tuning via `GridSearchCV` — for the decision tree, logistic regression, and random forest
- Feature selection via `SelectKBest`
- A business-oriented classification threshold analysis (precision/recall/F1 vs. threshold), showing how the choice of threshold trades off catching more buyers against more false positives
- Every hypothesis tested with an experiment, with a written conclusion

## Repository structure

```
├── README.md
├── research.ipynb                  # full analysis: EDA, preprocessing, models, tuning, conclusions
├── online_shoppers_intention.csv   # dataset
└── requirements.txt                # exact package versions used to run the notebook
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook research.ipynb
```

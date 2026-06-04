# mdq-card-classifier
Detecting hidden entrepreneurs among consumer bank cards using PU Learning
# MDQ Card Classifier — Finding Hidden Entrepreneurs

## Problem

Banks issue consumer cards with spending restrictions. Some cardholders 
use these cards for business purposes — violating product terms and creating 
regulatory risk. These "hidden entrepreneurs" are hard to detect because 
they have no business label: they look like regular consumers on paper.

## Approach

Standard binary classification fails here: it assumes all consumer-labeled 
cards are truly consumers. This is wrong — some are unlabeled businesses.

I framed this as a **Positive-Unlabeled (PU) Learning** problem and compared 
three models:

| Model | Why |
|-------|-----|
| Logistic Regression | Interpretable baseline — explainable coefficients |
| Naive LightGBM | Strong classifier, but assumes clean labels |
| **PU Bagging** (Mordelet & Vert, 2014) | Correct formulation — treats consumer cards as unlabeled, not negative |

## Key Design Decisions

- **No data leakage**: clipping bounds and rare MCC thresholds computed 
  on train transactions only, then applied to val/test
- **Dynamic threshold**: top-N candidates instead of fixed probability 
  cutoff — operationally more practical
- **Proxy precision**: no ground truth for consumer cards, so quality 
  is evaluated via B2B behavioral heuristics (MCC category, transaction 
  patterns, average amounts)

## Results

This project was submitted to the **MDQ Case Competition** (fintech track).

All three models achieve near-perfect metrics on labeled data — this is 
expected given the competition dataset, where known business and consumer 
cards are behaviorally distinct by design. The real challenge is not 
separating labeled classes, but finding **unlabeled businesses hiding 
in the consumer pool**.

The key metric is **Precision@100** on unlabeled consumer cards — 
how many of the top-100 flagged cards genuinely look like businesses 
by behavioral heuristics. PU Learning outperforms naive LightGBM here 
because it never penalizes a card for looking like a business.

## Stack

`Python` `LightGBM` `scikit-learn` `pandas` `numpy` `category_encoders` `matplotlib`

## Structure
├── MDQ_card_classifier_v17_clean.ipynb
├── .gitignore
├── LICENSE
└── README.md

## Data

Sourced from MDQ Case Competition. Not included due to competition terms.

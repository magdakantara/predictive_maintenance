# Predictive Maintenance – Failure Prediction

## Project Overview

This project focuses on predicting machine failures using historical operational data. The objective is to identify rare failure events in advance, enabling proactive maintenance and reducing unexpected downtime.

The dataset contains time-dependent sensor measurements and operational features, with a binary target variable indicating machine failure.

---

## Problem Characteristics

A key challenge in this task is the **extreme class imbalance**:

- Failure events represent **less than 0.1% of observations**
- The vast majority of records correspond to normal operation

This reflects realistic industrial environments, where failures are rare but operationally critical.  
As a result, model evaluation emphasizes **Precision–Recall performance** rather than accuracy.

---

## Project Structure
├── data_cleaning_engineering.py # Preprocessing & feature engineering, adding features like rolling mean for more insights trying to combat the imbalance.
├── modeling.py # Model training & evaluation
├── EDA notebooks/ # Exploratory analysis one for raw unprocessed data and one for the clean one
├── datasets/ # Data files of the processed csv files and some figures
└── archive/ # data folder


---

## Methodology

### 1. Data Preparation
- Temporal train/test split to avoid leakage  
- Feature engineering from operational variables  
- Handling missing values  
- Scaling where appropriate  

### 2. Modeling Approaches
- Time aware cross validation on the train file
- Logistic Regression (with class balancing)
- Random Forest
- HistGradientBoosting 

### 3. Evaluation Strategy

Due to the severe imbalance, performance is assessed mainly using:

- **Precision–Recall AUC (PR-AUC)**

Accuracy is not considered informative in this setting.

---

## Key Insight

Because failure events are extremely rare, the task is inherently difficult. Even well-calibrated models may exhibit modest PR-AUC values, which is expected in ultra-imbalanced industrial datasets. The focus of this project is therefore on:

- Understanding model behavior under extreme imbalance  
- Evaluating robustness across folds  
- Comparing detection trade-offs rather than maximizing a single metric  

---

## Tools & Libraries

- Python
- pandas / numpy
- scikit-learn
- matplotlib / seaborn

---

## Future Improvements

- Advanced resampling strategies (SMOTE, undersampling)
---

**Author:** Maria Magdalini Kantara  
BSc Data Science & AI – TU/e


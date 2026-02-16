"""
Preprocessing – Predictive Maintenance Dataset

This script cleans and engineers features from the raw dataset
(12 columns: date, device, failure, metric1–metric9).

Steps:
- Convert date to datetime
- Remove duplicates
- Sort by device and date
- Handle missing values (device-level forward/backward fill + median fallback)
- Winsorize extreme outliers (1%–99%)
- Log-transform highly skewed metrics
- Create time features (year, month, day, dayofweek)
- Create rolling mean and std features (3, 7, 14-day windows)
- Create deviation-from-baseline (delta) features

Result:
12 columns → ~79 columns after feature engineering.

Output:
Saves proces sed dataset to preprocessed.csv.
"""

import pandas as pd
import numpy as np


df = pd.read_csv("archive\predictive_maintenance_dataset.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

#  to maintain a list of metric columns once and reuse it everywhere
metrics = [c for c in df.columns if c.startswith("metric")]

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Failure counts:\n", df["failure"].value_counts(dropna=False))


# 1) Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped duplicates: {before - len(df)}")


# 2) Sort data by device and date
df = df.sort_values(["device", "date"]).reset_index(drop=True)


# 3) Handle missing values


# Drop rows missing essential identifiers/target
df = df.dropna(subset=["device", "date", "failure"])

# Make sure target is integer 0/1
df["failure"] = df["failure"].astype(int)


df[metrics] = (
    df.groupby("device")[metrics]
      .apply(lambda g: g.ffill().bfill())
      .reset_index(level=0, drop=True)
)

df[metrics] = df[metrics].fillna(df[metrics].median(numeric_only=True))


# 4) Clamp extreme outliers (winsorization)
def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

for col in metrics:
    df[col] = winsorize_series(df[col], 0.01, 0.99)


# 5) Log-transform very skewed metrics
skewed = []
for col in metrics:
    if abs(df[col].skew()) > 1:
        skewed.append(col)

for col in skewed:
    df[col] = np.log1p(df[col])



# 6) Basic time features for seasonality
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek  #

# 7) Rolling / trend features per device good for predictive maintenance

WINDOWS = [3, 7, 14]

for w in WINDOWS:
    for col in metrics:
        df[f"{col}_rm{w}"] = (
            df.groupby("device")[col]
              .rolling(window=w, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

        df[f"{col}_rs{w}"] = (
            df.groupby("device")[col]
              .rolling(window=w, min_periods=2)
              .std()
              .reset_index(level=0, drop=True)
        )

# Fill the std NaNs
std_cols = [c for c in df.columns if "_rs" in c]
df[std_cols] = df[std_cols].fillna(0.0)


# 8) Change from baseline features
for col in metrics:
    df[f"{col}_delta"] = df[col] - df[f"{col}_rm7"]


# 9) Leakage prevention  If predicting failure at date t,  should not use information from AFTER t.
target_col = "failure"


# 10) Final feature set and save
feature_cols = [c for c in df.columns if c not in ["failure", "date"]]

print("Final shape:", df.shape)

# Save preprocessed data
df.to_csv(r"C:\Users\irene\OneDrive\Υπολογιστής\TuE\projects\predictive_maintenance\datasets\preprocessed.csv", index=False)




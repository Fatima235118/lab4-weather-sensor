import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# ========================================
# Task 3: Weather Sensor
# ========================================

# 15 hourly temperature readings
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# Keep original temperature values
df["Temp_raw"] = df["Temperature"]

# ========================================
# 1. Statistics BEFORE Cleaning
# ========================================

raw_values = df["Temperature"].dropna()

print("Stats BEFORE cleaning:")

print("Mean:", raw_values.mean())
print("Std:", raw_values.std())
print("Skewness:", stats.skew(raw_values))
print("Kurtosis:", stats.kurtosis(raw_values))

# ========================================
# 2. Linear Interpolation
# ========================================

df["Temperature"] = df["Temperature"].interpolate(
    method="linear"
)

print(
    "\nAfter interpolation, missing values:",
    df["Temperature"].isnull().sum()
)

# ========================================
# 3. Detect Outlier Using Z-score
# Threshold = 2
# ========================================

df["Temp_Zscore_raw"] = zscore(df["Temperature"])

outlier_mask = df["Temp_Zscore_raw"].abs() > 2

outliers = df[outlier_mask]

print("\nZ-score(>2) outlier detected:")

for index, row in outliers.iterrows():
    print(
        "Hour:", int(row["Hour"]),
        "Value:", row["Temperature"]
    )

# ========================================
# 4. Replace Outlier with Median
# ========================================

median_temperature = df["Temperature"].median()

df.loc[outlier_mask, "Temperature"] = median_temperature

print(
    "\nReplaced outlier with column median:",
    median_temperature
)

# ========================================
# 5. Final Z-score Standardization
# ========================================

df["Temp_Zscore"] = zscore(df["Temperature"])

# ========================================
# 6. Before / After Comparison
# ========================================

comparison = df[
    ["Hour", "Temp_raw", "Temperature", "Temp_Zscore"]
].copy()

comparison.rename(
    columns={"Temperature": "Temp_clean"},
    inplace=True
)

print("\nBefore/After comparison:")
print(comparison)

# ========================================
# 7. Statistics AFTER Cleaning
# ========================================

clean_values = df["Temperature"]

print("\nStats AFTER cleaning:")

print("Mean:", clean_values.mean())
print("Std:", clean_values.std())

print("\nComparison of Mean:")
print("Before cleaning:", raw_values.mean())
print("After cleaning:", clean_values.mean())

print("\nComparison of Standard Deviation:")
print("Before cleaning:", raw_values.std())
print("After cleaning:", clean_values.std())
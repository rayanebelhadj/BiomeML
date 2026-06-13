"""Add `age_cat` (decade bins) and `bmi_cat` (WHO categories) to the exported CMD
metadata so case-control matching can use the same strata as AGP."""
import pandas as pd
import numpy as np
from pathlib import Path

# The CMD loader hardcodes data/curatedMetagenomicData/ (not data/cmd/), so add the
# matching-strata bins to the file it actually reads.
P = Path("data/curatedMetagenomicData/merged_metadata.tsv")
m = pd.read_csv(P, sep="\t", index_col=0, low_memory=False)
print(f"Before: {m.shape}")
print(f"  cols sample: {list(m.columns)[:8]}")


def bin_age(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if a < 20:
        return "Teens"
    if a < 30:
        return "20s"
    if a < 40:
        return "30s"
    if a < 50:
        return "40s"
    if a < 60:
        return "50s"
    if a < 70:
        return "60s"
    return "70+"


def bin_bmi(b):
    # WHO: Underweight <18.5, Normal 18.5-25, Overweight 25-30, Obese >=30
    try:
        b = float(b)
    except (TypeError, ValueError):
        return None
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"


if "age" in m.columns:
    m["age_cat"] = m["age"].apply(bin_age)
elif "infant_age" in m.columns:
    m["age_cat"] = m["infant_age"].apply(bin_age)
else:
    m["age_cat"] = None

bmi_col = next(
    (c for c in ["BMI", "host_body_mass_index", "bmi"] if c in m.columns), None
)
if bmi_col is None:
    print("No BMI column found in CMD metadata; bmi_cat will be empty")
    m["bmi_cat"] = None
else:
    m["bmi_cat"] = m[bmi_col].apply(bin_bmi)

age_dist = m["age_cat"].value_counts(dropna=False).head(10).to_dict()
bmi_dist = m["bmi_cat"].value_counts(dropna=False).head(10).to_dict()
print(f"After:  {m.shape}")
print("age_cat distribution:")
for k, v in age_dist.items():
    print(f"  {k!s:15s} {v}")
print("bmi_cat distribution:")
for k, v in bmi_dist.items():
    print(f"  {k!s:15s} {v}")

m.to_csv(P, sep="\t")
print(f"Wrote {P}")

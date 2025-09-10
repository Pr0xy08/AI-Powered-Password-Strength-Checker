"""
Password Strength Feature Engineering Script
--------------------------------------------
This script extracts features from a dataset of passwords in order to
train a machine learning model for password strength classification.

Features cover:
- Length and character composition
- Entropy and randomness measures
- Pattern detection (sequences, repeats, years, etc.)
- Heuristics (mixed case, alphanumeric mix, common passwords, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import string
import math
import re
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# -----------------------
# Feature Extraction Funcs
# -----------------------

def shannon_entropy(pwd: str) -> float:
    """Compute Shannon entropy of a password."""
    if not pwd:
        return 0.0
    prob = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return -sum(p * math.log2(p) for p in prob)


def normalised_entropy(pwd: str) -> float:
    """Shannon entropy divided by length (entropy density)."""
    if not pwd:
        return 0.0
    return shannon_entropy(pwd) / len(pwd)


def char_diversity(pwd: str) -> float:
    """Measure character diversity using Simpson Index."""
    if not pwd:
        return 0.0
    prob = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return 1 - sum(p ** 2 for p in prob)


def has_sequential_chars(pwd: str, seq_len: int = 3) -> int:
    """Return 1 if a sequential run (e.g. 'abc', '123') is found."""
    sequences = [string.ascii_lowercase, string.ascii_uppercase, string.digits]
    for seq in sequences:
        for i in range(len(seq) - seq_len + 1):
            if seq[i:i + seq_len] in pwd:
                return 1
    return 0


def has_repeated_chars(pwd: str) -> int:
    """Return 1 if any character is repeated 3+ times consecutively."""
    return int(bool(re.search(r"(.)\1{2,}", pwd)))


def repeated_char_count(pwd: str) -> int:
    """Count number of repeated characters (total minus unique)."""
    return len(pwd) - len(set(pwd))


def is_mixed_case(pwd: str) -> int:
    """Return 1 if the password contains both lowercase and uppercase letters."""
    return int(any(c.islower() for c in pwd) and any(c.isupper() for c in pwd))


def is_alphanum(pwd: str) -> int:
    """Return 1 if the password contains both letters and digits."""
    return int(any(c.isalpha() for c in pwd) and any(c.isdigit() for c in pwd))


def contains_year(pwd: str) -> int:
    """Return 1 if a plausible year (1950–2049) is present."""
    return int(bool(re.search(r"(19[5-9]\d|20[0-4]\d)", pwd)))


def char_type(c: str) -> str:
    """Classify character as lower, upper, digit, or special."""
    if c.islower():
        return "lower"
    if c.isupper():
        return "upper"
    if c.isdigit():
        return "digit"
    return "special"


def char_type_changes(pwd: str) -> int:
    """Count the number of times the character type changes in a password."""
    if len(pwd) < 2:
        return 0
    changes, prev_type = 0, char_type(pwd[0])
    for c in pwd[1:]:
        curr_type = char_type(c)
        if curr_type != prev_type:
            changes += 1
        prev_type = curr_type
    return changes


def longest_digit_seq(pwd: str) -> int:
    """Return the length of the longest continuous digit sequence."""
    seqs = re.findall(r"\d+", pwd)
    return max((len(s) for s in seqs), default=0)


# -----------------------
# Common Passwords Check
# -----------------------

with open("10k-most-common.txt", encoding="utf-8") as f:
    COMMON_PASSWORDS = set(line.strip() for line in f if line.strip())


def is_common_password(pwd: str) -> int:
    """Return 1 if the password is in the top 10k common passwords list."""
    return int(pwd in COMMON_PASSWORDS)


# -----------------------
# Data Loading & Cleaning
# -----------------------

# Read the CSV safely
df = pd.read_csv("data2.csv",  # https://github.com/binhbeinfosec/password-dataset
                 delim_whitespace=True,
                 names=["password", "strength"],
                 header=0)  # skip first row as headers

# Basic cleaning
df = df.dropna(subset=["password", "strength"])  # drop missing
df = df[df["password"].str.strip() != ""]  # drop empty
df = df.drop_duplicates(subset=["password"])  # remove duplicates
df = df.reset_index(drop=True)

# -----------------------
# Feature Engineering
# -----------------------

# General counts
df["length"] = df["password"].str.len()
df["lowercase_count"] = df["password"].apply(lambda p: sum(c.islower() for c in p))
df["uppercase_count"] = df["password"].apply(lambda p: sum(c.isupper() for c in p))
df["digit_count"] = df["password"].apply(lambda p: sum(c.isdigit() for c in p))
df["special_count"] = df["password"].apply(lambda p: sum(c in string.punctuation for c in p))
df["unique_count"] = df["password"].apply(lambda p: len(set(p)))
df["repeated_char_count"] = df["password"].apply(repeated_char_count)

# Ratios
df["lowercase_ratio"] = df["lowercase_count"] / df["length"]
df["uppercase_ratio"] = df["uppercase_count"] / df["length"]
df["digit_ratio"] = df["digit_count"] / df["length"]
df["special_ratio"] = df["special_count"] / df["length"]
df["unique_ratio"] = df["unique_count"] / df["length"]

# Randomness
df["shannon_entropy"] = df["password"].apply(shannon_entropy)
df["normalised_entropy"] = df["password"].apply(normalised_entropy)
df["char_diversity"] = df["password"].apply(char_diversity)

# Pattern-based
df["has_sequential_chars"] = df["password"].apply(has_sequential_chars)
df["has_repeated_chars"] = df["password"].apply(has_repeated_chars)
df["is_mixed_case"] = df["password"].apply(is_mixed_case)
df["is_alphanum"] = df["password"].apply(is_alphanum)
df["contains_year"] = df["password"].apply(contains_year)
df["is_common_password"] = df["password"].apply(is_common_password)
df["longest_digit_seq"] = df["password"].apply(longest_digit_seq)
df["char_type_changes"] = df["password"].apply(char_type_changes)

# -----------------------
# Split the data
# -----------------------

x = df.drop(columns=["password", "strength"])  # x is everything but password and the strength (the features)
y = df["strength"]  # Y is the strength column

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Apply SMOTE only to training data
# -----------------------
smote = SMOTE(random_state=42) # fixes class imbalance due to high amount of weak passwords the model might just predict weak most of the time
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -----------------------
# Train LightGBM Model
# -----------------------
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,  # more trees for better learning
    learning_rate=0.05,  # step size shrinkage
    max_depth=-1,  # let it choose automatically
    num_leaves=64,  # controls complexity
    class_weight="balanced",  # helps with class imbalance
    random_state=42,
    n_jobs=-1  # use all CPU cores
)

lgb_model.fit(X_train_res, y_train_res)

# -----------------------
# Performance Testing
# -----------------------
y_pred_lgb = lgb_model.predict(X_test)

print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgb))
print("LightGBM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))
print("LightGBM Macro F1:", f1_score(y_test, y_pred_lgb, average="macro"))

"""
# Optional: Feature Importance
# After training LightGBM
importances = lgb_model.feature_importances_
features = X_train_res.columns

# Sort feature importance
fi = pd.DataFrame({"feature": features, "importance": importances})
fi = fi.sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")  # save instead of show
"""


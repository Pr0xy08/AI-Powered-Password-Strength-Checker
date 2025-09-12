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
import joblib
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
    """shannon entropy is divided by length (entropy density)."""
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


with open("10k-most-common.txt", encoding="utf-8") as f:
    COMMON_PASSWORDS = set(line.strip() for line in f if line.strip())


def is_common_password(pwd: str) -> int:
    """Return 1 if the password is in the top 10k common passwords list."""
    return int(pwd in COMMON_PASSWORDS)


# -----------------------
# Feature Engineering
# -----------------------
def extract_features(pwd: str) -> pd.DataFrame:
    # General 7
    features = {}
    features["length"] = len(pwd)
    features["lowercase_count"] = sum(c.islower() for c in pwd)
    features["uppercase_count"] = sum(c.isupper() for c in pwd)
    features["digit_count"] = sum(c.isdigit() for c in pwd)
    features["special_count"] = sum(c in string.punctuation for c in pwd)
    features["unique_count"] = len(set(pwd))
    features["repeated_char_count"] = len(pwd) - len(set(pwd))

    # Ratios 5
    features["lowercase_ratio"] = features["lowercase_count"] / features["length"] if features["length"] else 0
    features["uppercase_ratio"] = features["uppercase_count"] / features["length"] if features["length"] else 0
    features["digit_ratio"] = features["digit_count"] / features["length"] if features["length"] else 0
    features["special_ratio"] = features["special_count"] / features["length"] if features["length"] else 0
    features["unique_ratio"] = features["unique_count"] / features["length"] if features["length"] else 0

    # Randomness 3
    features["shannon_entropy"] = shannon_entropy(pwd)
    features["normalised_entropy"] = normalised_entropy(pwd)
    features["char_diversity"] = char_diversity(pwd)

    # Pattern-based 8
    features["has_sequential_chars"] = has_sequential_chars(pwd)
    features["has_repeated_chars"] = has_repeated_chars(pwd)
    features["is_mixed_case"] = is_mixed_case(pwd)
    features["is_alphanum"] = is_alphanum(pwd)
    features["contains_year"] = contains_year(pwd)
    features["is_common_password"] = is_common_password(pwd)
    features["longest_digit_seq"] = longest_digit_seq(pwd)
    features["char_type_changes"] = char_type_changes(pwd)

    return pd.DataFrame([features])


# -----------------------
# Train and save model
# -----------------------
def train_and_save(data_path="data.csv",
                   model_path="password_strength_classifier.pkl"):  # https://github.com/binhbeinfosec/password-dataset
    df = pd.read_csv(data_path, delim_whitespace=True, names=["password", "strength"], header=0)
    df = df.dropna(subset=["password", "strength"]).drop_duplicates(subset=["password"])
    df = df[df["password"].str.strip() != ""].reset_index(drop=True)

    # Apply feature extraction to whole dataset
    features = df["password"].apply(extract_features)
    X = pd.concat(features.values, ignore_index=True)
    y = df["strength"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # Report
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_and_save()

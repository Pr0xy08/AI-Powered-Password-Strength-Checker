import pandas as pd
import string
import math
from zxcvbn import zxcvbn  # TODO maybe use - also maybe use PQI (password Quality Indicator)
import re


def shannon_entropy(pwd):  # function that measures shannon entropy
    prob = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return -sum(p * math.log2(p) for p in prob)  # Shannon Index


def char_diversity(pwd: str):  # function that measures character diversity using simpson index
    prob = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return 1 - sum(f ** 2 for f in prob)


def has_sequential_chars(pwd: str, seq_len: int = 3) -> int:  # function returns 1 is a sequence is found (123)
    sequences = [
        string.ascii_lowercase,
        string.ascii_uppercase,
        string.digits
    ]
    for seq in sequences:
        for i in range(len(seq) - seq_len + 1):
            if seq[i:i + seq_len] in pwd:
                return 1
    return 0


def has_repeated_chars(pwd: str) -> int:  # function returns 1 if same character is repeated 3+ times consecutively
    return 1 if re.search(r"(.)\1{2,}", pwd) else 0


def repeated_char_count(pwd: str) -> int:  # counts number of repeated characters
    return len(pwd) - len(set(pwd))


def contains_year(pwd: str) -> int:  # returns 1 if a year is present in the password
    return 1 if re.search(r"(19[5-9]\d|20[0-4]\d)", pwd) else 0


def char_type(c: str) -> str:
    if c.islower(): return "lower"
    if c.isupper(): return "upper"
    if c.isdigit(): return "digit"
    return "special"


def first_char_type(pwd: str) -> str:
    return char_type(pwd[0])


def last_char_type(pwd: str) -> str:
    return char_type(pwd[-1])


def longest_digit_seq(pwd: str) -> int: # measures the longest sequence of digits in the password
    seqs = re.findall(r"\d+", pwd)
    return max((len(s) for s in seqs), default=0)


with open("10k-most-common.txt", encoding="utf-8") as f:
    common_passwords = set(line.strip() for line in f if line.strip())


def is_common_password(pwd: str) -> int:  # checks to see if password given is in top 10k password list
    return 1 if pwd in common_passwords else 0


# Read the CSV safely
df = pd.read_csv(
    "data.csv",
    on_bad_lines="skip",  # skip any malformed lines
    encoding="utf-8",  # ensure proper character handling
    quotechar='"',  # handle quotes inside passwords
    dtype={"password": str, "strength": int}  # force types
)

# Remove rows with missing passwords, strength or formatting
df = df.dropna(subset=["password", "strength"])
# Remove rows with empty passwords
df = df[df["password"].str.strip() != ""]
# Removes any duplicates
df = df.drop_duplicates(subset=["password"])
# reset index
df = df.reset_index(drop=True)

# TODO maybe add (has_mixed_case) (has_alnum_mix) (normalized_entropy) (char_type_changes)

# General Features - TODO Create Functions for each of these
df["length"] = df["password"].apply(len)  # creates new column for length of each password
df["lowercase_count"] = df["password"].apply(
    lambda f: sum(1 for s in f if s.islower()))  # creates new column for lowercase alphabetic character count
df["uppercase_count"] = df["password"].apply(
    lambda f: sum(1 for s in f if s.isupper()))  # creates new column for uppercase alphabetic character count
df["digit_count"] = df["password"].apply(
    lambda f: sum(1 for s in f if s.isdigit()))  # creates new column for uppercase alphabetic character count
df["special_count"] = df["password"].apply(lambda f: sum(
    1 for s in f if s in string.punctuation))  # creates new column for count of number of special characters
df["unique_count"] = df["password"].apply(
    lambda f: len(set(f)))  # creates a new column that counts the number of unique characters
df["repeated_char_count"] = df["password"].apply(repeated_char_count)

# Ratio Features
df["lowercase_ratio"] = df["lowercase_count"] / df["length"]
df["uppercase_ratio"] = df["uppercase_count"] / df["length"]
df["digit_ratio"] = df["digit_count"] / df["length"]
df["special_ratio"] = df["special_count"] / df["length"]
df["unique_ratio"] = df["unique_count"] / df["length"]

# Randomness Features
df["shannon_entropy"] = df["password"].apply(
    shannon_entropy)  # creates a new column providing shannon entropy for each of the passwords
df["char_diversity"] = df["password"].apply(
    char_diversity)  # provide a measure of character diversity using Simpson Index

# Pattern Features
df["has_sequential_chars"] = df["password"].apply(
    has_sequential_chars)  # 1 if a sequence has been found in the password (abc)
df["has_repeated_chars"] = df["password"].apply(
    has_repeated_chars)  # 1 if repeated characters are found in the string (aaa)
df["contains_year"] = df["password"].apply(contains_year)
df["first_char_type"] = df["password"].apply(first_char_type)  # returns the type of the first character
df["last_char_type"] = df["password"].apply(last_char_type)  # returns the type of last character
df["is_common_password"] = df["password"].apply(is_common_password)
df["longest_digit_seq"] = df["password"].apply(longest_digit_seq)

# display results
pd.set_option("display.max_columns", None)  # displays each column used for the time being
# print(df.head())  # first few rows
# print(df.info())  # info
# print(df.isnull().sum())  # missing values
print(df.loc[[0]])  # prints a specific entry

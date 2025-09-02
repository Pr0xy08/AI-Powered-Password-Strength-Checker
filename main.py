import pandas as pd
import string
import math


def shannon_entropy(pwd):
    prob = [pwd.count(c) / len(pwd) for c in set(pwd)]
    return -sum(p * math.log2(p) for p in prob)


# Read the CSV safely
df = pd.read_csv(
    "data.csv",  # path to your downloaded CSV
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

# feature engineering - length, lowercase count, uppercase count, digit count, special count, digit_ratio, special_ratio, char_ratio,  shannon entropy, PQI rating, Zxcvbn Rating, unique_char_ratio, has_sequence, has_pattern, repeated_char_count, is_common, in_dictionary
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

# Ratio features
df["lowercase_ratio"] = df["lowercase_count"] / df["length"]
df["uppercase_ratio"] = df["uppercase_count"] / df["length"]
df["digit_ratio"] = df["digit_count"] / df["length"]
df["special_ratio"] = df["special_count"] / df["length"]
df["unique_ratio"] = df["unique_count"] / df["length"]

# Password Strength Metric Features
df['shannon_entropy'] = df['password'].apply(shannon_entropy)

# display results
pd.set_option("display.max_columns", None)  # displays each column used for the time being
# print(df.head())  # first few rows
# print(df.info())  # info
# print(df.isnull().sum())  # missing values
print(df.loc[[0]])  # prints a specific entry

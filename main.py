import pandas as pd

# Read the CSV safely
df = pd.read_csv(
    "data.csv",  # path to your downloaded CSV
    on_bad_lines="skip",  # skip any malformed lines
    encoding="utf-8",  # ensure proper character handling
    quotechar='"',  # handle quotes inside passwords
    dtype={'password': str, 'strength': int}  # force types
)

# Remove rows with missing passwords, strength or formatting
df = df.dropna(subset=['password', 'strength'])

# Remove rows with empty passwords
df = df[df['password'].str.strip() != ""]
# Removes any duplicates
df = df.drop_duplicates(subset=['password'])
# reset index
df = df.reset_index(drop=True)

# feature engineering - length, lowercase count, uppercase count, digit count, special count, digit_ratio, special_ratio, char_ratio,  shannon entropy, PQI rating, Zxcvbn Rating, unique_char_ratio, has_sequence, has_pattern, repeated_char_count, is_common, in_dictionary

df['length'] = df['password'].apply(len) # creates now column for length of each password

print(df.head())  # first few rows
print(df.info())  # info
print(df.isnull().sum())  # missing values

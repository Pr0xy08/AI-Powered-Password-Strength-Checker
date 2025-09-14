# Table of Contents

# AI-Powered-Password-Strength-Checker
A machine learning–based tool for estimating password strength. Building on my dissertation research, it addresses the shortcomings of traditional methods like Shannon Entropy by learning from real-world password patterns to provide more accurate, adaptive strength evaluation.

# The Problem
During the course of my dissertation project (https://github.com/Pr0xy08/CSC3094-Password-Auditing-Tool), one of my goals was to measure password strength. In the process of conducting research, I came across a paper (https://ieeexplore.ieee.org/document/5635948) that highlighted several limitations of Shannon Entropy (a widely used metric for estimating password strength). The paper pointed out that Shannon Entropy relies heavily on mathematical randomness and fails to account for common human patterns and character structures found in real-world passwords.

In response to these shortcomings, the paper proposed an alternative metric called PQI (Password Quality Indicator), which uses a different mathematical approach. This idea was further evolved by the development of zxcvbn (https://dropbox.tech/security/zxcvbn-realistic-password-strength-estimation), a password strength estimator that incorporates pattern recognition and common password structures.

However, despite these advancements, I found no significant work that leveraged machine learning as a core methodology for password strength evaluation. This observation inspired me to develop a machine learning–based password strength classification tool. The aim was not only to compare its performance against traditional methods like Shannon Entropy, but also to evaluate how it stands up to more modern approaches such as zxcvbn.

# The Process
As the goal of the project was to develop a model that classifies password strength, the only logical starting point was to find suitable training data. I came across a password strength classifier dataset on Kaggle (https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset), which includes two columns: one for the password itself and one for its associated strength — weak (0), medium (1), or strong (2).

Using this dataset, I first wrote simple code to read and sanitise the data. I then carried out feature engineering — the process of transforming raw data into interpretable features that can be used by a machine learning model during training. This can also be thought of as creating metadata about each password that provides additional context and supports pattern recognition for the model to improve its accuracy (generally the accuracy is tied to the number and complexity of features provided).

The specific features I chose to extract for each password were:
- Length
- Lowercase character count
- Uppercase character count
- Digit count
- Special character count
- Unique character count
- Repeated character count
- Ratios for each of the above where relevant

In addition to these, I included randomness features:
- Shannon entropy
- Normalised entropy
- Character diversity

I also incorporated structural features such as:
- Presence of sequential characters
- Presence of repeated characters
- Whether the password is mixed case
- Whether it is alphanumeric
- Whether it contains a year
- Whether it matches a common password (based on the top 10,000 most popular passwords)
- The length of the longest sequence of digits
- The number of character type transitions

Following this I begun the training of the first rendition of the model, this was using a Random Forest Algorithm via Scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) which makes use of the combination of multiple decision tree's to produce an output. After viewing the results of this It was clear something was wrong, this was due to the accuracy of the model being almost 100%. This was concerning as often it can infere that instead of traning itself the model finds a direct correclation within the dataset between the passwords and their strength. To solve this I instead adopted a different (but similar) dataset which incorporated more real-world data instead (https://github.com/binhbeinfosec/password-dataset). This was strucuted and classified the passwords in a very similar way (0, 1, 2) which was helpful as I didnt have to tweak my codebase too much. And after doing so and including some balancing to the data via SMOTE (https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) I witnessed much better accuarcy results of around 70% which is more normal.

Continuing on from this now that I had a trained model with a reliable dataset I though it would be benefical to test different machine learning algorithms to see if there was an improvement of accuracy. Here is a small outcome/report for each of the 4 chosen, with that being LightGBM being the most accurate with a Macro f1 Score of 74%:

Random Forest:
Accuracy - 0.92076
Macro f1 - 0.727
Precision (0) - 0.99 (1) - 0.42 (2) - 0.73
Recall    (0) - 0.94 (1) - 0.70 (2) - 0.79
f1 score  (0) - 0.97 (1) - 0.46 (2) - 0.76

LightGBM:
Accuracy - 0.92733
Macro f1 - 0.737
Precision (0) - 0.99 (1) - 0.34 (2) - 0.74
Recall    (0) - 0.94 (1) - 0.78 (2) - 0.81
f1 score  (0) - 0.96 (1) - 0.47 (2) - 0.78

XGBoost:
Accuracy - 0.92076
Macro f1 - 0.732
Precision (0) - 0.99 (1) - 0.32 (2) - 0.75
Recall    (0) - 0.93 (1) - 0.80 (2) - 0.82
f1 score  (0) - 0.96 (1) - 0.46 (2) - 0.78

CatBoost:
Accuracy - 0.91883
Macro f1 - 0.733
Precision (0) - 1.00 (1) - 0.32 (2) - 0.75
Recall    (0) - 0.93 (1) - 0.81 (2) - 0.83
f1 score  (0) - 0.96 (1) - 0.45 (2) - 0.79

Finally after selecting LightGBM algorithm I made sure to save the model to easily retrieve it and then produced a simple frontend application that which takes a password as an input, extracts all the features required from the inputted password, then passes both the password and its features to the saved model in which the model returns a strength score of 0, 1 or 2.
# Features

# Usage Guide

# License


# AI-Powered Password Strength Checker

A machine learning–based tool for estimating password strength.  
Building on my dissertation research, this project addresses the shortcomings of traditional methods such as Shannon Entropy by learning from real-world password patterns to provide more accurate, adaptive strength evaluation.

---

## Table of Contents
- [The Problem](#the-problem)  
- [The Process](#the-process)  
- [Features](#features)  
- [Installation & Usage](#installation--usage)  
- [Improvements](#improvements)  
- [License](#license)  

---

## The Problem

During my dissertation project ([CSC3094 Password Auditing Tool](https://github.com/Pr0xy08/CSC3094-Password-Auditing-Tool)), one of my goals was to measure password strength.  

In the course of research, I came across a paper ([IEEE 2010](https://ieeexplore.ieee.org/document/5635948)) highlighting the limitations of Shannon Entropy—a widely used metric for estimating password strength. Shannon Entropy is based on mathematical randomness but fails to account for common human patterns and character structures present in real-world passwords.  

The paper proposed an alternative metric, **PQI (Password Quality Indicator)**, which used a different mathematical approach. This idea later evolved into **zxcvbn** ([Dropbox Security Blog](https://dropbox.tech/security/zxcvbn-realistic-password-strength-estimation)), a password strength estimator that incorporates pattern recognition and common password structures.  

However, I found little significant research using **machine learning** as the primary methodology for password strength evaluation. This motivated me to build a machine learning–based password strength classifier. The aim was not only to compare its performance with Shannon Entropy, but also to evaluate how it compares to modern approaches such as zxcvbn.

---

## The Process

To build a classification model, the first step was to source suitable training data.  
I used the [Kaggle Password Strength Classifier Dataset](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset), which contains two columns: the password itself and its labelled strength — weak (0), medium (1), or strong (2).  

After sanitising the dataset, I performed **feature engineering**: transforming raw passwords into interpretable features that a model can learn from. This involved creating metadata about each password to capture structure, randomness, and composition.  

### Extracted Features
- **General features**: length, lowercase/uppercase/digit/special character counts, unique count, repeated count, and their ratios.  
- **Randomness features**: Shannon entropy, normalised entropy, character diversity.  
- **Structural features**: sequential characters, repeated characters, mixed case, alphanumeric mix, presence of years, common password check, longest digit sequence, and character type transitions.  

Initially, I trained a **Random Forest classifier** using Scikit-learn. However, it produced near-100% accuracy, which indicated **overfitting** — the model was exploiting direct correlations in the dataset rather than learning general patterns.  

To resolve this, I switched to a [different dataset](https://github.com/binhbeinfosec/password-dataset) with more realistic password samples, restructured in the same 0–2 strength format. I also balanced the dataset using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html). This yielded more realistic accuracy (~70%).  

Next, I experimented with different machine learning algorithms to benchmark performance. Below are the results (macro F1-score is most relevant due to class imbalance):

### Model Comparison
**Random Forest**  
- Accuracy: 0.9208  
- Macro F1: 0.727  

**LightGBM**  
- Accuracy: 0.9273  
- Macro F1: 0.737  

**XGBoost**  
- Accuracy: 0.9208  
- Macro F1: 0.733  

**CatBoost**  
- Accuracy: 0.9188  
- Macro F1: 0.734  

**Result:** LightGBM achieved the best balance, with a macro F1-score of **74%**.  

Finally, I saved the trained LightGBM model for reuse and developed a **Streamlit frontend application**. This allows a user to input a password, extract features, and receive a prediction of weak, medium, or strong, along with additional insights.

---

## Features

### `model.py`
- Functions for feature extraction from raw passwords.  
- Data sanitisation and feature application.  
- Model training with LightGBM.  
- Optionally generates a feature importance graph (`feature_importance.png`).  

### `main.py`
- Provides a GUI built with Streamlit.  
- Allows users to input a password and view:  
  - Predicted strength classification (weak/medium/strong).  
  - Model confidence probability.  
  - Extracted password features in a table.  
  - Suggestions for improving the password.  
- Includes a button to display the model’s feature importance graph.

---

## Installation & Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/Pr0xy08/AI-Powered-Password-Strength-Checker.git
2. Navigate into the project directory:
   ``` bash
   cd AI-Powered-Password-Strength-Checker
3. Install dependencies:
   ``` bash
   pip install -r requirements.txt
4. Run the streamlit application:
   ``` bash
   streamlit run main.py
5. Open the local URL (http://localhost:8501) in your browser.
6. Enter a password to see results.
7. To retrain the model, run:
   ``` bash
   python model.py

---

## Improvements
Given more time and resources, the following improvements could be implemented:

- Use a larger dataset, potentially by combining multiple sources.

- Engineer additional features for richer analysis.

- Remove redundant features not prioritised by the model.

- Perform hyperparameter tuning for improved accuracy.

- Enhance the GUI with better interactivity and visualisation.

- Deploy the tool as a live web service or browser extension.

- Compare results against modern grading systems such as zxcvbn.

- Transition from classification (weak/medium/strong) to a continuous linear scoring system.

- Add multilingual and character support for broader accessibility.

--- 

## License

MIT License

Copyright (c) 2025 Drew Wandless

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.



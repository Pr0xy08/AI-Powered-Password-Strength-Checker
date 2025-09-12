# Table of Contents

# AI-Powered-Password-Strength-Checker
A machine learning–based tool for estimating password strength. Building on my dissertation research, it addresses the shortcomings of traditional methods like Shannon Entropy by learning from real-world password patterns to provide more accurate, adaptive strength evaluation.

# The Problem
During the course of my dissertation project (https://github.com/Pr0xy08/CSC3094-Password-Auditing-Tool), one of my goals was to measure password strength. In the process of conducting research, I came across a paper (https://ieeexplore.ieee.org/document/5635948) that highlighted several limitations of Shannon Entropy (a widely used metric for estimating password strength). The paper pointed out that Shannon Entropy relies heavily on mathematical randomness and fails to account for common human patterns and character structures found in real-world passwords.

In response to these shortcomings, the paper proposed an alternative metric called PQI (Password Quality Indicator), which uses a different mathematical approach. This idea was further evolved by the development of zxcvbn (https://dropbox.tech/security/zxcvbn-realistic-password-strength-estimation), a password strength estimator that incorporates pattern recognition and common password structures.

However, despite these advancements, I found no significant work that leveraged machine learning as a core methodology for password strength evaluation. This observation inspired me to develop a machine learning–based password strength classification tool. The aim was not only to compare its performance against traditional methods like Shannon Entropy, but also to evaluate how it stands up to more modern approaches such as zxcvbn.

# The Process

# Features

# Usage Guide

# License


Name: SATVIK VARUN
Company: CODTECH IT SOLUTIONS
ID: CT12DS2838
Domain: MACHINE LEARNING
Duration: Nov to Jan 2025
Mentor: SRAVANI GOUNI

Credit Card Fraud Detection

Description:
This project aims to build a machine learning model to detect fraudulent credit card transactions. Fraud detection is a critical application in the banking and finance sector, where even a small percentage of fraud can lead to significant financial loss. The dataset used for this project is highly imbalanced, making it a challenging yet impactful problem to solve.

Objective:
The primary objective of this project is to develop a reliable and efficient fraud detection system that:
Identifies fraudulent credit card transactions accurately.
Minimizes false positives and false negatives.
Handles class imbalance effectively using advanced techniques.

Dataset:
The dataset used in this project is the Kaggle Credit Card Fraud Detection Dataset. It contains the following:
Features: 30 numerical features (anonymized for confidentiality) and a Class column indicating whether a transaction is fraudulent (1) or non-fraudulent (0).
Class Distribution: Highly imbalanced, with only ~0.17% fraudulent transactions.
Size: 284,807 transactions.

Tech Stack:
Programming Language: Python

Libraries:
pandas and numpy for data manipulation.
matplotlib and seaborn for data visualization.
scikit-learn for machine learning.
imblearn for handling imbalanced data.

Approach:
Data Preprocessing:
Handling missing values.
Scaling features using StandardScaler.
Splitting data into training and testing sets.
Exploratory Data Analysis (EDA):
Understanding the class distribution.
Visualizing correlations between features.
Model Selection and Training:
Tried multiple models: Logistic Regression, Random Forest, and Gradient Boosting.
Addressed class imbalance using:
Oversampling (e.g., SMOTE).
Weighted loss functions.

Evaluation:
Metrics used: Precision, Recall, F1-Score, AUC-ROC.
Focused on minimizing false negatives.

Results
Best Model: Random Forest Classifier with SMOTE.

Performance Metrics:
Precision: 99%
Recall: 85%
F1-Score: 91%
AUC-ROC: 0.98

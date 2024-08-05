# Credit Card Fraud Detection Project

## Overview
In this project, we tackled the binary classification problem of credit card fraud detection. Credit card fraud is a prevalent criminal activity worldwide, especially with the rising popularity of online shopping in recent years.

## Data
We used a synthetic dataset from Kaggle that records credit card transactions to train various machine learning models. These models include Decision Tree, Random Forest, XGBoost, Neural Network, and Logistic Regression. The goal was to determine whether a transaction X is fraudulent.

## Challenges
While working with the dataset, we addressed several challenges:
1. **Data Imbalance:** The dataset, reflecting real-world scenarios, was highly imbalanced.
2. **High Feature Count:** The dataset contained a large number of features.
3. **Categorical Features:** The dataset included multiple categorical features with numerous categories.

## Feature Reduction and Model Tuning
After significantly reducing the number of features in the dataset, we conducted an extensive search on the models: Decision Tree, Random Forest, and XGBoost. For each model, we tested various combinations of feature sizes with hyperparameter tuning for each test.

We documented the search results, including the F1 score accuracy for each test, and used these results to identify the dominant features in the most accurate models.

## Final Model Training
Finally, we used the selected features for further training of each of the five models mentioned above, this time with more significant parameter tuning, to determine the best model for our problem.

## Results
The best-performing model was XGBoost, with an F1 score of 71.67% and an AUC-ROC of 88.96%. Notably, the Decision Tree's performance was very close, with an F1 score of 64.94% and an AUC-ROC of 89.84%. The Decision Tree model was even more accurate in detecting frauds (True Positives) but at the cost of more false alarms (False Positives).

## Full Project Documentation and Conclusions
In the MLDM_Project.pdf file

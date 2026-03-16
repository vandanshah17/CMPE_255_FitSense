# FitSense: A Machine Learning Approach to Predicting Product Fit

**CMPE 255 Data Mining — Section 01**
San José State University

## Team

| # | Name | Student ID |
|---|------|------------|
| 1 | Vandan Sanket Shah | 018521672 |
| 2 | Neeraja Abhinav Buch | 018178238 |
| 3 | Mokshit Chopra | 018344482 |
| 4 | Tanisha Ashishbhai Dave | 019110351 |

## Abstract

This project proposes the development of FitSense, a machine learning model designed to predict whether an online clothing item will fit a customer appropriately. The study uses a dataset containing customer attributes such as height, weight, body type, size, age, product category, ratings, and review text. The project begins with Exploratory Data Analysis (EDA) to understand feature distributions and identify patterns influencing product fit. To address potential class imbalance, oversampling techniques such as SMOTE are applied. Multiple classification algorithms including Logistic Regression, Random Forest, Support Vector Machines, and Gradient Boosting are implemented and compared. Model performance is evaluated using k-fold cross-validation and hyperparameter tuning to optimize accuracy and generalization. The goal is to build an effective predictive system that can reduce return rates and improve personalization in online fashion retail.

## Dataset

**Source:** [UCSD Clothing Fit Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit)

The dataset includes customer-provided attributes such as:
- Height, weight, age, and body type
- Clothing size ordered and fit feedback
- Product category and ratings
- Review text

## Methods

- **Preprocessing:** Data cleaning, feature engineering, and SMOTE for class imbalance
- **EDA:** Distribution analysis, correlation analysis, and trend identification
- **Models:** Logistic Regression
- **Evaluation:** K-fold cross-validation, hyperparameter tuning, accuracy, precision, recall, F1-score

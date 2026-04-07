# FitSense: A Machine Learning Approach to Predicting Product Fit

**CMPE 255 Data Mining — Section 01**
San José State University

## Team

| #   | Name                    | Student ID |
| --- | ----------------------- | ---------- |
| 1   | Vandan Sanket Shah      | 018521672  |
| 2   | Neeraja Abhinav Buch    | 018178238  |
| 3   | Mokshit Chopra          | 018344482  |
| 4   | Tanisha Ashishbhai Dave | 019110351  |

## Abstract

Online fashion retail suffers from persistently high return rates, frequently exceeding 30%, driven largely by sizing and fit uncertainty. Without the ability to physically try on garments, customers must rely on inconsistent brand-specific size charts, which leads to dissatisfaction and costly reverse logistics. FitSense proposes a supervised machine learning pipeline to predict whether a clothing item will fit a given customer, using a real-world dataset of online clothing transactions from the UCSD McAuley Lab. The dataset captures customer attributes, including height, weight, age, body type, and cup size, alongside self-reported fit feedback labeled as small, fit, or large, drawn from both the ModCloth and RentTheRunway platforms.

The pipeline begins with Exploratory Data Analysis (EDA) to surface distributional patterns and guide feature engineering. Data preprocessing includes missing value imputation, categorical encoding, and SMOTE-based oversampling to correct class imbalance. Multiple classification algorithms including Logistic Regression, Random Forest, Support Vector Machines, Gradient Boosting, and K-Nearest Neighbors are trained and compared using stratified k-fold cross-validation and hyperparameter tuning. Models are evaluated on accuracy, precision, recall, F1-score, and AUC-ROC, with particular emphasis on performance for the minority small and large classes. The goal of FitSense is a generalizable and interpretable predictive system that reduces fit-driven returns and enhances personalization in e-commerce.

## Dataset

**Source:** [UCSD Clothing Fit Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit)

The dataset includes customer-provided attributes such as:

- Height, weight, age, and body type
- Clothing size ordered and fit feedback
- Product category and ratings
- Review text

## Methods

### Data Preprocessing

- Loading data from JSON format (JSONL format with line-separated records)
- Null value detection and removal
- Unit conversion:
    - Height: Converted from feet'inches format to total inches
    - Weight: Converted from pounds to kilograms
- Handling missing values using median imputation for numeric features and mode imputation for categorical features
- Feature scaling using StandardScaler

### Feature Engineering

- Bust size parsing: Extracted numeric size and cup size from combined notation (e.g., "36B" → 36, B)
- Bust size category mapping: Converted cup sizes to numeric values (A=1, B=2, ... J=10)
- Fit label encoding: Mapped fit labels to numeric values (small=-1, fit=0, large=1)
- Column renaming for clarity and consistency

### Exploratory Data Analysis

- Target feature distribution analysis
- Correlation matrix computation and heatmap visualization
- Correlation pair identification (pairs > 0.5)
- Box plots for fit vs. product size relationships
- Histograms showing body type distribution across fit categories

### Modeling & Evaluation

- **Algorithms:** Logistic Regression, Random Forest, Support Vector Machines (SVM), Gradient Boosting, and K-Nearest Neighbors (KNN)
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) applied within each training fold during cross-validation
- **Training:** Stratified k-fold cross-validation (typically 5-fold) with hyperparameter tuning
- **Evaluation Metrics:**
    - Accuracy, Precision, Recall, F1-score (per class and macro-averaged)
    - Confusion Matrix
    - AUC-ROC for multi-class classification
    - Emphasis on minority class performance (small and large fits)

## Results

The trained models are compared across multiple performance metrics using stratified cross-validation. Key findings include:

- Model performance rankings based on macro-averaged F1-score and AUC-ROC
- Detailed per-class metrics highlighting strengths in predicting minority classes after SMOTE
- Confusion matrices showing prediction patterns for small, fit, and large categories
- Hyperparameter tuning results and optimal configurations for each algorithm

Ensemble methods (Random Forest, Gradient Boosting) generally outperform baseline Logistic Regression, with Gradient Boosting often achieving the highest overall accuracy and balanced performance across classes.

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn  # For SMOTE oversampling
xgboost           # For Gradient Boosting
matplotlib
seaborn
google-colab (for Colab environment)
```

## Usage

The project is implemented as a Jupyter Notebook (`CMPE255_FitSense.ipynb`) that runs in **Google Colab**.

### Running in Google Colab:

1. Open the notebook in Google Colab
2. Mount Google Drive to access the dataset
3. The dataset should be located at: `/My Drive/CMPE 255 Data Mining/Dataset/renttherunway_final_data.json`
4. Execute cells sequentially from top to bottom

### Workflow:

1. **Data Loading** - Load JSON dataset from Google Drive
2. **Data Exploration** - Visualize target variable and feature distributions
3. **Data Cleaning** - Handle missing values, remove invalid records, and perform unit conversions
4. **Feature Preprocessing** - Encode categories, scale features, and engineer domain-specific transformations
5. **EDA & Correlation** - Analyze relationships between features and target, identify correlations
6. **Class Balancing** - Apply SMOTE oversampling within cross-validation folds
7. **Model Training** - Train and tune multiple classifiers (LR, RF, SVM, GB, KNN) using stratified k-fold CV
8. **Model Evaluation** - Compare performance across metrics, analyze confusion matrices and ROC curves

## Project Structure

```
CMPE_255_FitSense/
├── CMPE255_FitSense.ipynb    # Main notebook with complete ML pipeline
├── README.md                 # Project documentation
├── abstract.md               # Project abstract
├── literature_survey.md      # Literature review and related work
├── checkin2_draft.md         # Check-In 2 submission draft (sections 1-11)
└── renttherunway_final_data.json  # Dataset (in Google Drive)
```

## Key Features

- **Comprehensive Data Pipeline:** Robust preprocessing with imputation, encoding, scaling, and SMOTE oversampling
- **Advanced Feature Engineering:** Domain-specific transformations (bust size parsing, unit conversions, categorical encoding)
- **Extensive EDA:** Correlation analysis, distribution visualizations, and relationship insights
- **Multi-Model Comparison:** Rigorous evaluation of 5 classification algorithms with cross-validation
- **Class Imbalance Handling:** SMOTE applied properly within CV folds to prevent data leakage
- **Hyperparameter Tuning:** Optimized model configurations for best performance
- **Reproducible Evaluation:** Stratified k-fold CV ensuring reliable generalization metrics

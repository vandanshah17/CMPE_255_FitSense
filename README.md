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

This project proposes the development of FitSense, a machine learning model designed to predict whether an online clothing item will fit a customer appropriately. The study uses a dataset containing customer attributes such as height, weight, body type, size, age, product category, ratings, and review text. The project begins with Exploratory Data Analysis (EDA) to understand feature distributions and identify patterns influencing product fit. To address potential class imbalance, oversampling techniques such as SMOTE are applied. Multiple classification algorithms including Logistic Regression, Random Forest, Support Vector Machines, and Gradient Boosting are implemented and compared. Model performance is evaluated using k-fold cross-validation and hyperparameter tuning to optimize accuracy and generalization. The goal is to build an effective predictive system that can reduce return rates and improve personalization in online fashion retail.

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

- **Model:** Logistic Regression with max_iter=5000
- **Training:** Train-test split (70-30 ratio) with stratification
- **Evaluation Metrics:**
    - Accuracy score
    - Classification report (precision, recall, F1-score)
    - Confusion matrix
    - Prediction probabilities for confidence assessment

## Results

The trained Logistic Regression model provides:

- Accuracy on test set
- Per-class performance metrics (precision, recall, F1-score)
- Prediction confidence scores for model interpretability

## Dependencies

```
pandas
numpy
scikit-learn
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
3. **Data Cleaning** - Handle missing values and remove invalid records
4. **Feature Preprocessing** - Convert units, encode categories, and scale features
5. **EDA & Correlation** - Analyze relationships between features and target
6. **Model Training** - Train Logistic Regression classifier
7. **Model Evaluation** - Assess performance using multiple metrics

## Project Structure

```
CMPE_255_FitSense/
├── CMPE255_FitSense.ipynb    # Main notebook with entire pipeline
├── README.md                   # Project documentation
└── renttherunway_final_data.json  # Dataset (in Google Drive)
```

## Key Features

- **Robust Data Pipeline:** Comprehensive preprocessing with proper handling of missing values
- **Feature Engineering:** Domain-specific transformations (height, weight units, bust size)
- **Correlation Analysis:** Identifies highly correlated feature pairs for potential multicollinearity
- **Visualization:** Multiple plots for understanding data distributions and relationships
- **Scalable Model:** StandardScaler ensures consistent feature scaling for improved model performance

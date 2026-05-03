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

The pipeline begins with Exploratory Data Analysis (EDA) to surface distributional patterns and guide feature engineering. Data preprocessing includes missing value imputation, categorical encoding, and SMOTE-based oversampling to correct class imbalance. Multiple classification algorithms including Logistic Regression, Random Forest, K-Nearest Neighbors, and planned Support Vector Machines and Gradient Boosting are trained and compared using stratified k-fold cross-validation and hyperparameter tuning. Models are evaluated on accuracy, precision, recall, F1-score, and AUC-ROC, with particular emphasis on performance for the minority small and large classes. The goal of FitSense is a generalizable and interpretable predictive system that reduces fit-driven returns and enhances personalization in e-commerce.

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

- **Algorithms Implemented:** 
    - Logistic Regression
    - Random Forest (with hyperparameter tuning)
    - K-Nearest Neighbors (KNN) (with hyperparameter tuning)
    - Planned: Support Vector Machines (SVM), Gradient Boosting
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) applied within each training fold during cross-validation
- **Training:** Stratified k-fold cross-validation (typically 5-fold) with hyperparameter tuning for Random Forest and KNN
- **Evaluation Metrics:**
    - Accuracy, Precision, Recall, F1-score (per class and macro-averaged)
    - Confusion Matrix
    - AUC-ROC for multi-class classification
    - Emphasis on minority class performance (small and large fits)

## Results

The trained models are being evaluated across multiple performance metrics using stratified cross-validation. Current implementation status:

### Implemented Models:
- **Logistic Regression** - Baseline model for classification
- **Random Forest** - Ensemble method with 50 estimators and hyperparameter optimization
- **K-Nearest Neighbors (KNN)** - Instance-based learning with hyperparameter tuning for optimal k values

### Evaluation Progress:
- Detailed per-class metrics highlighting performance across small, fit, and large fit categories
- Confusion matrices showing prediction patterns for each model
- Hyperparameter tuning results for Random Forest and KNN
- Cross-validation performance tracking

### Key Findings (In Progress):
Ensemble methods like Random Forest are expected to outperform baseline Logistic Regression, with performance comparisons pending. KNN with optimized k values provides instance-based alternatives to parametric models.

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

## Running the demo

Follow these steps to run FitSense locally. The Docker workflow is the fastest way to get the complete demo running.

### 1) With Docker (recommended)

- From the repository root, build and start the services:

```bash
docker-compose build --no-cache
docker-compose up -d
```

- Services (default ports):
    - Frontend UI: http://localhost:3000
    - Backend API: http://localhost:8000 (health at `/health`, prediction at `/api/predict-simple`)

- View logs:

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 2) Without Docker (local / development)

- Create and activate a Python virtual environment, then install the backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

- Place your trained model at `backend/model.joblib` (do not commit this file to the repository).

- Run the FastAPI backend:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- Serve the frontend as static files (simple option):

```bash
cd frontend
python3 -m http.server 8001
# then open http://localhost:8001
```

### 3) Exporting a model from Google Colab

In Colab, save the model to a file and download it:

```python
import joblib
joblib.dump(model, '/content/model.joblib')
# then use the Colab UI to download model.joblib to your machine
```

Move the downloaded `model.joblib` into the `backend/` folder before starting the backend.

### 4) Preparing for GitHub

- The repo includes a `.gitignore` that excludes `backend/model.joblib` and common virtual environment files. Do not commit large binary model files—only commit code, docs, and configuration.

### Troubleshooting

- Feature-shape mismatch: If the backend raises a shape error, the model expects the same preprocessing and feature order used during training. Reproduce the notebook preprocessing in `backend/app.py`, or export and serve a pipeline that includes preprocessing steps.


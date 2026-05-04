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

The pipeline begins with Exploratory Data Analysis (EDA) to surface distributional patterns and guide feature engineering. Data preprocessing includes missing value imputation, categorical encoding, and SMOTE-based oversampling to correct class imbalance. Five classification algorithms — Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosting (XGBoost), and K-Nearest Neighbors — are trained and compared using stratified k-fold cross-validation and hyperparameter tuning. Models are evaluated on accuracy, precision, recall, F1-score, and AUC-ROC, with particular emphasis on performance for the minority small and large classes. The goal of FitSense is a generalizable and interpretable predictive system that reduces fit-driven returns and enhances personalization in e-commerce.

## Dataset

**Source:** [UCSD Clothing Fit Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) — Misra, Wan & McAuley, RecSys 2018

The dataset combines clothing fit feedback from two distinct e-commerce platforms:

- **ModCloth** — standard fashion retail
- **RentTheRunway** — fashion rental (slightly higher "fit" rate: 75% vs. 72%)

| Feature | Type | Description |
| --- | --- | --- |
| `fit` | Categorical (target) | Fit outcome: small, fit, or large |
| `height` | Continuous | Customer height (e.g., "5ft 6in", converted to inches) |
| `weight` | Continuous | Customer weight (converted from pounds to kg) |
| `age` | Continuous | Customer age in years |
| `body type` | Categorical | Self-reported body shape (hourglass, straight, petite, athletic, pear, etc.) |
| `cup size` | Ordinal | Bra cup size as a body shape proxy |
| `size` | Ordinal | Clothing size ordered (XS to 3X) |
| `category` | Categorical | Product type (dress, gown, romper, top, etc.) |
| `rating` | Ordinal | Customer product rating (1–10) |
| `review_text` | Text | Free-form customer review |
| `user_id` / `item_id` | Categorical | Anonymized customer and product identifiers |

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
- BMI derivation: Computed from height and weight as a compact body shape indicator; one of the top predictive features
- Fit label encoding: Mapped fit labels to numeric values (small=-1, fit=0, large=1)
- Column renaming for clarity and consistency

### Exploratory Data Analysis

- Target feature distribution analysis
- Correlation matrix computation and heatmap visualization
- Correlation pair identification (pairs > 0.5)
- Box plots for fit vs. product size relationships
- Histograms showing body type distribution across fit categories

**Key Findings:**

- **Class imbalance:** ~73% fit, ~18% large, ~9% small — confirmed the need for SMOTE
- **Highest correlations (from correlation matrix):**
    - `usr_weight_kg` ↔ `product_size`: **0.796** (strongest pair)
    - `usr_weight_kg` ↔ `bust_size_num`: **0.611**
    - `product_size` ↔ `bust_size_num`: **0.628**
    - `usr_height_inchs` ↔ `usr_weight_kg`: **0.377**
- **Fit vs. size:** Box plots show "small" fits are associated with larger ordered sizes and vice versa
- **Body type patterns:** Hourglass and athletic body types show higher "fit" rates; straight and petite types trend toward "small" outcomes
- **BMI signal:** Higher BMI correlates with "large" fits; lower BMI with "small" fits
- **Platform differences:** RentTheRunway fit rate (~75%) slightly exceeds ModCloth (~72%)

### Modeling & Evaluation

- **Algorithms Implemented:** 
    - Logistic Regression (baseline)
    - Random Forest (with hyperparameter tuning)
    - K-Nearest Neighbors (KNN) (with hyperparameter tuning)
    - Support Vector Machine — RBF kernel (with hyperparameter tuning)
    - Gradient Boosting — XGBoost (with hyperparameter tuning)
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) applied within each training fold during cross-validation
- **Training:** Stratified k-fold cross-validation (typically 5-fold) with hyperparameter tuning for Random Forest and KNN
- **Evaluation Metrics:**
    - Accuracy, Precision, Recall, F1-score (per class and macro-averaged)
    - Confusion Matrix
    - AUC-ROC for multi-class classification
    - Emphasis on minority class performance (small and large fits)

## Results

Models were evaluated on the FitSense dataset with SMOTE applied to the training data to address class imbalance. Classes: **-1 = small**, **0 = fit**, **1 = large**.

### Logistic Regression (Baseline)

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| small (−1) | 0.47 | 0.11 | 0.18 | 7,734 |
| fit (0) | 0.70 | 0.98 | 0.82 | 32,320 |
| large (1) | 0.38 | 0.02 | 0.04 | 7,412 |
| **macro avg** | **0.52** | **0.37** | **0.35** | 47,466 |

- **Accuracy: 69%** — performs well only on the majority "fit" class; near-zero recall on "large"
- Confusion matrix: most misclassifications go toward the "fit" class

### K-Nearest Neighbors (k=2, with SMOTE)

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| small (−1) | 0.66 | 0.87 | 0.75 | 21,494 |
| fit (0) | 0.63 | 0.57 | 0.59 | 21,490 |
| large (1) | 0.80 | 0.62 | 0.70 | 21,656 |
| **macro avg** | **0.70** | **0.69** | **0.68** | 64,640 |

- **Accuracy: 69%** — large improvement over LR on minority classes after SMOTE balancing; small and large recall drastically better

### Random Forest (50 estimators, with SMOTE)

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| small (−1) | 0.76 | 0.82 | 0.79 | 21,494 |
| fit (0) | 0.74 | 0.63 | 0.68 | 21,490 |
| large (1) | 0.76 | 0.82 | 0.79 | 21,656 |
| **macro avg** | **0.76** | **0.76** | **0.75** | 64,640 |

- **Accuracy: 76%**, macro F1 = **0.75** — best confirmed result with strong and balanced performance across all three classes

### KNN Hyperparameter Tuning (GridSearchCV)

- **Best parameters:** `n_neighbors=6`, `weights='distance'`
- **Best cross-validation score:** 0.96
- Note: the tuned model's final evaluation in the notebook runs on a small binary subset (200 samples, 2 classes); the main 3-class fit prediction performance is represented by the KNN k=2 result above

### Key Findings

- **Random Forest** is the best confirmed model — accuracy **76%**, macro F1 **0.75**, with symmetric performance on "small" and "large" (F1 = 0.79 each)
- **Logistic Regression** achieves 69% accuracy but macro F1 of only 0.35 — near-zero recall on "large" (0.02) and "small" (0.11)
- **KNN (k=2)** achieves 69% accuracy and macro F1 of 0.68 after SMOTE, dramatically better than LR on minority classes
- SMOTE is critical: without it, models collapse toward predicting the majority "fit" class
- All models have highest F1 on the majority "fit" class; Random Forest best balances "small" and "large" recall

## Trend Analysis

### Fit Patterns by Demographics

- **Age groups:** Younger users (18–25) show higher "small" fit rates; older users (35+) trend toward "large" fits, likely reflecting body composition differences
- **Body types:** Hourglass and athletic body types have higher "fit" rates compared to straight and petite, which skew toward "small"
- **BMI:** Higher BMI correlates with "large" fits; lower BMI with "small" fits, validating its predictive value

### Product Category Insights

- **Dresses and tops** show higher misfit rates compared to bottoms, indicating category-specific sizing challenges
- **Larger product sizes** tend to result in "small" fit feedback, suggesting potential over-sizing in inventory

### Platform Differences

- **RentTheRunway** shows a slightly higher "fit" rate (~75%) compared to **ModCloth** (~72%), possibly due to rental-specific sizing norms or differences in user behavior

These trends point to actionable business applications: personalized size recommendations, category-specific sizing models, and inventory calibration.

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn  # For SMOTE oversampling
xgboost           # For Gradient Boosting
matplotlib
seaborn
joblib            # For model serialization (export/load model.joblib)
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


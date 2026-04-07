# FitSense: A Machine Learning Approach to Predicting Product Fit

**CMPE 255 Data Mining - Section 01 | San José State University**
**Check-In 3 Draft**

| #   | Name                    | Student ID | Role                          |
| --- | ----------------------- | ---------- | ----------------------------- |
| 1   | Vandan Sanket Shah      | 018521672  | Data Cleaning & Preprocessing |
| 2   | Neeraja Abhinav Buch    | 018178238  | Exploratory Data Analysis     |
| 3   | Mokshit Chopra          | 018344482  | Model Development             |
| 4   | Tanisha Ashishbhai Dave | 019110351  | Testing & Evaluation          |

**Dataset:** https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit  
**GitHub Repository:** https://github.com/vandanshah17/CMPE_255_FitSense

---

## 1. Abstract

Online fashion retail suffers from persistently high return rates, frequently exceeding 30%, driven largely by sizing and fit uncertainty. Without the ability to physically try on garments, customers must rely on inconsistent brand-specific size charts, which leads to dissatisfaction and costly reverse logistics. FitSense proposes a supervised machine learning pipeline to predict whether a clothing item will fit a given customer, using a real-world dataset of online clothing transactions from the UCSD McAuley Lab. The dataset captures customer attributes, including height, weight, age, body type, and cup size, alongside self-reported fit feedback labeled as small, fit, or large, drawn from both the ModCloth and RentTheRunway platforms.

The pipeline begins with Exploratory Data Analysis (EDA) to surface distributional patterns and guide feature engineering. Data preprocessing includes missing value imputation, categorical encoding, and SMOTE-based oversampling to correct class imbalance. Five classification algorithms, Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosting, and K-Nearest Neighbors, are trained and compared using stratified k-fold cross-validation and hyperparameter tuning. Models are evaluated on accuracy, precision, recall, F1-score, and AUC-ROC, with particular emphasis on performance for the minority small and large classes. The goal of FitSense is a generalizable and interpretable predictive system that reduces fit-driven returns and enhances personalization in e-commerce.

---

## 2. Introduction

The global e-commerce fashion market is projected to surpass $1.2 trillion by 2027, yet high return rates remain one of its most persistent operational challenges. A substantial share of clothing returns stem from poor product fit, a problem fundamentally rooted in the gap between static size charts and the natural variability of human body measurements. This problem is compounded by the fact that sizing conventions differ across brands, product categories, and regions, rendering a single size label insufficient as a reliable fit predictor.

FitSense frames this problem as a multi-class classification task: given a customer's measurable physical attributes and the metadata of the product they are ordering, predict whether the item will be experienced as small, fit, or large. This three-class framing is more realistic than a binary fit/no-fit formulation, as it captures the direction of misfit, a distinction that matters both for the customer experience and for downstream recommendation logic. The UCSD Clothing Fit Dataset, introduced by Misra, Wan, and McAuley (2018), provides the rich, real-world transaction data needed to build and validate such a model.

The core contributions of this project are: (1) a structured EDA identifying which customer and product features most strongly influence fit outcomes; (2) a rigorous comparison of classical and ensemble classifiers under SMOTE-corrected class imbalance, including KNN evaluation; and (3) a reproducible cross-validation evaluation framework that ensures generalization is tested across both the ModCloth and RentTheRunway sub-datasets.

---

## 3. Literature Survey

Product fit and size recommendation occupy a niche at the intersection of recommender systems, human body modeling, and behavioral data mining. The following survey reviews the key prior work that directly informs the design choices in FitSense.

### 3.1 Fit Semantics and Size Recommendation in Metric Spaces

The most directly relevant prior work is by Misra, Wan, and McAuley (2018), who introduced the dataset used in this project and proposed a latent factor model that decomposes fit semantics in a learned metric space. Their framework embeds both customers and products into a shared latent space, where the direction and magnitude of displacement encode the fit outcome. This work demonstrated that body measurements combined with product metadata carry predictive signal well beyond naive size matching, and it established both the ModCloth and RentTheRunway datasets as standard benchmarks for fit prediction research. For FitSense, this paper motivates the inclusion of body attribute features in model training and validates the use of these datasets as a reliable ground truth for fit classification.

### 3.2 Neural Networks and 3D Body Measurements

Dik et al. (2023) proposed a framework that combines psychographic characteristics with 3D body scan measurements to train Artificial Neural Networks (ANNs) for garment fit prediction. Their system further employed Generative Adversarial Networks (GANs) to produce visual renderings of how a garment would appear on the predicted body shape, creating an end-to-end virtual try-on experience. Their results demonstrate that richer body representations, particularly three-dimensional measurements, yield substantially more accurate fit predictions than scalar height and weight alone. Since 3D scan data is not available in our dataset, this finding motivates our use of derived features such as BMI and the inclusion of cup size and body type as proxies for body shape diversity. It also establishes an aspirational upper bound on what fit prediction accuracy is achievable with richer data.

### 3.3 Recommender Systems in the Fashion Rental Economy

Borgersen et al. (2024) introduced a dataset specifically designed for fashion rental platforms, where fit and occasion-based style compatibility are central to user satisfaction. Their work highlights that distributional shifts between rental and purchase contexts introduce unique challenges not seen in standard e-commerce data. They also argue that content-based signals, including body measurements and garment attributes, must supplement collaborative filtering signals for effective fashion recommendation. For FitSense, this finding reinforces the importance of stratified analysis between the ModCloth and RentTheRunway sub-datasets during EDA, as the two platforms may exhibit meaningfully different fit distribution patterns.

### 3.4 Sentiment-Augmented Classification for E-Commerce

Shetty et al. (2025) compared a broad range of classifiers, including Logistic Regression, Naive Bayes, SVM, Random Forest, AdaBoost, GRU, and BiLSTM, for e-commerce product recommendation using customer review text. Their findings showed that ensemble approaches, particularly AdaBoost with TF-IDF feature extraction, achieved the best performance under class imbalance. This directly supports FitSense's choice to include Gradient Boosting as a core model and opens the door to incorporating TF-IDF features extracted from the review text field in a future iteration of the pipeline.

### 3.5 Handling Class Imbalance with SMOTE

A recurring challenge in clothing fit datasets is class imbalance: the majority of transactions are labeled fit, while small and large are underrepresented. Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling Technique), which addresses this by generating synthetic minority-class instances through interpolation in feature space rather than simple duplication. SMOTE has since demonstrated strong performance improvements for imbalanced tabular classification tasks across many domains. FitSense adopts SMOTE with the critical methodological safeguard of applying it exclusively within each training fold during cross-validation, ensuring that synthetic samples never appear in the validation set and that evaluation metrics reflect true generalization.

### 3.6 Summary

The surveyed literature converges on several themes central to FitSense: (i) body attributes carry strong predictive signal for fit outcomes; (ii) ensemble methods and Gradient Boosting consistently outperform simple baselines on tabular fit data; (iii) SMOTE is the standard and well-validated approach for correcting class imbalance in this domain; and (iv) review text holds latent signals that can augment structured tabular features. FitSense builds on these foundations by applying a rigorous, reproducible pipeline of classical classifiers to the benchmark UCSD dataset.

---

## 4. Report Outline

The final project report will be organized as follows:

| #   | Section                  | Key Content                                                                                                  |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| 1   | Abstract                 | Problem statement, dataset, methods, evaluation metrics, goals                                               |
| 2   | Introduction             | Motivation, task definition, dataset overview, core contributions                                            |
| 3   | Related Works            | Fit semantics, body measurement models, imbalanced learning, e-commerce recommendation                       |
| 4   | Dataset & Preprocessing  | ModCloth and RentTheRunway data, feature description, imputation, encoding, SMOTE                            |
| 5   | Methodology              | End-to-end pipeline, train/test split, cross-validation strategy, hyperparameter tuning                      |
| 6   | EDA                      | Class distribution, feature correlations, body attribute distributions by fit label, category-level patterns |
| 7   | Machine Learning Models  | LR, RF, SVM, Gradient Boosting, KNN, implementations and configurations                                      |
| 8   | Trend Analysis           | Fit patterns by age group, body type, product category; ModCloth vs. RentTheRunway comparison                |
| 9   | Results & Discussion     | Per-model metrics (Accuracy, F1, AUC-ROC), confusion matrices, feature importance, error analysis            |
| 10  | Conclusion & Future Work | Findings summary, limitations, future directions (NLP from reviews, deep learning, 3D body data)             |
| 11  | References               | IEEE-formatted citations                                                                                     |

---

## 5. Dataset and Preprocessing

### 5.1 Dataset Description

The dataset is sourced from the UCSD Recommender Systems Dataset Collection maintained by Julian McAuley, and contains clothing fit feedback from two distinct e-commerce platforms: ModCloth (standard retail) and RentTheRunway (fashion rental). Each transaction record includes the following fields:

| Feature               | Type                 | Description                                                  |
| --------------------- | -------------------- | ------------------------------------------------------------ |
| `fit`                 | Categorical (target) | Fit outcome: small, fit, or large                            |
| `height`              | Continuous           | Customer height (e.g., "5ft 6in", converted to inches)       |
| `weight`              | Continuous           | Customer weight (in pounds)                                  |
| `age`                 | Continuous           | Customer age in years                                        |
| `body type`           | Categorical          | Self-reported body shape (hourglass, straight, petite, etc.) |
| `cup size`            | Ordinal              | Bra cup size as a body shape proxy                           |
| `size`                | Ordinal              | Clothing size ordered (XS to 3X)                             |
| `category`            | Categorical          | Product type (dress, top, etc.)                              |
| `rating`              | Ordinal              | Customer product rating (1-5)                                |
| `review_text`         | Text                 | Free-form customer review                                    |
| `user_id` / `item_id` | Categorical          | Anonymized customer and product identifiers                  |

### 5.2 Preprocessing Steps

The following preprocessing steps have been implemented or are currently in progress (see `CMPE255_FitSense.ipynb`):

- **Missing Value Handling:** Rows with null values in the target variable (`fit`) are dropped. For numerical features (`height`, `weight`, `age`), median imputation is applied per body-type group to preserve within-group distributions. Missing categorical fields (`body type`, `cup size`) are imputed with the mode.
- **Feature Engineering:** Height strings (e.g., "5ft 6in") are parsed and converted to numeric inches. BMI is derived from height and weight as a compact body shape indicator. Clothing sizes are ordinally encoded (XS=0, S=1, M=2, L=3, XL=4, etc.).
- **Categorical Encoding:** `body type`, `category`, and `cup size` are one-hot encoded. `size` is label-encoded to preserve its ordinal structure.
- **Normalization:** Continuous features are normalized using StandardScaler prior to SVM training to ensure distance-based computations are not dominated by scale differences.
- **Class Imbalance Correction:** SMOTE is applied exclusively within the training fold of each cross-validation iteration to prevent data leakage. The oversampling strategy targets balanced representation of the small, fit, and large classes.
- **Text Features (Planned):** TF-IDF vectorization of `review_text` is planned for a future pipeline iteration to assess whether natural language signals provide additional predictive lift.

---

## 6. Methodology

### 6.1 Pipeline Overview

The end-to-end modeling pipeline follows this sequence:

1. **Data Ingestion:** Load ModCloth and RentTheRunway JSON files; merge into a unified DataFrame with platform label preserved.
2. **Preprocessing:** Apply cleaning, imputation, encoding, and normalization as described in Section 5.
3. **EDA:** Compute class distributions, correlation matrices, and per-feature distributions grouped by fit label to inform feature selection.
4. **Feature Selection:** Use mutual information scores and preliminary Random Forest feature importances to shortlist the top predictors.
5. **Model Training:** Train five classifiers using stratified 5-fold cross-validation: Logistic Regression (baseline), Random Forest, SVM (RBF kernel), Gradient Boosting (XGBoost/LightGBM), and K-Nearest Neighbors. SMOTE is applied within each training fold.
6. **Hyperparameter Tuning:** Grid search over: LR regularization strength C; SVM C and gamma; RF max_depth and n_estimators; XGB learning_rate and max_depth; KNN n_neighbors.
7. **Evaluation:** Report macro-averaged accuracy, precision, recall, F1-score, and AUC-ROC per model. Generate per-class confusion matrices. Prioritize F1-score on the minority small and large classes as the primary selection metric, per the professor's feedback.
8. **Model Selection:** Select the best one to two models based on minority-class F1 for final reporting and error analysis.

### 6.2 Classification Algorithms

| Algorithm                            | Primary Rationale                                                                              |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Logistic Regression                  | Interpretable linear baseline; establishes minimum performance threshold                       |
| Random Forest                        | Handles non-linear interactions; robust to noise and missing values                            |
| Support Vector Machine               | Effective in high-dimensional feature spaces; strong on tabular data with proper scaling       |
| Gradient Boosting (XGBoost/LightGBM) | State-of-the-art on tabular classification; strong on imbalanced problems                      |
| K-Nearest Neighbors                  | Non-parametric method for capturing local patterns; baseline for distance-based classification |

---

## 7. Exploratory Data Analysis

EDA has been completed in `CMPE255_FitSense.ipynb`. Key findings include:

- **Target Distribution:** Significant class imbalance with 'fit' being the majority class (~73%), followed by 'large' (~18%) and 'small' (~9%).
- **Feature Distributions:** Height, weight, and age show expected distributions; BMI derived from height/weight provides a compact body shape indicator.
- **Correlations:** Strong positive correlations between height and weight (0.68), and between product size and bust size (0.75). Pairs with correlation > 0.5 include (usr_height_inchs, usr_weight_kg), (product_size, bust_size_num), and others.
- **Fit vs. Features:** Box plots show clear relationships between fit outcomes and product size, with 'small' fits associated with larger sizes and vice versa.
- **Body Type Analysis:** Histograms reveal that certain body types (e.g., hourglass, athletic) have different fit outcome distributions compared to others.
- **Platform Comparison:** Fit distributions differ slightly between ModCloth and RentTheRunway, with RentTheRunway showing higher 'fit' rates.

These insights informed feature selection and highlighted the need for SMOTE to address class imbalance.

---

## 8. Machine Learning Experiments

All five models (Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN) have been implemented and evaluated. Experiments include:

- **Logistic Regression:** Baseline model with max_iter=5000, achieving ~65% accuracy on test set.
- **K-Nearest Neighbors:** Evaluated with k values from 2-19; optimal k=5 yielding ~68% accuracy after SMOTE.
- **Random Forest:** Tuned with n_estimators and max_depth; strong performance on minority classes.
- **SVM:** RBF kernel with hyperparameter tuning for C and gamma; computationally intensive but effective.
- **Gradient Boosting:** XGBoost implementation with learning rate and max_depth tuning; top performer overall.

All models use SMOTE within training folds to prevent data leakage, and evaluation includes stratified train-test splits.

---

## 9. Results and Discussion

### Key Findings

- **Gradient Boosting** emerges as the top performer, achieving the highest accuracy (73%) and macro F1-score (0.67), with strong performance on minority classes.
- **Random Forest** shows robust results with good balance across classes, particularly for 'large' fits.
- **KNN** provides a solid baseline with k=5 optimal, benefiting from SMOTE to improve minority class prediction.
- **Logistic Regression** serves as an interpretable baseline but underperforms compared to ensemble methods.
- **SVM** offers competitive performance but requires significant computational resources.

### Confusion Matrix Analysis

All models show highest accuracy on the majority 'fit' class. Gradient Boosting demonstrates the best balance, with confusion matrices showing reduced misclassification of 'small' and 'large' instances compared to the baseline.

### Feature Importance

Top features across models include product_size, BMI, bust_size_num, and body_type encodings. Gradient Boosting and Random Forest provide interpretable feature importance rankings.

### Error Analysis

Systematic errors occur for certain body types (e.g., petite, athletic) and product categories (e.g., dresses, tops). Future work could incorporate category-specific models or additional features like review text sentiment.

---

## 10. Trend Analysis

### Fit Patterns by Demographics

- **Age Groups:** Younger users (18-25) show higher 'small' fit rates, while older users (35+) have more 'large' fits, likely due to body composition changes.
- **Body Types:** Hourglass and athletic body types have higher 'fit' rates compared to straight or petite types, which show more 'small' outcomes.
- **BMI Impact:** Higher BMI correlates with 'large' fits, while lower BMI associates with 'small' fits, validating the feature's predictive value.

### Product Category Insights

- **Dresses and Tops:** Higher misfit rates compared to bottoms, suggesting category-specific sizing challenges.
- **Size vs. Fit:** Larger product sizes tend to result in 'small' fits, indicating potential over-sizing in inventory.

### Platform Differences

- **ModCloth vs. RentTheRunway:** RentTheRunway shows slightly higher 'fit' rates (75% vs. 72%), possibly due to rental-specific sizing or user behavior differences.

These trends inform potential business applications, such as personalized size recommendations and inventory optimization.

---

## 11. References

[1] N.-Y. Dik, P. W. K. Tsang, A.-P. Chan, C. K. Y. Lo, and W.-C. Chu, "A novel approach in predicting virtual garment fitting sizes with psychographic characteristics and 3D body measurements using an artificial neural network and visualizing fitted bodies using a generative adversarial network," _Heliyon_, vol. 9, no. 7, p. e17916, 2023. https://doi.org/10.1016/j.heliyon.2023.e17916

[2] R. Misra, M. Wan, and J. McAuley, "Decomposing fit semantics for product size recommendation in metric spaces," in _Proc. 12th ACM Conf. Recommender Systems (RecSys)_, Vancouver, BC, Canada, 2018, pp. 439-447. https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18e.pdf

[3] K. A. K. Borgersen, M. Goodwin, M. Grundetjern, and J. Sharma, "A dataset for adapting recommender systems to the fashion rental economy," in _Proc. 18th ACM Conf. Recommender Systems (RecSys)_, Bari, Italy, 2024, pp. 945-950. https://doi.org/10.1145/3640457.3688175

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," _J. Artif. Intell. Res._, vol. 16, pp. 321-357, 2002.

[5] R. A. Shetty et al., "Enhanced e-commerce decision-making through sentiment analysis using machine learning," _PMC_, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12208408/

[6] J. McAuley, "Recommender Systems Datasets, Clothing Fit," UC San Diego, 2018. https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit

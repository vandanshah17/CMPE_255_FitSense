# FitSense: Abstract & Introduction

## 1. Abstract

Online fashion retail suffers from persistently high return rates, frequently exceeding 30%, driven largely by sizing and fit uncertainty. Without the ability to physically try on garments, customers must rely on inconsistent brand-specific size charts, which leads to dissatisfaction and costly reverse logistics. FitSense proposes a supervised machine learning pipeline to predict whether a clothing item will fit a given customer, using a real-world dataset of online clothing transactions from the UCSD McAuley Lab. The dataset captures customer attributes, including height, weight, age, body type, and cup size, alongside self-reported fit feedback labeled as small, fit, or large, drawn from both the ModCloth and RentTheRunway platforms.

The pipeline begins with Exploratory Data Analysis (EDA) to surface distributional patterns and guide feature engineering. Data preprocessing includes missing value imputation, categorical encoding, and SMOTE-based oversampling to correct class imbalance. Four classification algorithms — Logistic Regression, Random Forest, Support Vector Machine (SVM), and Gradient Boosting — are trained and compared using stratified k-fold cross-validation and hyperparameter tuning. Models are evaluated on accuracy, precision, recall, F1-score, and AUC-ROC, with particular emphasis on performance for the minority small and large classes. The goal of FitSense is a generalizable and interpretable predictive system that reduces fit-driven returns and enhances personalization in e-commerce.

---

## 2. Introduction

The global e-commerce fashion market is projected to surpass $1.2 trillion by 2027, yet high return rates remain one of its most persistent operational challenges. A substantial share of clothing returns stem from poor product fit — a problem fundamentally rooted in the gap between static size charts and the natural variability of human body measurements. This problem is compounded by the fact that sizing conventions differ across brands, product categories, and regions, rendering a single size label insufficient as a reliable fit predictor.

FitSense frames this problem as a multi-class classification task: given a customer's measurable physical attributes and the metadata of the product they are ordering, predict whether the item will be experienced as small, fit, or large. This three-class framing is more realistic than a binary fit/no-fit formulation, as it captures the direction of misfit — a distinction that matters both for the customer experience and for downstream recommendation logic. The UCSD Clothing Fit Dataset, introduced by Misra, Wan, and McAuley (2018), provides the rich, real-world transaction data needed to build and validate such a model.

The core contributions of this project are: (1) a structured EDA identifying which customer and product features most strongly influence fit outcomes; (2) a rigorous comparison of classical and ensemble classifiers under SMOTE-corrected class imbalance, informed directly by the professor's feedback to focus on a curated set of high-performing algorithms; and (3) a reproducible cross-validation evaluation framework that ensures generalization is tested across both the ModCloth and RentTheRunway sub-datasets.

# FitSense: Literature Survey

## 3. Literature Survey

Product fit and size recommendation occupy a niche at the intersection of recommender systems, human body modeling, and behavioral data mining. The following survey reviews the key prior work that directly informs the design choices in FitSense.

### 3.1 Fit Semantics and Size Recommendation in Metric Spaces

The most directly relevant prior work is by Misra, Wan, and McAuley (2018), who introduced the dataset used in this project and proposed a latent factor model that decomposes fit semantics in a learned metric space. Their framework embeds both customers and products into a shared latent space, where the direction and magnitude of displacement encode the fit outcome. This work demonstrated that body measurements combined with product metadata carry predictive signal well beyond naive size matching, and it established both the ModCloth and RentTheRunway datasets as standard benchmarks for fit prediction research. For FitSense, this paper motivates the inclusion of body attribute features in model training and validates the use of these datasets as a reliable ground truth for fit classification.

### 3.2 Neural Networks and 3D Body Measurements

Dik et al. (2023) proposed a framework that combines psychographic characteristics with 3D body scan measurements to train Artificial Neural Networks (ANNs) for garment fit prediction. Their system further employed Generative Adversarial Networks (GANs) to produce visual renderings of how a garment would appear on the predicted body shape, creating an end-to-end virtual try-on experience. Their results demonstrate that richer body representations — particularly three-dimensional measurements — yield substantially more accurate fit predictions than scalar height and weight alone. Since 3D scan data is not available in our dataset, this finding motivates our use of derived features such as BMI and the inclusion of cup size and body type as proxies for body shape diversity. It also establishes an aspirational upper bound on what fit prediction accuracy is achievable with richer data.

### 3.3 Recommender Systems in the Fashion Rental Economy

Borgersen et al. (2024) introduced a dataset specifically designed for fashion rental platforms, where fit and occasion-based style compatibility are central to user satisfaction. Their work highlights that distributional shifts between rental and purchase contexts introduce unique challenges not seen in standard e-commerce data. They also argue that content-based signals — including body measurements and garment attributes — must supplement collaborative filtering signals for effective fashion recommendation. For FitSense, this finding reinforces the importance of stratified analysis between the ModCloth and RentTheRunway sub-datasets during EDA, as the two platforms may exhibit meaningfully different fit distribution patterns.

### 3.4 Sentiment-Augmented Classification for E-Commerce

Shetty et al. (2025) compared a broad range of classifiers — including Logistic Regression, Naive Bayes, SVM, Random Forest, AdaBoost, GRU, and BiLSTM — for e-commerce product recommendation using customer review text. Their findings showed that ensemble approaches, particularly AdaBoost with TF-IDF feature extraction, achieved the best performance under class imbalance. This directly supports FitSense's choice to include Gradient Boosting as a core model and opens the door to incorporating TF-IDF features extracted from the review text field in a future iteration of the pipeline.

### 3.5 Handling Class Imbalance with SMOTE

A recurring challenge in clothing fit datasets is class imbalance: the majority of transactions are labeled fit, while small and large are underrepresented. Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling Technique), which addresses this by generating synthetic minority-class instances through interpolation in feature space rather than simple duplication. SMOTE has since demonstrated strong performance improvements for imbalanced tabular classification tasks across many domains. FitSense adopts SMOTE with the critical methodological safeguard of applying it exclusively within each training fold during cross-validation, ensuring that synthetic samples never appear in the validation set and that evaluation metrics reflect true generalization.

### 3.6 Summary

The surveyed literature converges on several themes central to FitSense: (i) body attributes carry strong predictive signal for fit outcomes; (ii) ensemble methods and Gradient Boosting consistently outperform simple baselines on tabular fit data; (iii) SMOTE is the standard and well-validated approach for correcting class imbalance in this domain; and (iv) review text holds latent signals that can augment structured tabular features. FitSense builds on these foundations by applying a rigorous, reproducible pipeline of classical classifiers to the benchmark UCSD dataset.

---

## References

[1] N.-Y. Dik, P. W. K. Tsang, A.-P. Chan, C. K. Y. Lo, and W.-C. Chu, "A novel approach in predicting virtual garment fitting sizes with psychographic characteristics and 3D body measurements using an artificial neural network and visualizing fitted bodies using a generative adversarial network," *Heliyon*, vol. 9, no. 7, p. e17916, 2023. https://doi.org/10.1016/j.heliyon.2023.e17916

[2] R. Misra, M. Wan, and J. McAuley, "Decomposing fit semantics for product size recommendation in metric spaces," in *Proc. 12th ACM Conf. Recommender Systems (RecSys)*, Vancouver, BC, Canada, 2018, pp. 439–447. https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18e.pdf

[3] K. A. K. Borgersen, M. Goodwin, M. Grundetjern, and J. Sharma, "A dataset for adapting recommender systems to the fashion rental economy," in *Proc. 18th ACM Conf. Recommender Systems (RecSys)*, Bari, Italy, 2024, pp. 945–950. https://doi.org/10.1145/3640457.3688175

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[5] R. A. Shetty et al., "Enhanced e-commerce decision-making through sentiment analysis using machine learning," *PMC*, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12208408/

[6] J. McAuley, "Recommender Systems Datasets — Clothing Fit," UC San Diego, 2018. https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit

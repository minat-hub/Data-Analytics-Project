# Cardiovascular Disease Prediction

This project applies the data analytics life cycle to a cardiovascular disease dataset. It includes data understanding, cleaning, exploratory data analysis, visualization, predictive modeling, and patient segmentation.

## Project Files

- `data_analytics_project.ipynb` - Main analysis notebook
- `cardiovascular_disease_prediction.csv` - Patient dataset
- `requirements.txt` - Python dependencies

## Objectives

1. Identify a target variable that can be predicted.
2. Audit the dataset for missing values, duplicates, data types, and distributions.
3. Clean and prepare the data for analysis.
4. Explore relationships between patient characteristics and cardiovascular disease.
5. Build at least two analytics models from the core model families:
   - Classification: logistic regression for disease prediction
   - Clustering: K-means for patient segmentation

## Dataset

The target variable is `Cardiovascular_Disease`, a binary indicator:

- `0` - No cardiovascular disease
- `1` - Cardiovascular disease

The dataset contains 20,000 patient records and 35 columns covering demographics, clinical measurements, lifestyle factors, and socioeconomic information. The positive disease rate is approximately 30.6%.

Several columns contain missing values, including BMI, HDL, LDL, triglycerides, blood glucose, physical activity, sleep hours, and diet quality. There are no duplicate rows in the source dataset.

## Data Preparation

- Exact duplicate rows are removed.
- `Patient_ID` is excluded from modeling because it is an identifier rather than a predictive patient characteristic.
- Numeric missing values are filled with the median.
- Categorical missing values are filled with the most frequent category.
- Numeric features are standardized.
- Categorical features are one-hot encoded.
- Preprocessing is performed inside scikit-learn pipelines to reduce data leakage.
- The data is split into training and test sets using an 80/20 stratified split.

## Exploratory Analysis

The notebook includes visualizations for:

- Target-class balance
- Age distribution by disease status
- Systolic blood pressure by disease status
- Risk score by disease status
- Disease prevalence across hypertension, diabetes, smoking, family history, and chest pain categories

The analysis suggests that patients with cardiovascular disease generally have higher age, blood pressure, and risk scores. These are associations in the dataset and should not be interpreted as causal conclusions.

## Models and Results

### Logistic Regression Classification

The logistic regression model predicts whether a patient has cardiovascular disease. Results on the held-out test set:

| Metric | Baseline | Outliers removed | SelectKBest (16 features) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.954 | 0.952 | 0.956 |
| Precision | 0.951 | 0.949 | 0.953 |
| Recall | 0.895 | 0.893 | 0.901 |
| ROC-AUC | 0.951 | 0.951 | 0.951 |

Precision means that 95.3% of patients predicted to have cardiovascular disease by the 16-feature model actually had the disease. Recall means that the model identified 90.1% of patients with disease. The remaining cases are false negatives.

Two logistic-regression variations were also tested:

- **Outlier removal:** IQR-based training-row removal excluded 816 observations. It slightly reduced accuracy, precision, and recall, so it did not improve performance.
- **SelectKBest:** after one-hot encoding categorical columns and scaling numeric columns, ANOVA `f_classif` selection retained 16 encoded features. It improved accuracy from 95.4% to 95.6%, precision from 95.1% to 95.3%, and recall from 89.5% to 90.1%. ROC-AUC remained 95.1%.

Overall, the 16-feature `SelectKBest` model is the preferred candidate in the held-out comparison because it achieved the best accuracy, precision, and recall.

#### Five-Fold Cross-Validation

Five-fold stratified cross-validation was also performed. Imputation, scaling, encoding, feature selection, and IQR outlier bounds were applied within each training fold to prevent leakage.

| Model | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.960 ± 0.004 | 0.958 ± 0.007 | 0.909 ± 0.009 | 0.958 ± 0.005 |
| Outliers removed | 0.958 ± 0.004 | 0.953 ± 0.006 | 0.909 ± 0.009 | 0.958 ± 0.005 |
| SelectKBest (16 features) | 0.962 ± 0.004 | 0.961 ± 0.007 | 0.914 ± 0.006 | 0.958 ± 0.005 |

The 16-feature `SelectKBest` model achieved the highest mean accuracy, precision, and recall across folds. It also had a recall standard deviation of 0.006, indicating relatively consistent detection of cardiovascular-disease cases. Outlier removal did not improve cross-validation performance. An external test dataset is still recommended before real-world or clinical use.

### K-Means Clustering

Three K-means feature representations were compared for $k=2$ through $k=5$: all numeric health features, 16 `SelectKBest` encoded features, and all preprocessed standardized features. The target variable was excluded from every clustering matrix.

| Feature set | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---:|---:|---:|---:|
| All numeric features | 0.074 | 0.057 | 0.044 | 0.042 |
| 16 SelectKBest features | 0.189 | 0.189 | 0.163 | 0.179 |
| All preprocessed features | 0.058 | 0.049 | 0.048 | 0.036 |

The 16 `SelectKBest` encoded features produced the strongest separation. The silhouette score was 0.189 for both $k=2$ and $k=3$; the implementation selected $k=2$ as the smaller cluster count for the tie.

The winning two-cluster result contains:

- **Cluster 0:** 13,912 patients, average blood glucose 93.59, average risk score 8.74, and 21.6% cardiovascular-disease prevalence.
- **Cluster 1:** 6,088 patients, average blood glucose 132.92, average risk score 12.37, and 51.1% cardiovascular-disease prevalence.

The 0.189 silhouette score indicates modest separation rather than sharply distinct patient groups. These segments are useful for exploratory patient profiling and are not diagnostic or causal results.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook with Jupyter:

```bash
jupyter notebook data_analytics_project.ipynb
```

Alternatively, open the notebook in VS Code with the Jupyter extension installed and run the cells from top to bottom.

## Limitations

- The dataset may not represent every population or clinical setting.
- Missing values are handled with median or most-frequent imputation inside modeling pipelines.
- The held-out results and 5-fold cross-validation are strong, but an external test dataset is still needed before real-world or clinical use.
- IQR outlier removal did not improve held-out or cross-validation performance.
- K-means cluster separation was modest, with the best silhouette score of 0.189. Clusters are descriptive, not diagnostic or causal.

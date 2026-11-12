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

| Metric | Score |
| --- | ---: |
| Accuracy | 0.954 |
| Precision | 0.951 |
| Recall | 0.895 |
| ROC-AUC | 0.951 |

Precision of 0.951 means that 95.1% of patients predicted to have cardiovascular disease actually had the disease. Recall of 0.895 means that the model identified 89.5% of patients who actually had the disease. The remaining cases are false negatives.

Overall, the classification model performs strongly on this dataset. In a health-screening context, recall is particularly important because missed disease cases may be more serious than false alarms.

### K-Means Clustering

K-means is used to group patients using standardized numeric health and lifestyle variables. The analysis tests values of `k` from 2 through 5 and selects `k = 2`, which has the highest silhouette score.

The selected silhouette score is 0.072, indicating weak separation between the clusters. The clusters show broad differences in age, blood pressure, blood glucose, and risk score, but they overlap substantially. These segments are useful for exploratory profiling and are not diagnostic categories.

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

- The classification metrics are based on one held-out test split and should be confirmed with cross-validation and external data.
- The dataset may not represent every population or clinical setting.
- Clustering has weak separation according to the silhouette score.

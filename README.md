# Credit Card Users Churn Prediction
This is project is from my PostGrad Program in AI/ML at UT Austin.

## 1. Business Objective
Thera Bank has experienced a decline in credit card users, which impacts revenue streams from various fees. The objective of this project is to analyze customer data and build a machine learning model to predict **customer attrition** (churn).

The model's goal is to identify at-risk customers *before* they leave, allowing the bank to take proactive retention measures.

## 2. The Data
* **Source:** `BankChurners.csv`
* **Target Variable:** `Attrition_Flag` (Categories: "Existing Customer" or "Attrited Customer")

## 3. Key Challenge: Imbalanced Data
A critical part of this project was addressing the highly imbalanced dataset. Exploratory Data Analysis (EDA) revealed that the target class is split:
* **83.9%** Existing Customers (Class 0)
* **16.1%** Attrited Customers (Class 1)

Because of this imbalance, **Recall** was chosen as the primary success metric. We must minimize **False Negatives** (predicting a customer will *stay* when they are actually *about to leave*), as this is the most costly error for the bank.

## 4. Tools & Technologies
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Preprocessing:** `scikit-learn` (OneHotEncoder, SimpleImputer)
* **Sampling:** `imblearn` (SMOTE, RandomUnderSampler)
* **Modeling:** `scikit-learn` (DecisionTree, RandomForest, GradientBoosting, AdaBoost, Bagging), `xgboost` (XGBClassifier)
* **Tuning & Evaluation:** `scikit-learn` (RandomizedSearchCV, recall_score, confusion_matrix)

## 5. Methodology
The project followed a structured approach to find the best-performing model, focusing on the high-recall-score.

1.  **Preprocessing:**
    * Missing values (`Education_Level`, `Marital_Status`, `Income_Category`) were imputed using the most frequent value.
    * Categorical features were one-hot encoded.
    * The target variable `Attrition_Flag` was binarized (1 for "Attrited," 0 for "Existing").
2.  **Data Splitting:** The data was split into **Train (50%)**, **Validation (30%)**, and **Test (20%)** sets.
3.  **Sampling Strategy:** To combat the class imbalance, I benchmarked models using three different datasets:
    * **Original Data**
    * **Oversampled Data** (using `SMOTE`)
    * **Undersampled Data** (using `RandomUnderSampler`)
4.  **Model Benchmarking:** Several tree-based ensembles (Random Forest, GBM, AdaBoost, XGBoost, etc.) were trained on all three datasets. The models trained on **Undersampled Data** provided the best and most generalizable recall scores on the validation set.
5.  **Hyperparameter Tuning:** The top-performing models (`RandomForest`, `GradientBoosting`, `XGBoost`) were then hyperparameter-tuned using `RandomizedSearchCV` on the undersampled training data to maximize recall.

## 6. Final Model & Results
The best and final model was a **tuned GradientBoostingClassifier** trained on the undersampled data.

This model demonstrated strong generalization from the validation set to the final, unseen test set.

### Test Set Performance:
| Metric | Score |
| :--- | :--- |
| **Recall** | **0.840** |
| Accuracy | 0.816 |
| Precision | 0.364 |
| F1-Score | 0.508 |

### Key Feature Importances:
The model identified that customer transaction behavior is the strongest predictor of churn.
1.  **Total_Trans_Amt** (Total Transaction Amount)
2.  **Total_Trans_Ct** (Total Transaction Count)
3.  **Total_Revolving_Bal** (Total Revolving Balance)

## 7. Business Insights
The model and its feature importances confirm a clear insight: **customers who stop using their card are highly likely to churn.**

Attrited customers have significantly lower `Total_Trans_Amt` and `Total_Trans_Ct`. To reduce churn, the bank should focus on creating proactive campaigns to incentivize card usage, such as offering rewards, points, or low-interest periods for customers whose transaction frequency has dropped.

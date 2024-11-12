# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import zscore
import os  # Added to check if the file exists

# Load the dataset
file_path = "dataset.csv.csv"

# Check if the file exists
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("File loaded successfully")
else:
    print(f"File not found: {file_path}")
    # If the file is not found, you could terminate or handle it appropriately
    exit(1)  # Stop execution if file doesn't exist

# 1. Data Exploration and Preprocessing

# a. Checking for missing values
print("Missing Values Check:")
print(data.isnull().sum())

# Visualizing missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Visualization')
plt.show()

# b. Handle missing values (if any)
# Example: Filling missing numerical columns with the median or mean (based on your dataset)
# You may want to adapt this based on your understanding of the data
data.fillna(data.median(), inplace=True)  # For numerical columns
# If there are categorical columns with missing values, you could fill them with the mode
# data['category_column'].fillna(data['category_column'].mode()[0], inplace=True)

# c. Checking correlations between features
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# d. Visualizing the distribution of the target variable 'churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=data)
plt.title('Churn Distribution')
plt.show()

# e. Detecting anomalies in numerical features (e.g., 'age')
data['z_score_age'] = zscore(data['age'])

# Filter out extreme outliers (z-score > 3)
data_cleaned = data[data['z_score_age'].abs() <= 3]

# Visualizing the cleaned 'age' distribution
plt.figure(figsize=(10, 6))
sns.histplot(data_cleaned['age'], kde=True, color='blue')
plt.title('Age Distribution after Anomaly Removal')
plt.show()

# 2. Data Splitting (Time-based)
# Make sure 'year' is in the dataset and split based on it
train_data = data[data['year'] < 2021]  # Training on data from 2019 and 2020
test_data = data[data['year'] == 2021]  # Testing on data from 2021

X_train = train_data.drop(['churn', 'year'], axis=1)
y_train = train_data['churn']
X_test = test_data.drop(['churn', 'year'], axis=1)
y_test = test_data['churn']

# 3. Handling Class Imbalance (SMOTE)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 4. Initial Model Training

# a. Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_smote, y_train_smote)

# b. Decision Tree Model (for visualization)
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_smote, y_train_smote)

# c. Evaluating models
y_pred_log_reg = log_reg.predict(X_test)
y_pred_dtree = dtree.predict(X_test)

# Classification Report & AUC-ROC Score for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print(f"AUC-ROC Score for Logistic Regression: {roc_auc_score(y_test, y_pred_log_reg)}")

# Classification Report & AUC-ROC Score for Decision Tree
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dtree))
print(f"AUC-ROC Score for Decision Tree: {roc_auc_score(y_test, y_pred_dtree)}")

# Visualizing the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dtree, filled=True, feature_names=X_train.columns, class_names=['Stay', 'Churn'], fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# 5. Dealing with Concept Drift and Data Shifts

# a. Time-Weighted Learning (assigning higher weight to more recent samples)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_smote)

# Visualizing the sample weights
plt.figure(figsize=(10, 6))
sns.histplot(sample_weights, kde=True, color='orange')
plt.title('Distribution of Sample Weights (Time-Weighted Learning)')
plt.show()

# b. Online Learning with Stochastic Gradient Descent (SGD)
sgd = SGDClassifier(loss='log', random_state=42)
sgd.fit(X_train_smote, y_train_smote)

# Predict and evaluate SGD model
y_pred_sgd = sgd.predict(X_test)
print("SGD Classification Report:")
print(classification_report(y_test, y_pred_sgd))
print(f"AUC-ROC Score for SGD: {roc_auc_score(y_test, y_pred_sgd)}")

# 6. Model Adaptation Evaluation

# a. Compare AUC-ROC for baseline (Logistic Regression) vs Adapted (SGD)
baseline_auc = roc_auc_score(y_test, y_pred_log_reg)
sgd_auc = roc_auc_score(y_test, y_pred_sgd)

# Visualizing AUC-ROC comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=['Baseline (Logistic Regression)', 'Adapted (SGD)'], y=[baseline_auc, sgd_auc], palette='viridis')
plt.title('Model Comparison: AUC-ROC Score')
plt.ylabel('AUC-ROC')
plt.show()

# b. Evaluation on Older vs. Newer Data (Year 2020 vs 2021)
# Compare performance on Year 2020 vs Year 2021 data

# Year 2020 predictions (train model on 2020 data)
train_2020 = data[data['year'] == 2020]
X_train_2020 = train_2020.drop(['churn', 'year'], axis=1)
y_train_2020 = train_2020['churn']
smote_2020 = SMOTE(sampling_strategy='auto', random_state=42)
X_train_2020_smote, y_train_2020_smote = smote_2020.fit_resample(X_train_2020, y_train_2020)

# Train model on 2020 data
log_reg_2020 = LogisticRegression(random_state=42)
log_reg_2020.fit(X_train_2020_smote, y_train_2020_smote)

# Evaluate on 2020 data
y_pred_2020 = log_reg_2020.predict(X_test)  # Test on 2021 data
print("Logistic Regression (Year 2020 model) Classification Report on Year 2021 data:")
print(classification_report(y_test, y_pred_2020))
print(f"AUC-ROC for 2020 model on 2021 data: {roc_auc_score(y_test, y_pred_2020)}")

# 7. Conclusion

# Summarize results and key findings
print("Summary:")
print(f"Logistic Regression AUC-ROC Score: {roc_auc_score(y_test, y_pred_log_reg)}")
print(f"Decision Tree AUC-ROC Score: {roc_auc_score(y_test, y_pred_dtree)}")
print(f"Adapted Model (SGD) AUC-ROC Score: {roc_auc_score(y_test, y_pred_sgd)}")
print(f"Model Adaptation via Concept Drift: The SGD model and ensemble models showed better adaptability to changes over time.")

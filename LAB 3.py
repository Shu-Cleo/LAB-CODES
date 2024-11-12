# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Collection

# Load Titanic dataset from Seaborn
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 2: Data Cleaning

# Inspect for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Dataset:\n", missing_values)

# Handle missing values
# For 'age', impute the missing values with the median age
df['age'] = df['age'].fillna(df['age'].median())

# For 'embarked', impute the missing values with the mode (most frequent value)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Display the dataset after handling missing values
missing_values_after = df.isnull().sum()
print("\nMissing Values After Imputation:\n", missing_values_after)

# Step 3: Handling Outliers

# Identify outliers in the 'age' and 'fare' columns using box plots
plt.figure(figsize=(12, 6))

# Boxplot for 'age'
plt.subplot(1, 2, 1)
sns.boxplot(x=df['age'])
plt.title('Boxplot for Age')

# Boxplot for 'fare'
plt.subplot(1, 2, 2)
sns.boxplot(x=df['fare'])
plt.title('Boxplot for Fare')

plt.tight_layout()
plt.show()

# Remove outliers in the 'fare' column (values beyond the 99th percentile)
fare_99th_percentile = df['fare'].quantile(0.99)
df = df[df['fare'] <= fare_99th_percentile]

# Cap outliers in the 'age' column at the 95th percentile
age_95th_percentile = df['age'].quantile(0.95)
df['age'] = df['age'].apply(lambda x: min(x, age_95th_percentile))

# Box plots after removing and capping outliers
plt.figure(figsize=(12, 6))

# Boxplot for 'age'
plt.subplot(1, 2, 1)
sns.boxplot(x=df['age'])
plt.title('Boxplot for Age (After Handling Outliers)')

# Boxplot for 'fare'
plt.subplot(1, 2, 2)
sns.boxplot(x=df['fare'])
plt.title('Boxplot for Fare (After Handling Outliers)')

plt.tight_layout()
plt.show()

# Step 4: Data Normalization

# Normalize the 'age' and 'fare' columns using Min-Max Scaling
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Step 5: Feature Engineering

# Create a new 'family_size' column by summing 'sibsp' and 'parch'
df['family_size'] = df['sibsp'] + df['parch']

# Create a new 'title' column extracted from the 'name' column
df['title'] = df['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Display the dataset with new features
print("\nNew Features (Family Size and Title):")
print(df[['family_size', 'title']].head())

# Step 6: Feature Selection

# Correlation matrix to select important features for the model
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Feature Importance using Random Forest Classifier
X = df[['age', 'fare', 'family_size', 'pclass']]
y = df['survived']

# Train a Random Forest model to get feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance from Random Forest:\n", feature_importances)

# Step 7: Model Building

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Step 8: Model Evaluation

# Calculate various evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Display classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



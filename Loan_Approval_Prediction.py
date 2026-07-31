# Import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("./dataset/loan_approval_dataset.csv")
# df.head()
# df.info()
# df.describe()

df.columns = df.columns.str.strip()
# df.isnull().sum()

df["loan_status"].value_counts()

# Remove unnecessary spaces from categorical values
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# print(df["education"].unique())
# print(df["self_employed"].unique())
# print(df["loan_status"].unique())

# Convert categorical values into numerical values
le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])
df["self_employed"] = le.fit_transform(df["self_employed"])
df["loan_status"] = le.fit_transform(df["loan_status"])
df.head()


# Encoded Values
# Education:
# 0 = Graduate
# 1 = Not Graduate

# Self Employed:
# 0 = No
# 1 = Yes

# Loan Status:
# 0 = Approved
# 1 = Rejected

# Separate input features (X) and target variable (Y)
X = df.drop("loan_status", axis=1)
Y = df["loan_status"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=42) 

# Standardize numerical features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create the Logistic Regression model


# Train the model using training data
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, Y_train)

# Predict loan status for test data
Y_pred = logisticRegression.predict(X_test)

# Evaluating Model
print(f"Accuracy Score:{ accuracy_score(Y_test, Y_pred) * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nDetailed Report:\n", classification_report(Y_test, Y_pred))



# Confusion Matrix heatmap
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
           annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Approved', 'Rejected'],
            yticklabels=['Approved','Rejected']
           )

plt.xlabel("Predected")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Loan Status Distribution
plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x='loan_status',
    hue='loan_status',
    palette='Set2',
    legend=False
)

plt.xticks([0,1], ['Approved','Rejected'])
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")

plt.show()


# Education vs Loan Status
plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x='education',
    hue='loan_status',
    palette='viridis'
)

plt.xticks([0,1], ['Graduate','Not Graduate'])
plt.legend(['Approved','Rejected'])

plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.show()

# Annual Income Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x='income_annum',
    bins=50,
    kde=True,
    color='orange'
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")

plt.show()



#Feature Importance
coef = pd.Series(logisticRegression.coef_[0], index=X.columns)

coef = coef.sort_values()

plt.figure(figsize=(14,8))
coef.plot(kind='barh')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.show()
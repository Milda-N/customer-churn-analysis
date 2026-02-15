# Customer Churn Analysis (EDA + ML)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("../data/churn_data.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Basic churn rate
churn_rate = df["Churn"].value_counts(normalize=True) * 100
print("\nChurn Rate (%):")
print(churn_rate)

# Churn by Contract Type
contract_churn = pd.crosstab(df["ContractType"], df["Churn"], normalize="index") * 100
print("\nChurn by Contract Type (%):")
print(contract_churn)

contract_churn["Yes"].plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn %")
plt.show()

# Average tenure by churn
tenure_analysis = df.groupby("Churn")["Tenure"].mean()
print("\nAverage Tenure by Churn:")
print(tenure_analysis)

df.boxplot(column="Tenure", by="Churn")
plt.title("Tenure Distribution by Churn")
plt.suptitle("")
plt.show()

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn_Yes", axis=1)
y = df_encoded["Churn_Yes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

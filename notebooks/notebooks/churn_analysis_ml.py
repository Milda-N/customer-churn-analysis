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


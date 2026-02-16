import pandas as pd
import matplotlib.pyplot as plt
import os

print("STARTED")

# Load dataset
df = pd.read_csv(r"C:\Users\milda\Downloads\customer-churn-analysis-main\customer-churn-analysis-main\data\churn_data.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ===== FIRST GRAPH =====
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")

os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/churn_distribution.png")

plt.show()

# ===== SECOND GRAPH =====
plt.figure()

contract_churn = pd.crosstab(
    df["ContractType"], df["Churn"], normalize="index"
) * 100

contract_churn["Yes"].plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn %")

plt.savefig("visuals/churn_by_contract.png")

plt.show()

input("Press Enter to exit...")

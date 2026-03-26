import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleaned_house_data.csv")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution of the target variable (house prices)
plt.hist(df["MEDV"], bins=20)
plt.title("Distribution of House Prices")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.show()

# boxplot to identify outliers in the target variable
sns.boxplot(x=df["RM"])
plt.title("Boxplot of Average Rooms")
plt.show()


#scatter plot to visualize the relationship between average rooms and house prices
plt.scatter(df["RM"], df["MEDV"])
plt.title("Rooms vs House Price")
plt.xlabel("Average Rooms")
plt.ylabel("Median Value")
plt.show()

# Correlation heatmap
corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
